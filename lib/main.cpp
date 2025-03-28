#include <iostream>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <pcl/io/pcd_io.h>
#include "hazard_metrices.h"
#include "pointcloud_preprocessing.h"
#include "common.h"
#include "architecture.h"

using PointT = pcl::PointXYZI;

int main(int argc, char **argv)
{

    std::string config_file = "/home/airsim_user/Landing-Assist-Module-LAM/lib/config/pipeline2Octree.yaml";
    // Load YAML configuration.
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node params = config["ros__parameters"];

    // Set file paths.

    std::string pcd_file = params["pcd_file_path"].as<std::string>();

    bool g_visualize = params["visualize"].as<bool>();
    bool final_visualize = params["final_visualize"].as<bool>();
    bool voxel_downsample_pointcloud = params["voxel_downsample_pointcloud"].as<bool>();
    std::string final_result_visualization = params["final_result_visualization"].as<std::string>();
    float voxelSize = params["voxel_size"].as<float>();

    // Load input point cloud using the provided loadPCLCloud function.
    PCLResult pclResult;
    pclResult.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    pclResult.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    auto loaded_cloud = loadPCLCloud<PointT>(pcd_file);

    PCLResult final_result;
    final_result.outlier_cloud = loaded_cloud;

    if (voxel_downsample_pointcloud)
    {
        // Downsample if necessary (here, using a voxel size of 0.45 as example).
        downsamplePointCloudPCL<PointT>(loaded_cloud, pclResult.inlier_cloud, voxelSize);
        std::cout << "Downsampled cloud has " << pclResult.inlier_cloud->points.size() << " points." << std::endl;
    }
    else
    {
        // pclResult.downsampled_cloud = loaded_cloud;
        pclResult.inlier_cloud = loaded_cloud;
    }
    // Start with the downsampled cloud.
    pcl::PointCloud<PointT>::Ptr current_cloud = pclResult.inlier_cloud;

    pcl::PointCloud<PointT>::Ptr sor_result;

    // Variable to store safe landing zones
    //  std::vector<typename pcl::PointCloud<PointT>::Ptr>> slz;

    // Get the pipeline configuration.
    YAML::Node pipeline = params["pipeline"];

    // Process each step in the pipeline sequentially.
    for (std::size_t i = 0; i < pipeline.size(); ++i)
    {
        std::string step = pipeline[i]["step"].as<std::string>();
        bool enabled = pipeline[i]["enabled"].as<bool>();
        if (!enabled)
        {
            std::cout << "\n--- Skipping disabled step: " << step << " ---\n";
            continue;
        }
        std::cout << "\n--- Running pipeline step: " << step << " ---\n";

        if (step == "SOR")
        {
            int nb_neighbors = pipeline[i]["parameters"]["nb_neighbors"].as<int>();
            double std_ratio = pipeline[i]["parameters"]["std_ratio"].as<double>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            // Use the file-path version for SOR.
            OPEN3DResult result = apply_sor_filter(pcd_file, nb_neighbors, std_ratio);
            auto pclResult = convertOpen3DToPCL(result);
            // sor_result = pcl_cloud.inlier_cloud;

            if (g_visualize)
            {
                visualizePCL(pclResult, visualization);
            }
            // current_cloud = pcl_cloud.inlier_cloud;

        }
        else if (step == "Radial")
        {
            double voxel_size = pipeline[i]["parameters"]["voxel_size"].as<double>();
            double radius_search = pipeline[i]["parameters"]["radius_search"].as<double>();
            int min_neighbors = pipeline[i]["parameters"]["min_neighbors"].as<int>();

            PCLResult radial_result = applyRadiusFilter(pcd_file, voxel_size, radius_search, min_neighbors);
            if (g_visualize)
            {
                visualizePCL(radial_result);
            }
            // current_cloud = radial_result.inlier_cloud;
            pclResult.inlier_cloud = radial_result.inlier_cloud;
        }
        else if (step == "Bilateral")
        {
            double voxel_size = pipeline[i]["parameters"]["voxel_size"].as<double>();
            double sigma_s = pipeline[i]["parameters"]["sigma_s"].as<double>();
            double sigma_r = pipeline[i]["parameters"]["sigma_r"].as<double>();

            PCLResult bilateral_result = applyBilateralFilter(pcd_file, voxel_size, sigma_s, sigma_r);
            if (g_visualize)
            {
                visualizePCL(bilateral_result);
            }
            // current_cloud = bilateral_result.inlier_cloud;
            pclResult.inlier_cloud = bilateral_result.inlier_cloud;
        }
        else if (step == "2dGridmap"){
            float resolution = pipeline[i]["parameters"]["resolution"].as<float>();
            pclResult.inlier_cloud = create2DGridMap(pclResult.inlier_cloud,resolution);
        }
        else if (step == "RANSAC_OPEN3D")
        {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int ransac_n = pipeline[i]["parameters"]["ransac_n"].as<int>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            // Wrap the current PCL cloud into a temporary PCLResult.
            PCLResult temp;
            // temp.inlier_cloud = current_cloud;
            temp.inlier_cloud = pclResult.inlier_cloud;
            // Convert to Open3DResult.
            OPEN3DResult open3d_temp = convertPCLToOpen3D(temp);
            // Wrap the resulting Open3D cloud (here using the inlier cloud) into a variant.
            Open3DCloudInput open3d_input = open3d_temp.inlier_cloud;
            OPEN3DResult ransac_result = RansacPlaneSegmentation(open3d_input, voxel_size, distanceThreshold, ransac_n, maxIterations);
            PCLResult pcl_cloud = convertOpen3DToPCL(ransac_result);
            // current_cloud = pcl_cloud.inlier_cloud;
            pclResult.inlier_cloud = pcl_cloud.inlier_cloud;
        
            if (g_visualize)
            {
                visualizePCL(pcl_cloud, visualization);
            }
        }
        else if (step == "LeastSquare")
        {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult temp;
            // temp.inlier_cloud = current_cloud;
            temp.inlier_cloud = pclResult.inlier_cloud;
            OPEN3DResult open3d_temp = convertPCLToOpen3D(temp);
            Open3DCloudInput open3d_input = open3d_temp.inlier_cloud;
            OPEN3DResult least_square_result = LeastSquaresPlaneFitting(open3d_input, voxel_size, distanceThreshold);

            PCLResult pcl_cloud = convertOpen3DToPCL(least_square_result);
            pclResult.inlier_cloud = pcl_cloud.inlier_cloud;
            pclResult.plane_coefficients = pcl_cloud.plane_coefficients;
            if (g_visualize)
            {
                visualizePCL(pcl_cloud, visualization);
            }
        }
        else if (step == "PROSAC")
        {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult prosac_result = performPROSAC(pclResult.inlier_cloud, voxel_size, distanceThreshold, maxIterations);
            pclResult.inlier_cloud = prosac_result.inlier_cloud;
            pclResult.plane_coefficients = prosac_result.plane_coefficients;
            
            if (g_visualize)
            {
                visualizePCL(prosac_result, visualization);
            }
        }
        else if (step == "RANSAC")
        {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult ransac_result = performRANSAC(pclResult.inlier_cloud, voxel_size, distanceThreshold, maxIterations);
            pclResult.inlier_cloud = ransac_result.inlier_cloud;
            pclResult.plane_coefficients = ransac_result.plane_coefficients;
            
            if (g_visualize)
            {
                visualizePCL(ransac_result, visualization);
            }
        }
        else if (step == "LMEDS")
        {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult lmeds_result = performLMEDS(pclResult.inlier_cloud, voxel_size, distanceThreshold, maxIterations);
            pclResult.inlier_cloud = lmeds_result.inlier_cloud;
            pclResult.plane_coefficients = lmeds_result.plane_coefficients;
            
            if (g_visualize)
            {
                visualizePCL(lmeds_result, visualization);
            }
        }
        else if (step == "Average3DGradient")
        {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            float neighborRadius = pipeline[i]["parameters"]["neighborRadius"].as<float>();
            float gradientThreshold = pipeline[i]["parameters"]["gradientThreshold"].as<float>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult avgGrad_result = Average3DGradient(pclResult.inlier_cloud, voxelSize, neighborRadius, gradientThreshold, angleThreshold);
            pclResult.inlier_cloud = avgGrad_result.inlier_cloud;
            
            if (g_visualize)
            {
                visualizePCL(avgGrad_result, visualization);
            }
        }
        else if (step == "RegionGrowing")
        {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            int min_cluster_size = pipeline[i]["parameters"]["min_cluster_size"].as<int>();
            int max_cluster_size = pipeline[i]["parameters"]["max_cluster_size"].as<int>();
            int number_of_neighbours = pipeline[i]["parameters"]["number_of_neighbours"].as<int>();
            int normal_k_search = pipeline[i]["parameters"]["normal_k_search"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult regionGrowingResult = regionGrowingSegmentation(pclResult.inlier_cloud, voxelSize, angleThreshold, min_cluster_size, max_cluster_size, number_of_neighbours, normal_k_search);
            pclResult.inlier_cloud = regionGrowingResult.inlier_cloud;
            if (g_visualize)
            {
                visualizePCL(regionGrowingResult, visualization);
            }
        }
        else if (step == "RegionGrowing2")
        {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            int min_cluster_size = pipeline[i]["parameters"]["min_cluster_size"].as<int>();
            int max_cluster_size = pipeline[i]["parameters"]["max_cluster_size"].as<int>();
            int number_of_neighbours = pipeline[i]["parameters"]["number_of_neighbours"].as<int>();
            int normal_k_search = pipeline[i]["parameters"]["normal_k_search"].as<int>();
            float curvature_threshold = pipeline[i]["parameters"]["curvature_threshold"].as<float>();
            float height_threshold = pipeline[i]["parameters"]["height_threshold"].as<float>();

            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult regionGrowingResult = regionGrowingSegmentation2(pclResult.inlier_cloud, voxelSize, angleThreshold, min_cluster_size, max_cluster_size, number_of_neighbours, normal_k_search, curvature_threshold, height_threshold);
            // PCLResult regionGrowingResult = calculateNormalsAndCurvature(pclResult.inlier_cloud);
            pclResult.inlier_cloud = regionGrowingResult.inlier_cloud;
            if (g_visualize)
            {
                visualizePCL(regionGrowingResult, visualization);
            }
        }
        else if (step == "PCA")
        {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            int k = pipeline[i]["parameters"]["k"].as<int>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult PCA_Result = PrincipleComponentAnalysis(pclResult.inlier_cloud, voxelSize, angleThreshold, k);
        
            pclResult.inlier_cloud = PCA_Result.inlier_cloud;
            pclResult.plane_coefficients = PCA_Result.plane_coefficients;
            
            if (g_visualize)
            {
                visualizePCL(PCA_Result, visualization);
            }
        }
        else if (step == "CheckSLZ-r")
        {
            double landingRadius = pipeline[i]["parameters"]["landingRadius"].as<double>();
            double removalThreshold = pipeline[i]["parameters"]["removalThreshold"].as<double>();
            double clusterTolerance = pipeline[i]["parameters"]["clusterTolerance"].as<double>();
            auto environment_cloud = loadPCLCloud<PointT>(pcd_file);
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();
            PCLResult SLZResult = findSafeLandingZones(pclResult.inlier_cloud,
                                                       environment_cloud,
                                                       landingRadius,
                                                       removalThreshold,
                                                       clusterTolerance);
            pclResult.inlier_cloud = SLZResult.inlier_cloud;
            if (g_visualize)
            {
                visualizePCL(SLZResult, visualization);
            }
        }
        else if (step == "SphericalNeighbourhood")
        {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            int k = pipeline[i]["parameters"]["k"].as<int>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            double radius = pipeline[i]["parameters"]["radius"].as<double>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();
            int landingZoneNumber = pipeline[i]["parameters"]["landingZoneNumber"].as<int>();
            int maxAttempts = pipeline[i]["parameters"]["maxAttempts"].as<int>();
            
            
            PCLResult result;
     
        

            std::vector<SLZDCandidatePoints> candidatePoints;
            std::tie(result, candidatePoints) =kdtreeNeighborhoodPCAFilter(pclResult.inlier_cloud,
                                            radius, voxelSize, k, angleThreshold,
                                            landingZoneNumber, maxAttempts);
            
            
            // candidatePoints.push_back(finalCandidate);
            // Add plane coeffiecient to the struct we gonaa pass to calculate roughness
            result.plane_coefficients = pclResult.plane_coefficients;
            auto rankedCandidates = rankCandidatePatches(candidatePoints, result);
           
           
            pclResult.inlier_cloud = result.inlier_cloud;
            if (g_visualize)
            {
                // visualizePCL(result, visualization);
                visualizeRankedCandidatePatches(rankedCandidates, result);
                
            }
        }
        else if(step == "HazarMetrices"){
            std::string hazardMetricsName = pipeline[i]["parameters"]["hazard"].as<std::string>();
            auto hazard = rankCandidatePatchFromPCLResult(pclResult, hazardMetricsName);
        }
        else
        {
            std::cerr << "Unknown pipeline step: " << step << std::endl;
            return -1;
        }
    }

    // Final visualization of the chained output.
    if (g_visualize || final_visualize)
    {
        std::cout << "\n--- Final Processed Cloud ---\n";

        final_result.inlier_cloud = pclResult.inlier_cloud;
        visualizePCL(final_result, final_result_visualization);
    }

    return 0;
}
