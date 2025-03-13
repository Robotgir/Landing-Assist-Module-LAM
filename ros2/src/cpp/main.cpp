#include <iostream>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <pcl/io/pcd_io.h>
#include "hazard_metrices.h"
#include "pointcloud_preprocessing.h"
#include "../../../lib/common/common.h"


using PointT = pcl::PointXYZI;

int main(int argc, char **argv) {

    std::string config_file ="/home/airsim_user/Landing-Assist-Module-LAM/config/pipeline.yaml";
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
    PCLResult result;
    result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    auto [loaded_cloud, performDownsampling] = loadPCLCloud<PointT>(pcd_file);

    PCLResult final_result;
    final_result.outlier_cloud = loaded_cloud;
    
    if (performDownsampling && voxel_downsample_pointcloud) {
        // Downsample if necessary (here, using a voxel size of 0.45 as example).
        downsamplePointCloudPCL<PointT>(loaded_cloud, result.downsampled_cloud, voxelSize);
        std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
    } else {
        result.downsampled_cloud = loaded_cloud;
    }
    // Start with the downsampled cloud.
    pcl::PointCloud<PointT>::Ptr current_cloud = result.downsampled_cloud;

    // Get the pipeline configuration.
    YAML::Node pipeline = params["pipeline"];

    // Process each step in the pipeline sequentially.
    for (std::size_t i = 0; i < pipeline.size(); ++i) {
        std::string step = pipeline[i]["step"].as<std::string>();
        bool enabled = pipeline[i]["enabled"].as<bool>();
        if (!enabled) {
            std::cout << "\n--- Skipping disabled step: " << step << " ---\n";
            continue;
        }
        std::cout << "\n--- Running pipeline step: " << step << " ---\n";

        if (step == "SOR") {
            int nb_neighbors = pipeline[i]["parameters"]["nb_neighbors"].as<int>();
            double std_ratio = pipeline[i]["parameters"]["std_ratio"].as<double>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            // Use the file-path version for SOR.
            OPEN3DResult sor_result = apply_sor_filter(pcd_file, nb_neighbors, std_ratio);
            PCLResult pcl_cloud = convertOpen3DToPCL(sor_result);
            if (g_visualize) {
                visualizeOPEN3D(sor_result, visualization);
            }
            current_cloud = pcl_cloud.inlier_cloud;
        }
        else if (step == "Radial") {
            double voxel_size = pipeline[i]["parameters"]["voxel_size"].as<double>();
            double radius_search = pipeline[i]["parameters"]["radius_search"].as<double>();
            int min_neighbors = pipeline[i]["parameters"]["min_neighbors"].as<int>();

            PCLResult radial_result = applyRadiusFilter(pcd_file, voxel_size, radius_search, min_neighbors);
            if (g_visualize) {
                visualizePCL(radial_result);
            }
            current_cloud = radial_result.inlier_cloud;
        }
        else if (step == "Bilateral") {
            double voxel_size = pipeline[i]["parameters"]["voxel_size"].as<double>();
            double sigma_s = pipeline[i]["parameters"]["sigma_s"].as<double>();
            double sigma_r = pipeline[i]["parameters"]["sigma_r"].as<double>();

            PCLResult bilateral_result = applyBilateralFilter(pcd_file, voxel_size, sigma_s, sigma_r);
            if (g_visualize) {
                visualizePCL(bilateral_result);
            }
            current_cloud = bilateral_result.inlier_cloud;
        }
        else if (step == "RANSAC_OPEN3D") {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int ransac_n = pipeline[i]["parameters"]["ransac_n"].as<int>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            // Wrap the current PCL cloud into a temporary PCLResult.
            PCLResult temp;
            temp.inlier_cloud = current_cloud;
            // Convert to Open3DResult.
            OPEN3DResult open3d_temp = convertPCLToOpen3D(temp);
            // Wrap the resulting Open3D cloud (here using the inlier cloud) into a variant.
            Open3DCloudInput open3d_input = open3d_temp.inlier_cloud;
            OPEN3DResult ransac_result = RansacPlaneSegmentation(open3d_input, voxel_size, distanceThreshold, ransac_n, maxIterations);
            PCLResult pcl_cloud = convertOpen3DToPCL(ransac_result);
            current_cloud = pcl_cloud.inlier_cloud;
            if (g_visualize) {
                visualizePCL(pcl_cloud, visualization);
            }
        }
        else if (step == "LeastSquare") {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult temp;
            temp.inlier_cloud = current_cloud;
            OPEN3DResult open3d_temp = convertPCLToOpen3D(temp);
            Open3DCloudInput open3d_input = open3d_temp.inlier_cloud;
            OPEN3DResult least_square_result = LeastSquaresPlaneFitting(open3d_input, voxel_size, distanceThreshold);
            PCLResult pcl_cloud = convertOpen3DToPCL(least_square_result);
            current_cloud = pcl_cloud.inlier_cloud;
            if (g_visualize) {
                visualizePCL(pcl_cloud, visualization);
            }
        }
        else if (step == "PROSAC") {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult prosac_result = performPROSAC(current_cloud, voxel_size, distanceThreshold, maxIterations);
            current_cloud = prosac_result.inlier_cloud;
            if (g_visualize) {
                visualizePCL(prosac_result, visualization);
            }
        }
        else if (step == "RANSAC") {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult ransac_result = performRANSAC(current_cloud, voxel_size, distanceThreshold, maxIterations);
            current_cloud = ransac_result.inlier_cloud;
            if (g_visualize) {
                visualizePCL(ransac_result, visualization);
            }
        }
        else if (step == "LMEDS") {
            float voxel_size = pipeline[i]["parameters"]["voxel_size"].as<float>();
            float distanceThreshold = pipeline[i]["parameters"]["distanceThreshold"].as<float>();
            int maxIterations = pipeline[i]["parameters"]["maxIterations"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult lmeds_result = performLMEDS(current_cloud, voxel_size, distanceThreshold, maxIterations);
            current_cloud = lmeds_result.inlier_cloud;
            if (g_visualize) {
                visualizePCL(lmeds_result, visualization);
            }
        }
        else if (step == "Average3DGradient") {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            float neighborRadius = pipeline[i]["parameters"]["neighborRadius"].as<float>();
            float gradientThreshold = pipeline[i]["parameters"]["gradientThreshold"].as<float>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult avgGrad_result = Average3DGradient(current_cloud, voxelSize, neighborRadius, gradientThreshold, angleThreshold);
            current_cloud = avgGrad_result.inlier_cloud;
            if (g_visualize) {
                visualizePCL(avgGrad_result, visualization);
            }
        }
        else if (step == "RegionGrowing") {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            int min_cluster_size = pipeline[i]["parameters"]["min_cluster_size"].as<int>();
            int max_cluster_size = pipeline[i]["parameters"]["max_cluster_size"].as<int>();
            int number_of_neighbours = pipeline[i]["parameters"]["number_of_neighbours"].as<int>();
            int normal_k_search = pipeline[i]["parameters"]["normal_k_search"].as<int>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult regionGrowingResult = regionGrowingSegmentation(current_cloud, voxelSize, angleThreshold, min_cluster_size, max_cluster_size, number_of_neighbours, normal_k_search);
            current_cloud = regionGrowingResult.inlier_cloud;
            if (g_visualize) {
                visualizePCL(regionGrowingResult, visualization);
            }
        }
        else if (step == "PCA") {
            float voxelSize = pipeline[i]["parameters"]["voxelSize"].as<float>();
            int k = pipeline[i]["parameters"]["k"].as<int>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();

            PCLResult PCA_Result = PrincipleComponentAnalysis(current_cloud, voxelSize, angleThreshold, k);
            current_cloud = PCA_Result.inlier_cloud;
            if (g_visualize) {
                visualizePCL(PCA_Result, visualization);
            }
        }else if (step =="CheckSLZ-r"){
            double landingRadius = pipeline[i]["parameters"]["landingRadius"].as<double>();
            double removalThreshold = pipeline[i]["parameters"]["removalThreshold"].as<double>();
            auto [environment_cloud, performDownsampling] = loadPCLCloud<PointT>(pcd_file);
            std::string visualization = pipeline[i]["parameters"]["visualization"].as<std::string>();
            PCLResult SLZResult = findSafeLandingZones(current_cloud,
                                                       pcd_file,
                                                       landingRadius,
                                                       removalThreshold);
            current_cloud = SLZResult.inlier_cloud;
            if (g_visualize) {
                visualizePCL(SLZResult, visualization);
            }
        }
        else {
            std::cerr << "Unknown pipeline step: " << step << std::endl;
            return -1;
        }
    }

    // Final visualization of the chained output.
    if (g_visualize || final_visualize) {
        std::cout << "\n--- Final Processed Cloud ---\n";
        
        final_result.inlier_cloud = current_cloud;
        visualizePCL(final_result,final_result_visualization);
    }

    return 0;
}
