#include <iostream>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"
#include <pcl/io/pcd_io.h>
#include "hazard_metrices.h"
#include "pointcloud_preprocessing.h"
#include "common.h"
#include "architecture.h"
#include <chrono>

using Point_pcl = pcl::PointXYZI;

int main(int argc, char **argv)
{

    std::string config_file = "/home/airsim_user/Landing-Assist-Module-LAM/lib/config/pipeline2_prosac.yaml";
    // Load YAML configuration.
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node params = config["parameters"];

    // Set file paths.
    std::string cloud_file_path = params["cloud_file_path"].as<std::string>();

    bool viz_individual_method = params["viz_individual_method"].as<bool>();
    bool final_visualize = params["final_visualize"].as<bool>();
    bool voxel_downsample_pointcloud = params["voxel_downsample_pointcloud"].as<bool>();
    std::string viz_inlier_or_outlier_or_both_final = params["viz_inlier_or_outlier_or_both_final"].as<std::string>();
    float voxel_size = params["voxel_size"].as<float>();
    float relief_threshold = params["relief_threshold"].as<float>();

    // Load input point cloud using the provided loadPointCloudFromFile function.
    auto loaded_cloud = loadPointCloudFromFile(cloud_file_path);

    // created instance of struct processingResult
    processingResult processResult;
    processResult.inlier_cloud = loaded_cloud;
    processResult.outlier_cloud = std::make_shared<open3d::geometry::PointCloud>();
    PointCloud down_sampled_pointcloud;

    // downsampling if set true in yaml file
    if (voxel_downsample_pointcloud)
    {
        down_sampled_pointcloud = downSamplePointCloud(processResult.inlier_cloud, voxel_size);
        std::visit(
            [](const auto& pointcloud) {
                if constexpr (std::is_same_v<std::decay_t<decltype(pointcloud)>, PointCloudPcl>) {
                    std::cout << "Downsampled cloud has " << pointcloud->points.size() << " points." << std::endl;
                } else {
                    std::cout << "Downsampled cloud has " << pointcloud->points_.size() << " points." << std::endl;
                }
            },
            down_sampled_pointcloud);
        processResult.inlier_cloud = down_sampled_pointcloud;
     
    }
    else
    {
        processResult.inlier_cloud = loaded_cloud;
    }

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
            std::string viz_inlier_or_outlier_or_both = pipeline[i]["parameters"]["viz_inlier_or_outlier_or_both"].as<std::string>();
            auto process_inlier_cloud = std::get<PointCloudOpen3D>(processResult.inlier_cloud);
            auto sor_result = applySorFilterOpen3d(process_inlier_cloud, nb_neighbors, std_ratio);
            auto sor_pclResult_inlier = convertOpen3DToPCL(std::get<PointCloudOpen3D>(sor_result.inlier_cloud));
            auto sor_pclResult_outlier = convertOpen3DToPCL(std::get<PointCloudOpen3D>(sor_result.outlier_cloud));
            sor_result.inlier_cloud = sor_pclResult_inlier;
            sor_result.outlier_cloud = sor_pclResult_outlier;
            processResult.inlier_cloud = sor_result.inlier_cloud;
            processResult.outlier_cloud = sor_result.outlier_cloud;
            if (viz_individual_method)
            {
                visualizePCL(sor_result, viz_inlier_or_outlier_or_both);
            }

        }
        else if (step == "Radial")
        {
            double radius_search = pipeline[i]["parameters"]["radius_search"].as<double>();
            int min_neighbors = pipeline[i]["parameters"]["min_neighbors"].as<int>();
            std::string viz_inlier_or_outlier_or_both = pipeline[i]["parameters"]["viz_inlier_or_outlier_or_both"].as<std::string>();
            auto process_inlier_cloud = convertOpen3DToPCL(std::get<PointCloudOpen3D>(processResult.inlier_cloud));
            auto process_outlier_cloud = convertOpen3DToPCL(std::get<PointCloudOpen3D>(processResult.outlier_cloud));
            auto radial_pclResult = applyRadiusFilterPcl(process_inlier_cloud, radius_search, min_neighbors);
            processResult.inlier_cloud = radial_pclResult.inlier_cloud;
            processResult.outlier_cloud = radial_pclResult.outlier_cloud;
            if (viz_individual_method)
            {
                visualizePCL(radial_pclResult, viz_inlier_or_outlier_or_both);
            }
        }
        else if (step == "SphericalNeighbourhood")
        {
            auto start = std::chrono::high_resolution_clock::now();

            int k = pipeline[i]["parameters"]["k"].as<int>();
            float angleThreshold = pipeline[i]["parameters"]["angleThreshold"].as<float>();
            double radius = pipeline[i]["parameters"]["radius"].as<double>();
            std::string viz_inlier_or_outlier_or_both = pipeline[i]["parameters"]["viz_inlier_or_outlier_or_both"].as<std::string>();
            int maxlandingZones = pipeline[i]["parameters"]["maxlandingZones"].as<int>();
            int maxAttempts = pipeline[i]["parameters"]["maxAttempts"].as<int>();
            float textSize = pipeline[i]["parameters"]["visualization_textSize"].as<float>();
    
            std::vector<LandingZoneCandidatePoint> candidatePoints;
            auto process_inlier_cloud = convertOpen3DToPCL(std::get<PointCloudOpen3D>(processResult.inlier_cloud));
            auto process_outlier_cloud = convertOpen3DToPCL(std::get<PointCloudOpen3D>(processResult.outlier_cloud));
            candidatePoints = kdtreeNeighbourhoodPCAFilterOMP(process_inlier_cloud,
                                            radius, k, angleThreshold, relief_threshold,
                                            maxlandingZones, maxAttempts);
            
            
            auto rankedCandidates = rankCandidatePatches(candidatePoints);
           
             // End the timer
            auto end = std::chrono::high_resolution_clock::now();
            // Calculate the elapsed time in seconds (or choose another unit)
            std::chrono::duration<double> duration = end - start;
            // Print the elapsed time
            std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
            if (viz_individual_method)
            {
                visualizePCL(processResult, viz_inlier_or_outlier_or_both);
                visualizePCLWithRankedCandidates(processResult, rankedCandidates, viz_inlier_or_outlier_or_both, textSize);
                
            }
        }
        else
        {
            std::cerr << "Unknown pipeline step: " << step << std::endl;
            return -1;
        }
    }

    // Final visualization of the chained output.
    if (final_visualize)
    {
        std::cout << "\n--- Final Processed Cloud ---\n";
        visualizePCL(processResult, viz_inlier_or_outlier_or_both_final);
    }

    return 0;
}
