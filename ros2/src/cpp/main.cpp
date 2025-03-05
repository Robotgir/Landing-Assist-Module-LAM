#include <iostream>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"

// Include headers from your libraries.
#include "hazard_metrices.h"
#include "pointcloud_preprocessing.h"

// --- Hazard Metrics Wrapper Functions ---

PCLResult runPCA(const std::string &file_path, float voxel_size, float slope_threshold, int k) {
    std::cout << "[PCA] Running with voxel_size=" << voxel_size
              << ", slope_threshold=" << slope_threshold << ", k=" << k << std::endl;
    return PrincipleComponentAnalysis(file_path, voxel_size, slope_threshold, k);
}

OPEN3DResult runRansac_Open3D(const std::string &file_path, double voxel_size,
                              double distance_threshold, int ransac_n, int num_iterations) {
    std::cout << "[RANSAC_Open3D] Running with voxel_size=" << voxel_size
              << ", distance_threshold=" << distance_threshold << ", ransac_n=" << ransac_n
              << ", num_iterations=" << num_iterations << std::endl;
    return RansacPlaneSegmentation(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);
}

// (Additional wrappers for hazard metrics would follow the same pattern.)

// --- Hazard Metrics Processing Function ---
void processHazardBlock(const std::string &key, const YAML::Node &hazardConfig, const std::string &file_path) {
    YAML::Node block = hazardConfig[key];
    if (!block["enabled"].as<bool>()) {
        std::cout << "Hazard block " << key << " is disabled; skipping.\n";
        return;
    }
    std::string method = block["method"].as<std::string>();
    std::cout << "\nProcessing Hazard Block: " << key << " using method: " << method << std::endl;
    
    if (method == "PCA") {
        float voxel_size = block["parameters"]["voxel_size"].as<float>();
        float slope_threshold = block["parameters"]["slope_threshold"].as<float>();
        int k = block["parameters"]["k"].as<int>();
        PCLResult result = runPCA(file_path, voxel_size, slope_threshold, k);
        visualizePCL(result);
    }
    else if (method == "RANSAC_Open3D") {
        double voxel_size = block["parameters"]["voxel_size"].as<double>();
        double distance_threshold = block["parameters"]["distance_threshold"].as<double>();
        int ransac_n = block["parameters"]["ransac_n"].as<int>();
        int num_iterations = block["parameters"]["num_iterations"].as<int>();
        OPEN3DResult result = runRansac_Open3D(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);
        VisualizeOPEN3D(result);
    }
    // Implement additional else-if branches for:
    // "RANSAC_PCL", "PROSAC", "LeastSquares_Open3D", "LMEDS", "Average3DGradient",
    // "RegionGrowing", "CalculateRoughness_PCL", "CalculateRoughness_Open3D",
    // "CalculateRelief_PCL", "CalculateRelief_Open3D", "CalculateDataConfidence_PCL", "CalculateDataConfidence_Open3D".
}

// --- Preprocessing Data Structuring Processing ---
void processDataStructuringMethods(const YAML::Node &dsMethods, const std::string &file_path) {
    // Iterate over each data structuring method.
    for (YAML::const_iterator it = dsMethods.begin(); it != dsMethods.end(); ++it) {
        std::string methodName = it->first.as<std::string>();
        YAML::Node methodNode = it->second;
        if (!methodNode["enabled"].as<bool>()) {
            std::cout << "[DataStructuring] Method " << methodName << " is disabled; skipping.\n";
            continue;
        }
        std::cout << "\n[DataStructuring] Running method: " << methodName << std::endl;
        if (methodName == "GridMap") {
            float gridmap_resolution = methodNode["parameters"]["gridmap_resolution"].as<float>();
            pcl::PointCloud<pcl::PointXYZ>::Ptr grid_map = create2DGridMap(file_path, gridmap_resolution);
            if (grid_map)
                visualize2DGridMap(grid_map);
        }
        else if (methodName == "3DGridMap") {
            double voxel_size = methodNode["parameters"]["voxel_size"].as<double>();
            VoxelGridResult result = create_3d_grid(file_path, voxel_size);
            if (!result.voxel_grid_ptr || !result.cloud_ptr) {
                std::cerr << "Failed to create 3D Grid Map." << std::endl;
                continue;
            }
            Visualize3dGridMap(result.voxel_grid_ptr);
        }
        else if (methodName == "Octree") {
            int max_depth = methodNode["parameters"]["max_depth"].as<int>();
            OctreeResult result = create_octree(file_path, max_depth);
            if (!result.octree || !result.cloud_ptr) {
                std::cerr << "Failed to create Octree." << std::endl;
                continue;
            }
            std::cout << "Octree created successfully." << std::endl;
        }
        else if (methodName == "KDTree") {
            float K = methodNode["parameters"]["K"].as<float>();
            KDTreeResult result = create_kdtree(file_path, K);
            if (!result.kdtree || !result.cloud_ptr) {
                std::cerr << "Failed to create KDTree." << std::endl;
                continue;
            }
            std::cout << "KDTree created successfully." << std::endl;
        }
        else if (methodName == "Octomap") {
            std::string octomap_filename = methodNode["parameters"]["octomap_filename"].as<std::string>();
            double octomap_resolution = methodNode["parameters"]["octomap_resolution"].as<double>();
            convertPointCloudToOctomap(file_path, octomap_filename, octomap_resolution);
            std::cout << "Octomap saved to " << octomap_filename << std::endl;
        }
        else {
            std::cerr << "Unknown data structuring method: " << methodName << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    // Use fixed file paths for demonstration.
    std::string pcd_file = "/home/airsim_user/Landing-Assist-Module-LAM/lib/hazard_metrices/test.pcd";
    std::string config_file = "/home/airsim_user/Landing-Assist-Module-LAM/config/config.yaml";

    // Load YAML configuration.
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node params = config["ros__parameters"];

    // -------------------------------
    // Preprocessing Section
    // -------------------------------
    YAML::Node preprocConfig = params["preprocessing"];
    std::cout << "\n--- Running Preprocessing Blocks ---" << std::endl;

    // Data Structuring: Process all enabled methods.
    if (preprocConfig["data_structuring"]["enabled"].as<bool>()) {
        YAML::Node dsMethods = preprocConfig["data_structuring"]["methods"];
        processDataStructuringMethods(dsMethods, pcd_file);
    }

    // Filtering Section.
    if (preprocConfig["filtering"]["enabled"].as<bool>()) {
        YAML::Node filterMethods = preprocConfig["filtering"]["methods"];
        
        // VoxelGrid Filter.
        if (filterMethods["voxel_grid_filter"]["enabled"].as<bool>()) {
            float voxel_downsample_size = filterMethods["voxel_grid_filter"]["parameters"]["voxel_downsample_size"].as<float>();
            auto downsampled_cloud = apply_voxel_grid_filter(pcd_file, voxel_downsample_size);
            if (downsampled_cloud)
                VisualizeGeometry(downsampled_cloud);
        }
        // SOR Filter.
        if (filterMethods["sor"]["enabled"].as<bool>()) {
            int nb_neighbors = filterMethods["sor"]["parameters"]["nb_neighbors"].as<int>();
            double std_ratio = filterMethods["sor"]["parameters"]["std_ratio"].as<double>();
            SORFilterResult sor_result = apply_sor_filter(pcd_file, nb_neighbors, std_ratio);
            if (sor_result.filtered_cloud)
                visualize_sor_filtered_point_cloud(sor_result.original_cloud, sor_result.filtered_cloud);
        }
        // Radius Outlier Removal Filter.
        if (filterMethods["radius_outlier_removal"]["enabled"].as<bool>()) {
            double radius_search = filterMethods["radius_outlier_removal"]["parameters"]["radius_search"].as<double>();
            int min_neighbors = filterMethods["radius_outlier_removal"]["parameters"]["min_neighbors"].as<int>();
            float translation_offset = filterMethods["radius_outlier_removal"]["parameters"]["translation_offset"].as<float>();
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud_original) != -1) {
                auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, 0.1f);
                auto cloud_radius_filtered = applyRadiusFilter<pcl::PointXYZI>(cloud_downsampled, radius_search, min_neighbors);
                if (!cloud_radius_filtered->empty())
                    visualizeClouds<pcl::PointXYZI>(cloud_downsampled, cloud_radius_filtered,
                                                    "Radius Outlier Removal", "original cloud",
                                                    "radius_filtered cloud", 2, translation_offset);
            }
        }
        // Bilateral Filter.
        if (filterMethods["bilateral_filter"]["enabled"].as<bool>()) {
            double sigma_s = filterMethods["bilateral_filter"]["parameters"]["sigma_s"].as<double>();
            double sigma_r = filterMethods["bilateral_filter"]["parameters"]["sigma_r"].as<double>();
            float translation_offset = filterMethods["bilateral_filter"]["parameters"]["translation_offset"].as<float>();
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud_original) != -1) {
                auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, 0.1f);
                auto cloud_bilateral_filtered = applyBilateralFilter<pcl::PointXYZI>(cloud_downsampled, sigma_s, sigma_r);
                if (!cloud_bilateral_filtered->empty())
                    visualizeClouds<pcl::PointXYZI>(cloud_downsampled, cloud_bilateral_filtered,
                                                    "Bilateral Filter", "original cloud",
                                                    "bilateral_filtered cloud", 2, translation_offset);
            }
        }
    }

    // -------------------------------
    // Hazard Metrices Section
    // -------------------------------
    YAML::Node hazardConfig = params["hazard_metrices"];
    std::cout << "\n--- Running Hazard Metrics Blocks ---" << std::endl;

    // Read the explicit processing order for hazard metrics from YAML.
    if (params["processing_order"]["hazard"]) {
        std::vector<std::string> order;
        for (std::size_t i = 0; i < params["processing_order"]["hazard"].size(); ++i) {
            order.push_back(params["processing_order"]["hazard"][i].as<std::string>());
        }
        // Process each hazard metric block in the specified order.
        for (const auto &key : order) {
            if (hazardConfig[key]) {
                processHazardBlock(key, hazardConfig, pcd_file);
            } else {
                std::cerr << "Warning: Key " << key << " not found in hazard_metrices." << std::endl;
            }
        }
    } else {
        std::cerr << "No processing_order defined in YAML for hazard metrics. Exiting." << std::endl;
        return -1;
    }

    return 0;
}
