#ifndef POINTCLOUD_PREPROCESSING_H
#define POINTCLOUD_PREPROCESSING_H

#include <common.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <thread>
#include <chrono>

#include <cmath>
// Open3D headers
#include <open3d/Open3D.h>
// Ocotmap headers
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
//PCL headers
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/bilateral.h>        // Custom Bilateral Filter
#include <pcl/filters/impl/bilateral.hpp> // Implementation
#include <pcl/search/kdtree.h>             // Include for KdTree
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/grid_minimum.h>

#include <unordered_set>

#include <pcl/filters/statistical_outlier_removal.h>


//=============================== FILTERING OUTLIER REMOVAL ===============================================
inline processingResult applySorFilterOpen3d(
    const PointCloudOpen3D& input_cloud,
    int nb_neighbors,
    double std_ratio)
{
    processingResult result;
    result.processing_method = "StatisticalOutlierRemoval_open3d";

    // Apply SOR filter: RemoveStatisticalOutliers returns a pair: (filtered_cloud, inlier_indices)
    auto [filtered_cloud, inlier_indices] = input_cloud->RemoveStatisticalOutliers(nb_neighbors, std_ratio);

    // Get outliers by selecting the complement of inlier indices
    auto noise_cloud = input_cloud->SelectByIndex(inlier_indices, true);

    // Store the filtered cloud and noise cloud in the result structure
    result.inlier_cloud = filtered_cloud;  // Assign PointCloudOpen3D to PointCloud variant
    result.outlier_cloud = noise_cloud;    // Assign PointCloudOpen3D to PointCloud variant

    // Print sizes
    std::cout << "[INFO] Applied Statistical Outlier Removal. Filtered cloud size: "
              << filtered_cloud->points_.size() << std::endl;
    std::cout << "[INFO] Noise cloud size: " << noise_cloud->points_.size() << std::endl;

    return result;
}
// Reference Link: https://www.open3d.org/docs/0.6.0/cpp_api/namespaceopen3d_1_1geometry.html#add56e2ec673de3b9289a25095763af6d
// https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.cpp#L602

inline processingResult applyRadiusFilterPcl(
    const PointCloudPcl& input_cloud,
    float radius_search,
    int min_neighbors)
{
    processingResult result;
    auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
    auto result_outlier_cloud = std::get<PointCloudPcl>(result.outlier_cloud);
    result.processing_method = "Radius Outlier Removal PCL";
               
    // Set up the Radius Outlier Removal filter.
    pcl::RadiusOutlierRemoval<PointPcl> ror;
    ror.setInputCloud(input_cloud);
    ror.setRadiusSearch(radius_search);
    ror.setMinNeighborsInRadius(min_neighbors);

    // First pass: get inliers (points that meet the criteria).
    ror.setNegative(false);
    ror.filter(*result_inlier_cloud);
    std::cout << "[INFO] Applied Radius Outlier Removal PCL with resulting inliers. Cloud size: " 
              << result_inlier_cloud->size() << std::endl;

    // Second pass: get outliers (points that do not meet the criteria).
    ror.setNegative(true);
    ror.filter(*result_outlier_cloud);
    // result_outlier_cloud = cloud_outliers;
    std::cout << "[INFO] Applied Radius Outlier Removal with resulting outliers. Cloud size: " 
              << result_outlier_cloud->size() << std::endl;

    return result;
}

//// Referencle Link : http://pointclouds.org/documentation/classpcl_1_1_radius_outlier_removal.html
////                   https://github.com/PointCloudLibrary/pcl/blob/master/filters/src/radius_outlier_removal.cpp#L47



//   Reference Link: https://pointclouds.org/documentation/classpcl_1_1_bilateral_filter.html
//                   https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/bilateral.h#L56

#endif // POINTCLOUD_PREPROCESSING_H
