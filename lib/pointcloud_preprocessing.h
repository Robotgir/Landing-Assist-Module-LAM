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

using PointT = pcl::PointXYZI;

template <typename PointT>
using CloudInput = std::variant<std::string, typename pcl::PointCloud<PointT>::Ptr>;

//======================================== DATA STRUCTURING GRID BASED ==================================================

// Struct to hold both PointCloud and VoxelGrid
struct VoxelGridResult {
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr;
    std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_ptr;
};

// // Function to load point cloud and create voxel grid
inline VoxelGridResult create_3d_grid(const std::string& filePath, double voxel_size) {
    VoxelGridResult result;

    // 1. Load the Point Cloud
    
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filePath, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filePath << std::endl;
        return result; // cloud_ptr and voxel_grid_ptr remain nullptr
    }
    std::cout << "Loaded point cloud with " << result.cloud_ptr->points_.size() << " points." << std::endl;

    //  2. Create the Voxel Grid
    // Open3D provides the VoxelDownSample function to create a voxel grid
    result.voxel_grid_ptr = open3d::geometry::VoxelGrid::CreateFromPointCloud(*result.cloud_ptr, voxel_size);
    if (result.voxel_grid_ptr->voxels_.empty()) {
        std::cerr << "3d Voxel grid creation failed. Possibly due to an inappropriate voxel size." << std::endl;
        result.voxel_grid_ptr = nullptr;
        return result;
    }
    std::cout << "3d Voxel grid created with voxel size " << voxel_size << " and " 
              << result.voxel_grid_ptr->voxels_.size() << " voxels." << std::endl;

    return result;
}
// Reference link to 3d grid map : https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_voxel_grid.html#ae9239348e9de069abaf5912b92dd2b84

// Function to visualize a single Open3D geometry
inline void Visualize3dGridMap(const std::shared_ptr<open3d::geometry::Geometry>& geometry,
                               const std::string& window_title = "3D Grid Map",
                               int width = 1600,
                               int height = 900) 
{
    if (!geometry) {
    std::cerr << "Invalid input geometry. Cannot visualize." << std::endl;
    return;
    }

    // If the geometry is a point cloud, set the color to green
    if (auto pointcloud = std::dynamic_pointer_cast<open3d::geometry::PointCloud>(geometry)) {
    pointcloud->colors_.resize(pointcloud->points_.size(), Eigen::Vector3d(0.0, 1.0, 0.0)); // Green color (R=0, G=1, B=0)
    }

    // Create a vector to hold the geometry pointers
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    geometries.push_back(geometry);

    // Call the Open3D visualization function
    open3d::visualization::DrawGeometries(geometries, window_title, width, height);
}


// Function to create a 2D grid map using the GridMinimum filter.
inline pcl::PointCloud<PointT>::Ptr create2DGridMap(const std::string &filePath, float resolution) {
    // Load input point cloud
    pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(filePath, *input_cloud) == -1) {
        std::cerr << "ERROR: Could not read file " << filePath << std::endl;
        return nullptr;
    }
    std::cout << "Loaded " << input_cloud->size() << " points from " << filePath << std::endl;

    // Create the GridMinimum filter object using the provided resolution.
    // The filter will downsample the point cloud by selecting the minimum z value in each grid cell.
    pcl::GridMinimum<PointT> grid_min_filter(resolution);
    grid_min_filter.setInputCloud(input_cloud);

    // Apply the filter
    pcl::PointCloud<PointT>::Ptr grid_cloud(new pcl::PointCloud<PointT>());
    grid_min_filter.filter(*grid_cloud);

    std::cout << "Created 2D grid map with " << grid_cloud->size() << " points (grid cells)." << std::endl;
    return grid_cloud;
}
// Reference Link to 2d gridmap : http://pointclouds.org/documentation/classpcl_1_1_grid_minimum.html#details
// https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/grid_minimum.h


// Function to visualize a 2D grid map (represented as a point cloud)
// using the PCLVisualizer.
inline void visualize2DGridMap(pcl::PointCloud<PointT>::Ptr cloud) {
    if (!cloud || cloud->empty()) {
        std::cerr << "ERROR: Cannot visualize an empty grid map." << std::endl;
        return;
    }

    // Create a PCL Visualizer object.
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("2D Grid Map Visualization"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    // Assign a color (e.g., green) for the grid map points.
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler(cloud, 0, 255, 0);
    viewer->addPointCloud<PointT>(cloud, color_handler, "grid_map");

    // Set rendering properties (e.g., point size)
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "grid_map");
    viewer->addCoordinateSystem(1.0);

    // Main visualization loop.
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


//======================================= DATA STRUCTURING TREE STRUCTURE BASED ===============================================================


// Function to convert point cloud to octomap
inline void convertPointCloudToOctomap(const std::string& pcd_filePath, const std::string& octomap_filePath, double resolution = 0.05)
{
    // Load the point cloud
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(pcd_filePath, *cloud) == -1)
    {
        std::cerr << "[ERROR] Could not read PCD file: " << pcd_filePath << std::endl;
        return;
    }
    std::cout << "[INFO] Loaded " << cloud->size() << " points from " << pcd_filePath << std::endl;

    // Create OctoMap

    octomap::OcTree tree(resolution);

    // Insert points into OctoMap
    for (const auto& point : cloud->points)
    {
        tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
    }

    // Update inner nodes to ensure consistency
    tree.updateInnerOccupancy();

    // Save OctoMap as a binary file
    tree.writeBinary(octomap_filePath);
    std::cout << "[INFO] OctoMap saved as " << octomap_filePath << std::endl;
}

//Struct to hold kdtree

struct KDTreeResult {
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr;
    std::shared_ptr<open3d::geometry::KDTreeFlann> kdtree;
    std::vector<int> neighbor_indices;
    std::vector<double> neighbor_distances;
};

// Function to load point cloud, build KDTree

inline KDTreeResult create_kdtree(const std::string& filePath, float K) {
    KDTreeResult result;

    // 1. Load the Point Cloud
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filePath, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filePath << std::endl;
        return result; // cloud_ptr and kdtree remain nullptr
    }
    std::cout << "Loaded point cloud with " << result.cloud_ptr->points_.size() << " points." << std::endl;

    // 2. Build the KDTreeFlann
    result.kdtree = std::make_shared<open3d::geometry::KDTreeFlann>();
    result.kdtree->SetGeometry(*result.cloud_ptr);
    std::cout << "KDTree built successfully." << std::endl;

    return result;
}
//Reference Link :https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_k_d_tree_flann.html#ae9239348e9de069abaf5912b92dd2b84

// Struct to hold octree
struct OctreeResult {
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr;
    std::shared_ptr<open3d::geometry::Octree> octree;
    std::shared_ptr<open3d::geometry::OctreeNode> found_node;
    std::shared_ptr<open3d::geometry::OctreeNodeInfo> node_info;
};

// Function to load point cloud, build KDTree
inline OctreeResult create_octree(const std::string& filePath, int max_depth) {
    OctreeResult result;

    // 1. Load the Point Cloud
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filePath, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filePath << std::endl;
        return result; // cloud_ptr and octree remain nullptr
    }
    std::cout << "Loaded point cloud with " << result.cloud_ptr->points_.size() << " points." << std::endl;

    // 2. Build the Octree
    result.octree = std::make_shared<open3d::geometry::Octree>(max_depth);
    result.octree->ConvertFromPointCloud(*result.cloud_ptr);
    std::cout << "Octree built successfully with max depth " << max_depth << "." << std::endl;

    
    return result;
}
//Reference Link : https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_octree.html
 
//=========================================== FILTERING DOWN SAMPLING ======================================================


inline std::shared_ptr<open3d::geometry::PointCloud> apply_voxel_grid_filter(
    const std::string& filePath,double voxel_size) 
{

    // 1. Load the Point Cloud
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filePath, *cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filePath << std::endl;
        return cloud_ptr; // cloud_ptr and octree remain nullptr
    }
    std::cout << "Loaded point cloud with " << cloud_ptr->points_.size() << " points." << std::endl;
    auto downsampled = cloud_ptr->VoxelDownSample(voxel_size);
    if (downsampled->points_.empty()) {
        std::cerr << "Warning: Downsampled cloud is empty. Try a smaller voxel size." << std::endl;
        return nullptr;
    }

    std::cout << "Downsampled point cloud has " << downsampled->points_.size() << " points." << std::endl;
    return downsampled;
}
// Reference Link: https://www.open3d.org/docs/0.11.0/cpp_api/classopen3d_1_1geometry_1_1_point_cloud.html#a50efddf2d460dccf3de46cd0d38071af

// Function to visualize a single Open3D geometry
inline void VisualizeGeometry(const std::shared_ptr<const open3d::geometry::Geometry>& geometry,
                       const std::string& window_title = "Downsampled Voxel Grid (Voxel grid filter )",
                       int width = 1600,
                       int height = 900) {
    // Create a vector to hold the geometry pointers
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    geometries.push_back(geometry);
    
    // Call the Open3D visualization function
    open3d::visualization::DrawGeometries(geometries, window_title, width, height);
}
//=============================== FILTERING OUTLIER REMOVAL PCL ===============================================
PCLResult applySORFilterPCL(const std::string &pcd_file, int meanK = 50, double stddevMulThresh = 1.0)
{
    PCLResult result;
    result.downsampled_cloud.reset(new PointCloudT);
    result.inlier_cloud.reset(new PointCloudT);
    result.outlier_cloud.reset(new PointCloudT);
    result.plane_coefficients.reset(new pcl::ModelCoefficients);
    result.pcl_method = "SOR";

    // Load the input cloud from file.
    pcl::PCDReader reader;
    PointCloudT::Ptr cloud(new PointCloudT);
    if (reader.read<PointT>(pcd_file, *cloud) == -1)
    {
        std::cerr << "Failed to load " << pcd_file << std::endl;
        return result;
    }

    std::cerr << "Cloud before filtering:" << std::endl;
    std::cerr << *cloud << std::endl;

    // Save the original cloud for visualization.
    result.downsampled_cloud = cloud;

    // Create and configure the SOR filter.
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddevMulThresh);

    // Filter to extract inliers (points kept by the filter).
    PointCloudT::Ptr cloud_inliers(new PointCloudT);
    sor.filter(*cloud_inliers);
    result.inlier_cloud = cloud_inliers;

    std::cerr << "Cloud after filtering (inliers):" << std::endl;
    std::cerr << *cloud_inliers << std::endl;

    // Now set the filter to negative mode to extract outliers.
    sor.setNegative(true);
    PointCloudT::Ptr cloud_outliers(new PointCloudT);
    sor.filter(*cloud_outliers);
    result.outlier_cloud = cloud_outliers;

    std::cerr << "Cloud outliers:" << std::endl;
    std::cerr << *cloud_outliers << std::endl;

    return result;
}
//=============================== FILTERING OUTLIER REMOVAL ===============================================

inline OPEN3DResult apply_sor_filter(
    const std::string& filePath,
    int nb_neighbors,
    double std_ratio)
{
    OPEN3DResult result;
    result.open3d_method = "StatisticalOutlierRemoval";

    // Load the original point cloud
    auto original_cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filePath, *original_cloud)) {
        std::cerr << "Failed to read point cloud: " << filePath << std::endl;
        return result;
    }
    std::cout << "Loaded point cloud with " << original_cloud->points_.size() << " points." << std::endl;

    // Apply SOR filter: RemoveStatisticalOutliers returns a pair: (filtered_cloud, inlier_indices)
    // Here, filtered_cloud contains the inliers (i.e. noise removed) and inlier_indices are their indices.
    auto [filtered_cloud, inlier_indices] = original_cloud->RemoveStatisticalOutliers(nb_neighbors, std_ratio);

    // Instead of manually computing the complement, we can use the invert flag in SelectByIndex.
    // In this case, the noise (outliers) are the complement of the inlier indices.
    auto noise_cloud = original_cloud->SelectByIndex(inlier_indices, true);

    // Swap the assignment so that:
    //   - result.inlier_cloud holds the noise (points removed by filtering)
    //   - result.outlier_cloud holds the filtered inlier points.
    result.inlier_cloud = filtered_cloud;
    result.outlier_cloud = noise_cloud;

    // For SOR filter, downsampled_cloud and plane_model are not applicable.
    result.downsampled_cloud = nullptr;
    result.plane_model = Eigen::Vector4d(0, 0, 0, 0);

    return result;
}

// Reference Link: https://www.open3d.org/docs/0.6.0/cpp_api/namespaceopen3d_1_1geometry.html#add56e2ec673de3b9289a25095763af6d
// https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.cpp#L602

inline PCLResult applyRadiusFilter(
    const std::string& file_path,
    double voxel_size=0.45,
    double radius_search=0.9,
    int min_neighbors=40)
{
    PCLResult result;

    result.pcl_method = "Radius Outlier Removal";
    result.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.outlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();

    // Load the original point cloud from file.
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1) {
        std::cerr << "[ERROR] Failed to load original PCD file: " << file_path << std::endl;
        return result;  // Return with empty clouds on error.
    }
    std::cout << "[INFO] Loaded " << cloud->size() 
              << " points from " << file_path << std::endl;

    // Downsample the cloud using a voxel grid filter.
    // Assume that downsamplePointCloudPCL is a helper function defined elsewhere.
    // typename pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>());
    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxel_size);
    

    // Set up the Radius Outlier Removal filter.
    pcl::RadiusOutlierRemoval<PointT> ror;
    ror.setInputCloud(result.downsampled_cloud);
    ror.setRadiusSearch(radius_search);
    ror.setMinNeighborsInRadius(min_neighbors);

    // First pass: get inliers (points that meet the criteria).
    ror.setNegative(false);
    ror.filter(*result.inlier_cloud);
    // result.inlier_cloud = result.inlier_cloud;
    std::cout << "[INFO] Applied Radius Outlier Removal for inliers. Cloud size: " 
              << result.inlier_cloud->size() << std::endl;

    // Second pass: get outliers (points that do not meet the criteria).
    ror.setNegative(true);
    ror.filter(*result.outlier_cloud);
    // result.outlier_cloud = cloud_outliers;
    std::cout << "[INFO] Applied Radius Outlier Removal for outliers. Cloud size: " 
              << result.outlier_cloud->size() << std::endl;

    return result;
}

//// Referencle Link : http://pointclouds.org/documentation/classpcl_1_1_radius_outlier_removal.html
////                   https://github.com/PointCloudLibrary/pcl/blob/master/filters/src/radius_outlier_removal.cpp#L47




//=============================== FILTERING SMOOTHING ======================================================

// Function to apply Bilateral Filter

inline PCLResult applyBilateralFilter(
    const std::string& file_path,
    double voxel_size,
    double sigma_s,
    double sigma_r)
{
    PCLResult result;
    result.pcl_method = "BilateralFilter";
    result.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.outlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    
    // Load the original point cloud from file.
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1) {
        std::cerr << "[ERROR] Failed to load original PCD file: " << file_path << std::endl;
        return result;  // Return with empty clouds on error.
    }
    std::cout << "[INFO] Loaded " << cloud->size() 
              << " points from " << file_path << std::endl;

    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxel_size);

    // Initialize the bilateral filter
    pcl::BilateralFilter<PointT> bilateral_filter;

    
    
    // Set the input cloud
    bilateral_filter.setInputCloud(cloud);
    
    // Set filter parameters
    bilateral_filter.setHalfSize(sigma_s);
    bilateral_filter.setStdDev(sigma_r);
    
    // Use KdTree as the search method
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    bilateral_filter.setSearchMethod(tree);

    // result.inlier_cloud = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    // Apply the filter
    bilateral_filter.filter(*result.inlier_cloud);

    std::cout << "[INFO] Applied Bilateral Filter. Filtered cloud size: " 
              << result.inlier_cloud->size() << std::endl;

    // Store the filtered cloud
  

    // No inlier/outlier separation or plane fitting in bilateral filtering
    result.outlier_cloud = nullptr;
    result.plane_coefficients = nullptr;

    return result;
}

//   Reference Link: https://pointclouds.org/documentation/classpcl_1_1_bilateral_filter.html
//                   https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/bilateral.h#L56

#endif // POINTCLOUD_PREPROCESSING_H
