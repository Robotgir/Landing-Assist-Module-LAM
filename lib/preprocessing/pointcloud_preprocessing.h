#ifndef POINTCLOUD_PREPROCESSING_H
#define POINTCLOUD_PREPROCESSING_H

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// DATA STRUCTURING GRID BASED //////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Struct to hold both PointCloud and VoxelGrid
struct VoxelGridResult {
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr;
    std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_ptr;
};

// // Function to load point cloud and create voxel grid
inline VoxelGridResult create_3d_grid(const std::string& filename, double voxel_size) {
    VoxelGridResult result;

    // 1. Load the Point Cloud
    
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filename << std::endl;
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
inline void Visualize3dGridMap(const std::shared_ptr<const open3d::geometry::Geometry>& geometry,
                       const std::string& window_title = "3d Grid Map",
                       int width = 1600,
                       int height = 900) {
    // Create a vector to hold the geometry pointers
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    geometries.push_back(geometry);
    
    // Call the Open3D visualization function
    open3d::visualization::DrawGeometries(geometries, window_title, width, height);
}

// Function to create a 2D grid map using the GridMinimum filter.
inline pcl::PointCloud<pcl::PointXYZ>::Ptr create2DGridMap(const std::string &filename, float resolution) {
    // Load input point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *input_cloud) == -1) {
        std::cerr << "ERROR: Could not read file " << filename << std::endl;
        return nullptr;
    }
    std::cout << "Loaded " << input_cloud->size() << " points from " << filename << std::endl;

    // Create the GridMinimum filter object using the provided resolution.
    // The filter will downsample the point cloud by selecting the minimum z value in each grid cell.
    pcl::GridMinimum<pcl::PointXYZ> grid_min_filter(resolution);
    grid_min_filter.setInputCloud(input_cloud);

    // Apply the filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr grid_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    grid_min_filter.filter(*grid_cloud);

    std::cout << "Created 2D grid map with " << grid_cloud->size() << " points (grid cells)." << std::endl;
    return grid_cloud;
}
// Reference Link to 2d gridmap : http://pointclouds.org/documentation/classpcl_1_1_grid_minimum.html#details
// https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/grid_minimum.h


// Function to visualize a 2D grid map (represented as a point cloud)
// using the PCLVisualizer.
inline void visualize2DGridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    if (!cloud || cloud->empty()) {
        std::cerr << "ERROR: Cannot visualize an empty grid map." << std::endl;
        return;
    }

    // Create a PCL Visualizer object.
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("2D Grid Map Visualization"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    // Assign a color (e.g., green) for the grid map points.
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, color_handler, "grid_map");

    // Set rendering properties (e.g., point size)
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "grid_map");
    viewer->addCoordinateSystem(1.0);

    // Main visualization loop.
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// DATA STRUCTURING TREE STRUCTURE BASED ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to convert point cloud to octomap
inline void convertPointCloudToOctomap(const std::string& pcd_filename, const std::string& octomap_filename, double resolution = 0.05)
{
    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filename, *cloud) == -1)
    {
        std::cerr << "[ERROR] Could not read PCD file: " << pcd_filename << std::endl;
        return;
    }
    std::cout << "[INFO] Loaded " << cloud->size() << " points from " << pcd_filename << std::endl;

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
    tree.writeBinary(octomap_filename);
    std::cout << "[INFO] OctoMap saved as " << octomap_filename << std::endl;
}

//Struct to hold kdtree

struct KDTreeResult {
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr;
    std::shared_ptr<open3d::geometry::KDTreeFlann> kdtree;
    std::vector<int> neighbor_indices;
    std::vector<double> neighbor_distances;
};

// Function to load point cloud, build KDTree

inline KDTreeResult create_kdtree(const std::string& filename, float K) {
    KDTreeResult result;

    // 1. Load the Point Cloud
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filename << std::endl;
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
inline OctreeResult create_octree(const std::string& filename, int max_depth) {
    OctreeResult result;

    // 1. Load the Point Cloud
    result.cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *result.cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filename << std::endl;
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
 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// FILTERING DOWN SAMPLING //////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<open3d::geometry::PointCloud> apply_voxel_grid_filter(
    const std::string& filename,double voxel_size) 
{

    // 1. Load the Point Cloud
    std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *cloud_ptr)) {
        std::cerr << "Failed to read point cloud: " << filename << std::endl;
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// FILTERING OUTLIER REMOVAL //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Struct to hold original and filtered point clouds
struct SORFilterResult {
    std::shared_ptr<open3d::geometry::PointCloud> original_cloud;
    std::shared_ptr<open3d::geometry::PointCloud> filtered_cloud;
};

// Function to apply Statistical Outlier Removal (SOR) filter
inline SORFilterResult apply_sor_filter(
    const std::string& filename,
    int nb_neighbors,
    double std_ratio)
{
    SORFilterResult result;
    // 1. Load the Point Cloud
    result.original_cloud = std::make_shared<open3d::geometry::PointCloud>();
    
    if (!open3d::io::ReadPointCloud(filename, *result.original_cloud)) {
        std::cerr << "Failed to read point cloud: " << filename << std::endl;
        return result; // result.original_cloud and octree remain nullptr
    }
    std::cout << "Loaded point cloud with " << result.original_cloud->points_.size() << " points." << std::endl;


    // Apply SOR filter
    std::vector<size_t> inlier_indices;
    auto [sor_filtered_cloud, indices] = result.original_cloud->RemoveStatisticalOutliers(nb_neighbors, std_ratio);
    result.filtered_cloud = sor_filtered_cloud; // Assign filtered cloud to SORFilterResult member

    if (!result.filtered_cloud) {
        std::cerr << "Warning: SOR filtering resulted in an empty cloud." << std::endl;
        return result;  // Return early with error message printed
    }

  
    //std::cout << "SOR filter applied. Filtered point cloud has " << sor_filtered_cloud->points_.size() << " points." << std::endl;
    return result;
}
// Reference Link: https://www.open3d.org/docs/0.6.0/cpp_api/namespaceopen3d_1_1geometry.html#add56e2ec673de3b9289a25095763af6d
// https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.cpp#L602

// Function to visualize original and filtered point clouds
inline void visualize_sor_filtered_point_cloud(
    const std::shared_ptr<open3d::geometry::PointCloud>& original,
    const std::shared_ptr<open3d::geometry::PointCloud>& filtered)
{
    if (!original || !filtered) {
        std::cerr << "Error: One or both point clouds to visualize are null." << std::endl;
        return;
    }

    // Assign colors for differentiation
    auto original_colored = std::make_shared<open3d::geometry::PointCloud>(*original);
    original_colored->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0)); // Red for original

    auto filtered_colored = std::make_shared<open3d::geometry::PointCloud>(*filtered);
    filtered_colored->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0)); // Green for filtered

    // Prepare geometries for visualization
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    geometries.push_back(original_colored);
    geometries.push_back(filtered_colored);

    // Visualize
    open3d::visualization::DrawGeometries(geometries, "SOR Filtere Original (Red) and Filtered (Green) Point Clouds", 1600, 900);
}

// Function to downsample a point cloud using VoxelGrid filter
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr downsamplePointCloud(
    const typename pcl::PointCloud<PointT>::Ptr &cloud, 
    float leaf_size)
{
    typename pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*cloud_downsampled);
    std::cout << "[INFO] Downsampled cloud size: " << cloud_downsampled->size() << std::endl;
    return cloud_downsampled;
}


// Function to apply Radius Outlier Removal filter
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr applyRadiusFilter(
    const typename pcl::PointCloud<PointT>::Ptr &cloud,
    double radius_search,
    int min_neighbors)
{
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());

    pcl::RadiusOutlierRemoval<PointT> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(radius_search);
    ror.setMinNeighborsInRadius(min_neighbors);
    ror.filter(*cloud_filtered);

    std::cout << "[INFO] Applied Radius Outlier Removal. Filtered cloud size: " 
              << cloud_filtered->size() << std::endl;

    return cloud_filtered;
}
//// Referencle Link : http://pointclouds.org/documentation/classpcl_1_1_radius_outlier_removal.html
////                       https://github.com/PointCloudLibrary/pcl/blob/master/filters/src/radius_outlier_removal.cpp#L47



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// FILTERING SMOOTHING /////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to apply Bilateral Filter
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr applyBilateralFilter(
    const typename pcl::PointCloud<PointT>::Ptr &cloud,
    double sigma_s,
    double sigma_r) // Changed sigma_r to double
{
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());

    // Initialize the bilateral filter
    pcl::BilateralFilter<PointT> bilateral_filter;
    
    // Set the input cloud
    bilateral_filter.setInputCloud(cloud);
    
    // Set filter parameters
    bilateral_filter.setHalfSize(sigma_s);  // Spatial standard deviation
    bilateral_filter.setStdDev(sigma_r);     // Range standard deviation
    
    // Create and set the search method
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    bilateral_filter.setSearchMethod(tree);
    
    // Apply the filter
    bilateral_filter.filter(*cloud_filtered); // Use filter() as per implementation
    
    std::cout << "[INFO] Applied Bilateral Filter. Filtered cloud size: " 
              << cloud_filtered->size() << std::endl;

    return cloud_filtered;
}
//   Reference Link: https://pointclouds.org/documentation/classpcl_1_1_bilateral_filter.html
//                   https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/bilateral.h#L56

// Function to visualize two point clouds (as defined above)
template<typename PointT>
void visualizeClouds(
    const typename pcl::PointCloud<PointT>::Ptr &cloud_downsampled,
    const typename pcl::PointCloud<PointT>::Ptr &filtered_cloud,
    const std::string &window_title,
    const std::string &original_cloud_name,
    const std::string &filtered_cloud_name,
    int point_size,
    float translation_offset)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(1.0, 1.0, 1.0); // White background

    // Translate the filtered cloud to add a visible space between the two clouds
    typename pcl::PointCloud<PointT>::Ptr translated_filtered_cloud(new pcl::PointCloud<PointT>(*filtered_cloud));

    for (auto &point : translated_filtered_cloud->points)
    {
        point.x += translation_offset; // Translate the filtered cloud along the x-axis
        //point.y += translation_offset; // Translate the filtered cloud along the y-axis
        //point.z += translation_offset; // Translate the filtered cloud along the z-axis

    }

    // Original cloud in green
    pcl::visualization::PointCloudColorHandlerCustom<PointT> orig_color(cloud_downsampled,0 ,255 , 0); // green
        // Filtered cloud in red
    pcl::visualization::PointCloudColorHandlerCustom<PointT> filt_color(translated_filtered_cloud,255, 0, 0); // red

        // Add the original point cloud to the viewer
    viewer->addPointCloud<PointT>(cloud_downsampled, orig_color, original_cloud_name);
        // Set the point size for the original cloud
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                               point_size,
                                               original_cloud_name);

        // Add the filtered point cloud to the viewer
    viewer->addPointCloud<PointT>(translated_filtered_cloud, filt_color, filtered_cloud_name);
        // Set the point size for the filtered cloud
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  point_size,
                                                 filtered_cloud_name);
    //}

    // Add a coordinate system (scale = 1.0)
    viewer->addCoordinateSystem(1.0);
    //viewer->initCameraParameters();

    // Main loop to keep the visualizer window open
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

#endif // POINTCLOUD_PREPROCESSING_H
