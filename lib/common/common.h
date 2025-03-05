#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#include <open3d/Open3D.h>


#include <variant>

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

template <typename PointT>
using CloudInput = std::variant<std::string, typename pcl::PointCloud<PointT>::Ptr>;


//======================================STRUCT TO HOLD PCL RESULT ============================================
struct PCLResult {
    PointCloudT::Ptr downsampled_cloud;
    PointCloudT::Ptr inlier_cloud;
    PointCloudT::Ptr outlier_cloud;
    std::string pcl_method;
    pcl::ModelCoefficients::Ptr plane_coefficients;
  };
  //=======================================STRUCT TO HOLD OPEN3D RESULT==========================================
  struct OPEN3DResult {
      std::shared_ptr<open3d::geometry::PointCloud> inlier_cloud;
      std::shared_ptr<open3d::geometry::PointCloud> outlier_cloud;
      std::shared_ptr<open3d::geometry::PointCloud> downsampled_cloud;
      std::string open3d_method;
      Eigen::Vector4d plane_model;  // To store the plane model: [a, b, c, d]
  };
//============Helper function to downsample open3d point cloud=======================
inline std::shared_ptr<open3d::geometry::PointCloud> downSamplePointCloudOpen3d(
    const std::shared_ptr<open3d::geometry::PointCloud>& pcd, double voxelSize) {
    
    auto downsampled_pcd = pcd->VoxelDownSample(voxelSize);
    std::cout << "Downsampled point cloud has " 
              << downsampled_pcd->points_.size() << " points." << std::endl;
    return downsampled_pcd;
}

//====================== Helper function to downsample the point cloud. PCL ==============================================
template <typename PointT>
inline void downsamplePointCloudPCL(const typename pcl::PointCloud<PointT>::Ptr &input_cloud,
                                     typename pcl::PointCloud<PointT>::Ptr &output_cloud,
                                     float voxelSize)
{
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(voxelSize, voxelSize, voxelSize);
    vg.filter(*output_cloud);
}
//============================================================================================================

//-----------------------------------Helper function to load file path or pointcloud -------------------------
// Helper function to load the point cloud and return both the cloud and the flag.
template <typename PointT>
inline std::pair<typename pcl::PointCloud<PointT>::Ptr, bool> loadCloudWithFlag(const CloudInput<PointT>& input) {
    // Create a new point cloud pointer.
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    bool performDownsampling = false;

    // Check the type of input.
    if (std::holds_alternative<std::string>(input)) {
        // Input is a file path.
        std::string file_path = std::get<std::string>(input);
        if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1) {
            std::cerr << "Failed to load " << file_path << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Loaded cloud with " << cloud->points.size() << " points." << std::endl;
        performDownsampling = true;
    } else {
        // Input is already a point cloud.
        cloud = std::get<typename pcl::PointCloud<PointT>::Ptr>(input);
        std::cout << "Using provided cloud with " << cloud->points.size() << " points." << std::endl;
    }

    return {cloud, performDownsampling};
}

//====================================== VISUALIZATION PCL ====================================================

inline void visualizePCL(const PCLResult &result, const std::string& cloud = "both")
{
  // Create a visualizer object.
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer(result.pcl_method + " PCL RESULT "));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);

  // Add the outlier cloud (red) if available.
  if (result.outlier_cloud && !result.outlier_cloud->empty() && (cloud == "outlier_cloud" || cloud == "both"))
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> outlierColorHandler(result.outlier_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZI>(result.outlier_cloud, outlierColorHandler, "non_plane_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "non_plane_cloud");
  }
  
  // Add the inlier cloud (green) if available.
  if (result.inlier_cloud && !result.inlier_cloud->empty() && (cloud == "inlier_cloud" || cloud == "both"))
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> inlierColorHandler(result.inlier_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZI>(result.inlier_cloud, inlierColorHandler, "plane_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "plane_cloud");
  }
  
  // Main loop to keep the visualizer window open.
  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

//==========================================VISULAIZE OPEN3D=====================================================================================

// Visualization function for RANSAC plane segmentation result.
inline void visualizeOPEN3D(const OPEN3DResult& result) {
    // Clone the point clouds to avoid modifying the originals.
    auto inlier_cloud = std::make_shared<open3d::geometry::PointCloud>(*result.inlier_cloud);
    auto outlier_cloud = std::make_shared<open3d::geometry::PointCloud>(*result.outlier_cloud);

    // Set the colors: inliers to green and outliers to red.
    inlier_cloud->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0)); // Green
    outlier_cloud->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0)); // Red

    // Combine the point clouds into a vector for visualization.
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    geometries.push_back(inlier_cloud);
    geometries.push_back(outlier_cloud);

    // Launch the visualizer.
    open3d::visualization::DrawGeometries(geometries, result.open3d_method + " OPEN3D  Result", 800, 600);
}

#endif 