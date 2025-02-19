#ifndef HAZARD_METRICES_H
#define HAZARD_METRICES_H

#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/lmeds.h>  

#include <pcl/filters/extract_indices.h>

#include <pcl/surface/mls.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/common.h> 

#include <eigen3/Eigen/Dense>
#include <open3d/Open3D.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;


//#include <pcl/common/pca.h>

//----------------------------------------------------------------------------
// Function: computeNormalsAndClassifyPoints
// Computes normals using PCA on local neighborhoods, calculates slopes,
// and classifies points into inliers (slope ≤ threshold) and outliers (slope > threshold).
//----------------------------------------------------------------------------

inline std::shared_ptr<open3d::geometry::PointCloud> downSamplePointCloudOpen3d(
    const std::shared_ptr<open3d::geometry::PointCloud>& pcd, double voxel_size) {
    
    auto downsampled_pcd = pcd->VoxelDownSample(voxel_size);
    std::cout << "Downsampled point cloud has " 
              << downsampled_pcd->points_.size() << " points." << std::endl;
    return downsampled_pcd;
}

template <typename PointT>
void downsamplePointCloudPCL(typename pcl::PointCloud<PointT>::ConstPtr input_cloud,
                          typename pcl::PointCloud<PointT>::Ptr output_cloud,
                          float voxel_size = 0.05f)  // Default voxel size: 5 cm
{
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(input_cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);  // Set voxel size
    voxel_grid.filter(*output_cloud);

    std::cout << "Downsampled point cloud: " << output_cloud->size() << " points." << std::endl;
}



// Helper function to downsample the point cloud.
template <typename PointT>
inline void downsamplePointCloudPCL(const typename pcl::PointCloud<PointT>::Ptr &input_cloud,
                                     typename pcl::PointCloud<PointT>::Ptr &output_cloud,
                                     float leaf_size)
{
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*output_cloud);
}
//======================================STRUCT TO HOLD PCL RESULT ============================================
struct PCLResult {
  PointCloudT::Ptr downsampled_cloud;
  PointCloudT::Ptr inlier_cloud;
  PointCloudT::Ptr outlier_cloud;
  std::string pcl_method;
};
//=======================================STRUCT TO HOLD OPEN3D RESULT==========================================
struct OPEN3DResult {
    std::shared_ptr<open3d::geometry::PointCloud> inlier_cloud;
    std::shared_ptr<open3d::geometry::PointCloud> outlier_cloud;
    std::shared_ptr<open3d::geometry::PointCloud> downsampled_cloud;
    std::string open3d_method;
};
//====================================== VISUALIZATION PCL ====================================================
inline void visualizePCL(const PCLResult &result)
{
  // Create a visualizer object.
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(result.pcl_method + " PCL RESULT "));
  viewer->setBackgroundColor(1.0, 1.0, 1.0);

  // Add the outlier cloud (red) if available.
  if (result.outlier_cloud && !result.outlier_cloud->empty())
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> outlierColorHandler(result.outlier_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(result.outlier_cloud, outlierColorHandler, "non_plane_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "non_plane_cloud");
  }
  
  // Add the inlier cloud (green) if available.
  if (result.inlier_cloud && !result.inlier_cloud->empty())
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> inlierColorHandler(result.inlier_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(result.inlier_cloud, inlierColorHandler, "plane_cloud");
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
inline void VisualizeOPEN3D(const OPEN3DResult& result) {
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

//================================================================================================================================================


template <typename PointT>
PCLResult computeNormalsAndClassifyPoints(const std::string& file_path,
                                          pcl::PointCloud<pcl::Normal>::Ptr normals,
                                          float voxel_size = 0.45f,
                                          float slope_threshold = 20.0f,  // degrees
                                          int k = 10)  // k-nearest neighbors
{
    PCLResult result;
    
    // Allocate the result point clouds.
    result.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.outlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();

    // Load the original point cloud.
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s\n", file_path.c_str());
        return result;
    }
    std::cout << "[Block 1] Loaded " << cloud->size() << " points from " << file_path << std::endl;

    // (Optional) Downsample the cloud. If you wish to downsample, uncomment and adjust the following:
    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxel_size);
    //std::cout << "Pointcloud downsampled " << result.downsampled_cloud->size();
    // For now, we'll assume no downsampling is applied and just use the loaded cloud.
    result.downsampled_cloud = cloud;

    // Build a kd-tree for neighborhood searches.
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);

    // Prepare the normals cloud.
    normals->points.resize(cloud->points.size());
    normals->width    = cloud->width;
    normals->height   = cloud->height;
    normals->is_dense = cloud->is_dense;

    std::vector<int> neighbor_indices(k);
    std::vector<float> sqr_distances(k);

    // Process each point in the cloud.
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        if (tree.nearestKSearch(cloud->points[i], k, neighbor_indices, sqr_distances) > 0)
        {
            // Compute local centroid and covariance.
            Eigen::Vector4f local_centroid;
            pcl::compute3DCentroid(*cloud, neighbor_indices, local_centroid);

            Eigen::Matrix3f covariance;
            pcl::computeCovarianceMatrixNormalized(*cloud, neighbor_indices, local_centroid, covariance);

            // The eigenvector corresponding to the smallest eigenvalue is the surface normal.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
            Eigen::Vector3f normal = solver.eigenvectors().col(0);

            normals->points[i].normal_x = normal.x();
            normals->points[i].normal_y = normal.y();
            normals->points[i].normal_z = normal.z();

            // Calculate slope as the angle with vertical (0,0,1).
            float dot_product = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
            float slope = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);

            // Classify point based on slope threshold.
            if (slope <= slope_threshold)
                result.inlier_cloud->push_back(cloud->points[i]);  // Safe landing point.
            else
                result.outlier_cloud->push_back(cloud->points[i]);   // Unsafe point.
        }
        else
        {
            normals->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
            normals->points[i].normal_y = std::numeric_limits<float>::quiet_NaN();
            normals->points[i].normal_z = std::numeric_limits<float>::quiet_NaN();
        }
    }

    std::cout << "Inliers (slope ≤ " << slope_threshold << "°): " << result.inlier_cloud->size() << std::endl;
    std::cout << "Outliers (slope > " << slope_threshold << "°): " << result.outlier_cloud->size() << std::endl;
    result.pcl_method="PCA";
    return result;
}

//=========================================================================================================================


// RANSAC plane segmentation function.
// It loads a point cloud from file, downsamples it internally,
// performs RANSAC plane segmentation, and splits the downsampled cloud
// into inlier and outlier point clouds.
inline OPEN3DResult RansacPlaneSegmentation(
    const std::string& file_path,
    double voxel_size,
    double distance_threshold,
    int ransac_n,
    int num_iterations)
{
    OPEN3DResult result;

    // Load the point cloud from file.
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(file_path, *pcd)) {
        std::cerr << "Error: Unable to read point cloud from " << file_path << std::endl;
        return result;  // Return empty result on error.
    }
    std::cout << "Loaded point cloud with " << pcd->points_.size() << " points." << std::endl;

    // Downsample the point cloud.
    result.downsampled_cloud = downSamplePointCloudOpen3d(pcd, voxel_size);
    std::cout << "Downsampled point cloud now has " << result.downsampled_cloud->points_.size() << " points." << std::endl;

    // Ensure the downsampled point cloud has color information.
    if (!result.downsampled_cloud->HasColors()) {
        result.downsampled_cloud->colors_.resize(result.downsampled_cloud->points_.size(), Eigen::Vector3d(1, 1, 1));
    }

    // Perform RANSAC plane segmentation.
    // 'SegmentPlane' returns a pair: the plane model and the vector of inlier indices.
    double probability = 0.9999;
    Eigen::Vector4d plane_model;
    auto [plane, indices] = result.downsampled_cloud->SegmentPlane(distance_threshold, ransac_n, num_iterations, probability);
    plane_model = plane;
    std::cout << "Plane model: " << plane_model.transpose() << std::endl;
    std::cout << "Found " << indices.size() << " inliers for the plane." << std::endl;

    // Create new point clouds for inliers and outliers.
    result.inlier_cloud = std::make_shared<open3d::geometry::PointCloud>();
    result.outlier_cloud = std::make_shared<open3d::geometry::PointCloud>();
  

    // Build a set of inlier indices for quick lookup.
    std::unordered_set<size_t> inlier_set(indices.begin(), indices.end());
    size_t total_points = result.downsampled_cloud->points_.size();

    // Loop over each point in the downsampled point cloud and split the points.
    for (size_t i = 0; i < total_points; ++i) {
        if (inlier_set.find(i) != inlier_set.end()) {
            result.inlier_cloud->points_.push_back(result.downsampled_cloud->points_[i]);
            if (!result.downsampled_cloud->colors_.empty()) {
                result.inlier_cloud->colors_.push_back(result.downsampled_cloud->colors_[i]);
            }
        } else {
            result.outlier_cloud->points_.push_back(result.downsampled_cloud->points_[i]);
            if (!result.downsampled_cloud->colors_.empty()) {
                result.outlier_cloud->colors_.push_back(result.downsampled_cloud->colors_[i]);
            }
        }
    }

    std::cout << "Inlier cloud has " << result.inlier_cloud->points_.size() << " points." << std::endl;
    std::cout << "Outlier cloud has " << result.outlier_cloud->points_.size() << " points." << std::endl;
    result.open3d_method ="RANSAC";
    return result;
}





// Function to perform RANSAC segmentation.
// Parameters:
// - filePath: Path to the input PCD file.
// - voxelSize: Leaf size for the voxel grid downsampling.
// - distanceThreshold: Maximum distance from the plane for a point to be considered an inlier.
// - maxIterations: Maximum number of iterations for the RANSAC algorithm.
inline PCLResult performRANSAC(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
{
  // Initialize the result structure with new point clouds
  PCLResult result;
  result.downsampled_cloud = pcl::make_shared<PointCloudT>();
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();

  // Load the point cloud from file
  PointCloudT::Ptr cloud(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1)
  {
    PCL_ERROR("Couldn't read file %s \n", file_path.c_str());
    return result;
  }
  std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;

  // Use the provided downsampling function to downsample the cloud
  downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
  // If detecting a table or ground plane → SACMODEL_PLANE + SAC_RANSAC
  // If extracting pipes or poles → SACMODEL_CYLINDER + SAC_RANSAC
  // If detecting objects with circular features → SACMODEL_CIRCLE3D + SAC_LMEDS
  // If handling large noisy datasets → SACMODEL_PLANE + SAC_RRANSAC
  // If prioritizing accuracy over speed → SACMODEL_PLANE + SAC_MLESAC
  // Extract inliers (the plane) from the cloud
  // Set up RANSAC segmentation for a plane model
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(maxIterations);
  seg.setDistanceThreshold(distanceThreshold);
  seg.setInputCloud(result.downsampled_cloud);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty())
  {
    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    return result;
  }
  std::cout << "RANSAC found " << inliers->indices.size() << " inliers." << std::endl;

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(result.downsampled_cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*result.inlier_cloud);

  // Extract outliers (points not on the plane)
  extract.setNegative(true);
  extract.filter(*result.outlier_cloud);
  result.pcl_method="RANSAC";
  return result;
}


inline PCLResult performPROSAC(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
{
  // Initialize the result structure with new point clouds
  PCLResult result;
  result.downsampled_cloud = pcl::make_shared<PointCloudT>();
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();

  // Load the point cloud from file
  PointCloudT::Ptr cloud(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1)
  {
    PCL_ERROR("Couldn't read file %s \n", file_path.c_str());
    return result;
  }
  std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;

  // Use the provided downsampling function to downsample the cloud
  downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
  // If detecting a table or ground plane → SACMODEL_PLANE + SAC_RANSAC
  // If extracting pipes or poles → SACMODEL_CYLINDER + SAC_RANSAC
  // If detecting objects with circular features → SACMODEL_CIRCLE3D + SAC_LMEDS
  // If handling large noisy datasets → SACMODEL_PLANE + SAC_RRANSAC
  // If prioritizing accuracy over speed → SACMODEL_PLANE + SAC_MLESAC
  // Extract inliers (the plane) from the cloud
  // Set up PROSAC segmentation for a plane model
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_PROSAC);
  seg.setMaxIterations(maxIterations);
  seg.setDistanceThreshold(distanceThreshold);
  seg.setInputCloud(result.downsampled_cloud);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty())
  {
    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    return result;
  }
  std::cout << "PROSAC found " << inliers->indices.size() << " inliers." << std::endl;

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(result.downsampled_cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*result.inlier_cloud);

  // Extract outliers (points not on the plane)
  extract.setNegative(true);
  extract.filter(*result.outlier_cloud);
  result.pcl_method="PROSAC";
  return result;
}


inline std::tuple<Eigen::Vector3d, std::shared_ptr<open3d::geometry::PointCloud>>
RansacPlaneAndComputeCentroid(
    const std::string& file_path,
    double voxel_size,
    double distance_threshold,
    int ransac_n,
    int num_iterations) {

    // Perform RANSAC segmentation to detect the ground plane.
    OPEN3DResult result = RansacPlaneSegmentation(file_path, voxel_size, distance_threshold,
                                                  ransac_n, num_iterations);

    Eigen::Vector3d centroid(0, 0, 0);

    // Check if the inlier point cloud is empty.
    if (result.inlier_cloud->points_.empty()) {
        std::cerr << "Warning: No inlier points provided. Returning zero centroid." << std::endl;
        return std::make_tuple(centroid,result.downsampled_cloud);
    }

    // Iterate over the inlier points (stored in the points_ vector) to compute the centroid.
    for (const auto &pt : result.inlier_cloud->points_) {
        centroid += pt;
    }
    centroid /= static_cast<double>(result.inlier_cloud->points_.size());

    return std::make_tuple(centroid,result.downsampled_cloud);
}

inline OPEN3DResult LeastSquaresPlaneFitting(const std::string &file_path, double voxel_size, double distance_threshold) {
    // Load the point cloud from file.

    OPEN3DResult result;
    int ransac_n =3;
    int num_iterations = 1000;
    auto [centroid,downsampled_pcd] = RansacPlaneAndComputeCentroid(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);

    // auto pcd = open3d::io::CreatePointCloudFromFile(file_path);
    // if (!pcd || pcd->points_.empty()) {
    //     throw std::runtime_error("Failed to load point cloud from file: " + file_path);
    // }
    // std::cout << "Loaded point cloud with " << pcd->points_.size() << " points." << std::endl;

    // // // Downsample the point cloud using a voxel grid.
    // auto downsampled_pcd = pcd->VoxelDownSample(voxel_size);
    // std::cout << "Downsampled point cloud to " << downsampled_pcd->points_.size() << " points." << std::endl;

    // Compute the covariance matrix.
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto &pt : downsampled_pcd->points_) {
        Eigen::Vector3d diff = pt - centroid;
        covariance += diff * diff.transpose();
    }

    // Compute the eigen decomposition of the covariance matrix.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }
    // The eigenvector corresponding to the smallest eigenvalue is the plane normal.
    Eigen::Vector3d normal = eigen_solver.eigenvectors().col(0);
    normal.normalize();

    // Compute the plane equation: n.dot(x) + d = 0.
    double d = -normal.dot(centroid);
    std::cout << "Fitted plane: normal = " << normal.transpose() << ", d = " << d << std::endl;

    // Separate points into inliers and outliers based on distance from the plane.
    auto inliers = std::make_shared<open3d::geometry::PointCloud>();
    auto outliers = std::make_shared<open3d::geometry::PointCloud>();

    for (const auto &pt : downsampled_pcd->points_) {
        double distance = std::fabs(normal.dot(pt) + d);
        if (distance < distance_threshold) {
            inliers->points_.push_back(pt);
            // Color inliers green.
            inliers->colors_.push_back({0.0, 1.0, 0.0});
        } else {
            outliers->points_.push_back(pt);
            // Color outliers red.
            outliers->colors_.push_back({1.0, 0.0, 0.0});
        }
    }

    std::cout << "Found " << inliers->points_.size() << " inliers and "
              << outliers->points_.size() << " outliers." << std::endl;
    result.inlier_cloud = inliers;
    result.outlier_cloud = outliers;
    result.open3d_method ="LEAST SQUARE PLANE FITTING";
    return result;
    //return std::make_tuple(inliers, outliers);
}


inline PCLResult performLMEDS(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
{
  // Initialize the result structure with new point clouds
  PCLResult result;
  result.downsampled_cloud = pcl::make_shared<PointCloudT>();
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();

  // Load the point cloud from file
  PointCloudT::Ptr cloud(new PointCloudT);
  if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1)
  {
    PCL_ERROR("Couldn't read file %s \n", file_path.c_str());
    return result;
  }
  std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;

  // Use the provided downsampling function to downsample the cloud
  downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
  // If detecting a table or ground plane → SACMODEL_PLANE + SAC_RANSAC
  // If extracting pipes or poles → SACMODEL_CYLINDER + SAC_RANSAC
  // If detecting objects with circular features → SACMODEL_CIRCLE3D + SAC_LMEDS
  // If handling large noisy datasets → SACMODEL_PLANE + SAC_RRANSAC
  // If prioritizing accuracy over speed → SACMODEL_PLANE + SAC_MLESAC
  // Extract inliers (the plane) from the cloud
  // Set up LMEDS segmentation for a plane model
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_LMEDS);
  seg.setMaxIterations(maxIterations);
  seg.setDistanceThreshold(distanceThreshold);
  seg.setInputCloud(result.downsampled_cloud);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty())
  {
    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    return result;
  }
  std::cout << "LMEDS found " << inliers->indices.size() << " inliers." << std::endl;

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(result.downsampled_cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*result.inlier_cloud);

  // Extract outliers (points not on the plane)
  extract.setNegative(true);
  extract.filter(*result.outlier_cloud);
  result.pcl_method="LMEDS";
  return result;
}


// Function to compute normals using the Average 3D Gradient method,
// classify points based on a slope threshold, and return inliers and outliers.
// std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
// computeNormalsUsingAverage3DGradient(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
//                                       float slope_threshold = 20.0f,
//                                       float max_depth_change_factor = 0.02f,
//                                       float normal_smoothing_size = 10.0f)
// {
//     // Container for computed normals.
//     pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

//     // Set up the integral image normal estimation (using the AVERAGE_3D_GRADIENT method).
//     pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     ne.setNormalEstimationMethod(
//         ne.AVERAGE_3D_GRADIENT);
//     ne.setMaxDepthChangeFactor(max_depth_change_factor);
//     ne.setNormalSmoothingSize(normal_smoothing_size);
//     ne.setInputCloud(cloud);
//     ne.compute(*normals);

//     // Containers for inliers (green) and outliers (red).
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr inliers(new pcl::PointCloud<pcl::PointXYZRGB>);
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers(new pcl::PointCloud<pcl::PointXYZRGB>);

//     // For each point, compute the angle between its normal and the vertical vector (0, 0, 1).
//     for (size_t i = 0; i < cloud->points.size(); i++)
//     {
//         const pcl::PointXYZ &pt = cloud->points[i];
//         const pcl::Normal &n = normals->points[i];

//         Eigen::Vector3f normal(n.normal_x, n.normal_y, n.normal_z);
//         float dot = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
//         float angle = std::acos(dot) * 180.0f / static_cast<float>(M_PI);

//         // Create a colored point with the original coordinates.
//         pcl::PointXYZRGB colored_point;
//         colored_point.x = pt.x;
//         colored_point.y = pt.y;
//         colored_point.z = pt.z;

//         if (angle <= slope_threshold)
//         {
//             // Inlier (slope is below the threshold): color it green.
//             colored_point.r = 0;
//             colored_point.g = 255;
//             colored_point.b = 0;
//             inliers->push_back(colored_point);
//         }
//         else
//         {
//             // Outlier (slope exceeds the threshold): color it red.
//             colored_point.r = 255;
//             colored_point.g = 0;
//             colored_point.b = 0;
//             outliers->push_back(colored_point);
//         }
//     }

//     return std::make_pair(inliers, outliers);
// }

// // Function to visualize the classified point clouds.
// void visualizeClassifiedCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inliers,
//                               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outliers)
// {
//     pcl::visualization::PCLVisualizer viewer("Classified Cloud Viewer");
//     viewer.setBackgroundColor(0.0, 0.0, 0.0);

//     // Add inliers (green) to the viewer.
//     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> inlier_color(inliers);
//     viewer.addPointCloud<pcl::PointXYZRGB>(inliers, inlier_color, "inliers");
//     viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "inliers");

//     // Add outliers (red) to the viewer.
//     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> outlier_color(outliers);
//     viewer.addPointCloud<pcl::PointXYZRGB>(outliers, outlier_color, "outliers");
//     viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "outliers");

//     while (!viewer.wasStopped())
//     {
//         viewer.spinOnce();
//     }
// }


// Region growing segmentation function.
// Parameters:
//   file_path      : path to the PCD file (modify this string as needed)
//   voxel_leaf    : voxel grid leaf size for downsampling
// Returns:
//   A RegionGrowingSegmentationResult structure containing the downsampled cloud,
//   the inliers (points in any region), and the outliers.
inline PCLResult regionGrowingSegmentation(
    const std::string &file_path,
    float voxel_leaf,
    int min_cluster_size = 50,         // Minimum number of points per cluster.
    int max_cluster_size = 1000000,      // Maximum number of points per cluster.
    int number_of_neighbours = 30,       // Nearest neighbours used in region growing.
    int normal_k_search = 50,            // K nearest neighbors for normal estimation.
    // Lower the smoothness threshold to about 2° (in radians) so only nearly identical normals join.
    float smoothness_threshold = 2.0 / 180.0 * M_PI,
    // Lower the curvature threshold to only allow points with very low curvature (indicative of flat surfaces).
    float curvature_threshold = 0.9,
    // Instead of a dot product threshold, provide an angle (in degrees). For instance, 20°.
    float horizontal_angle_threshold_deg = 10.0f
)
{

  // Convert the angle in degrees to a dot product threshold.
  float horizontal_dot_threshold = std::cos(horizontal_angle_threshold_deg * M_PI / 180.0f);

  // Load the input cloud from file.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) == -1) {
    std::cerr << "Failed to load " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Loaded cloud with " << cloud->points.size() << " points." << std::endl;

  // Downsample the cloud using a VoxelGrid filter.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
  voxel.filter(*cloud_downsampled);
  std::cout << "Downsampled cloud has " << cloud_downsampled->points.size() << " points." << std::endl;

  // Remove any NaN points.
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_downsampled, *cloud_downsampled, indices);

  // Estimate normals on the downsampled cloud.
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud_downsampled);
  normal_estimator.setKSearch(normal_k_search);
  normal_estimator.compute(*normals);

  // Create indices for valid points.
  pcl::IndicesPtr valid_indices(new std::vector<int>(indices));

  // Set up the region growing segmentation algorithm.
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(min_cluster_size);
  reg.setMaxClusterSize(max_cluster_size);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(number_of_neighbours);
  reg.setInputCloud(cloud_downsampled);
  reg.setIndices(valid_indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(smoothness_threshold);
  reg.setCurvatureThreshold(curvature_threshold);

  // Extract clusters.
  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);
  std::cout << "Found " << clusters.size() << " clusters." << std::endl;

  // Create containers for inliers and outliers.
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Global vertical axis (assumed here to be the Z-axis).
  Eigen::Vector3f vertical(0.0f, 0.0f, 1.0f);

  // Process each cluster: compute the average normal and check its alignment with the vertical.
  for (const auto &cluster : clusters) {
    Eigen::Vector3f avg_normal(0.0f, 0.0f, 0.0f);
    for (const auto &idx : cluster.indices) {
      const auto &n = normals->points[idx];
      avg_normal += Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
    }
    if (!cluster.indices.empty())
      avg_normal /= static_cast<float>(cluster.indices.size());

    // Normalize the average normal.
    if (avg_normal.norm() != 0)
      avg_normal.normalize();

    // Check alignment with vertical.
    float dot = std::fabs(avg_normal.dot(vertical));
    if (dot >= horizontal_dot_threshold) {
      // Cluster is horizontal; add its points to the inliers.
      for (const auto &idx : cluster.indices) {
        inliers_cloud->points.push_back(cloud_downsampled->points[idx]);
      }
    }
    else {
      // Cluster is inclined; add its points to outliers.
      for (const auto &idx : cluster.indices) {
        outliers_cloud->points.push_back(cloud_downsampled->points[idx]);
      }
    }
  }

  // Also, add any downsampled points not part of any cluster to the outliers.
  std::set<int> cluster_indices;
  for (const auto &cluster : clusters) {
    for (const auto &idx : cluster.indices) {
      cluster_indices.insert(idx);
    }
  }
  for (size_t i = 0; i < cloud_downsampled->points.size(); ++i) {
    if (cluster_indices.find(i) == cluster_indices.end()) {
      outliers_cloud->points.push_back(cloud_downsampled->points[i]);
    }
  }

  // Propagate header information.
  inliers_cloud->header = cloud_downsampled->header;
  outliers_cloud->header = cloud_downsampled->header;

  // Return the segmentation result.
  PCLResult result;
  result.inlier_cloud = inliers_cloud;
  result.outlier_cloud = outliers_cloud;
  result.pcl_method="REGION GROWING SEGMENTATION";
  return result;
}

#endif 