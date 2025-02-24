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

#include <pcl/common/pca.h>
#include <pcl/surface/convex_hull.h>

#include <omp.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;


//============Helper function to downsample open3d point cloud=======================
inline std::shared_ptr<open3d::geometry::PointCloud> downSamplePointCloudOpen3d(
    const std::shared_ptr<open3d::geometry::PointCloud>& pcd, double voxel_size) {
    
    auto downsampled_pcd = pcd->VoxelDownSample(voxel_size);
    std::cout << "Downsampled point cloud has " 
              << downsampled_pcd->points_.size() << " points." << std::endl;
    return downsampled_pcd;
}

//===========Helper function to downsample the point cloud. PCL ==============================================
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
//============================================================================================================

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

//======================================PCA (Principle Component Analysis)(PCL)==========================================================================================================

template <typename PointT>
PCLResult PrincipleComponentAnalysis(const std::string& file_path,
                                     float voxel_size = 0.45f,
                                     float slope_threshold = 20.0f,  // degrees
                                     int k = 10)
{
    PCLResult result;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Allocate result point clouds.
    result.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.outlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
    result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();

    // Load the original point cloud.
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", file_path.c_str());
        return result;
    }
    std::cout << "[Block 1] Loaded " << cloud->size() << " points from " << file_path << std::endl;

    // (Optional) Downsample the cloud.
    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxel_size);
    // If no downsampling is desired, simply use the loaded cloud.
    //result.downsampled_cloud = cloud;

    // Build a kd-tree for neighborhood searches.
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(result.downsampled_cloud);

    // Prepare the normals result.downsampled_cloud.
    normals->points.resize(result.downsampled_cloud->points.size());
    normals->width    = result.downsampled_cloud->width;
    normals->height   = result.downsampled_cloud->height;
    normals->is_dense = result.downsampled_cloud->is_dense;

    // Pre-allocate temporary containers.
    std::vector<int> neighbor_indices(k);
    std::vector<float> sqr_distances(k);
    typename pcl::PointCloud<PointT>::Ptr local_cloud(new pcl::PointCloud<PointT>());
    local_cloud->points.reserve(k);

    // Process each point.
    for (size_t i = 0; i < result.downsampled_cloud->points.size(); i++)
    {
        // Find k-nearest neighbors for current point.
        if (tree.nearestKSearch(result.downsampled_cloud->points[i], k, neighbor_indices, sqr_distances) > 0)
        {
            // Build the local neighborhood.
            local_cloud->points.resize(neighbor_indices.size());
            for (size_t j = 0; j < neighbor_indices.size(); j++)
                local_cloud->points[j] = result.downsampled_cloud->points[neighbor_indices[j]];

            // Use PCL PCA on the local cloud.
            pcl::PCA<PointT> pca;
            pca.setInputCloud(local_cloud);
            // Directly obtain the eigenvectors.
            Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
            // The first column (smallest eigenvalue) represents the normal.
            Eigen::Vector3f normal = eigen_vectors.col(0);

            normals->points[i].normal_x = normal.x();
            normals->points[i].normal_y = normal.y();
            normals->points[i].normal_z = normal.z();

            // Compute the slope (angle with vertical).
            float dot_product = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
            float slope = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);

            // Classify based on the slope.
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
    result.pcl_method = "PCA";
    return result;
}

//===========================================RANSAC Plane Segmentation (OPEN3D)=======================================================================

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
    result.plane_model = plane_model;

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

//===========================================RANSAC Plane Segmentation (PCL)=======================================================================

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
  
  // Store the plane coefficients in the result structure
  result.plane_coefficients = coefficients;
  
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

//===========================================PROSAC Plane Segmentation (PCL)=======================================================================

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

  // Store the plane coefficients in the result structure
  result.plane_coefficients = coefficients;

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

//===========================================RANSAC Plane Based Centroid Detection (OPEN3D)=======================================================================

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

//====================================Least Squares Plane Fitting (OPEN3D)=======================================================================

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

//=======================================Least of Median Square Plane Fitting (PCL)=======================================================================

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
  
  // Store the plane coefficients in the result structure
  result.plane_coefficients = coefficients;
  
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

//===========================================Average 3D Gradient (PCL)=======================================================================

PCLResult Average3DGradientSegmentation(const std::string &file_path,
                                          double angle_threshold,
                                          double voxel_size)
{
    PCLResult result;
    result.downsampled_cloud = pcl::make_shared<PointCloudT>();
    result.inlier_cloud = pcl::make_shared<PointCloudT>();
    result.outlier_cloud = pcl::make_shared<PointCloudT>();
    result.plane_coefficients = pcl::make_shared<pcl::ModelCoefficients>();
    result.pcl_method = "Average3DGradient";

    // Load the point cloud from file.
    PointCloudT::Ptr cloud(new PointCloudT);
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", file_path.c_str());
        return result;
    }
    std::cout << "Loaded point cloud with " << cloud->points.size() << " points." << std::endl;

    // Downsample the cloud using VoxelGrid filter.
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
    vg.filter(*result.downsampled_cloud);
    std::cout << "Downsampled point cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;

    // Compute normals for the downsampled cloud.
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(result.downsampled_cloud);
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setKSearch(100);
    //ne.setRadiusSearch(50);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // For each point, calculate the angle between its normal and the vertical vector (0,0,1).
    // A point is considered an inlier (flat region) if the angle (in degrees) is less than angle_threshold.
    for (size_t i = 0; i < result.downsampled_cloud->points.size(); ++i) {
        // Ensure the normal is valid.
        pcl::Normal n = normals->points[i];
        if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z)) {
            // Add invalid normals to outliers.
            result.outlier_cloud->points.push_back(result.downsampled_cloud->points[i]);
            continue;
        }
        // The vertical vector is (0, 0, 1). Dot product is simply n.normal_z (if normals are unit length).
        double angle_rad = std::acos(n.normal_z);
        double angle_deg = angle_rad * 180.0 / M_PI;
        
        if (angle_deg <= angle_threshold) {
            // If the slope (angle from vertical) is below the threshold, consider this point as inlier.
            result.inlier_cloud->points.push_back(result.downsampled_cloud->points[i]);
        } else {
            result.outlier_cloud->points.push_back(result.downsampled_cloud->points[i]);
        }
    }

    std::cout << "Inlier (flat) cloud has " << result.inlier_cloud->points.size() << " points." << std::endl;
    std::cout << "Outlier cloud has " << result.outlier_cloud->points.size() << " points." << std::endl;

    return result;
}

//===========================================RegionGrowing Segmentation (PCL)=======================================================================

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


//===============================Calculate Roughness (PCL)====================================================================================
// Function to calculate roughness based on the PROSAC segmentation result.
// The function computes the roughness from the downsampled cloud using the plane coefficients.
inline double calculateRoughnessPCL(const PCLResult &result)
{
  if (result.plane_coefficients->values.size() < 4 || result.inlier_cloud->points.empty())
  {
    std::cerr << "Invalid plane coefficients or empty inlier cloud. Cannot compute roughness." << std::endl;
    return -1.0;
  }
  
  // Extract plane parameters (ax + by + cz + d = 0).
  double a = result.plane_coefficients->values[0];
  double b = result.plane_coefficients->values[1];
  double c = result.plane_coefficients->values[2];
  double d = result.plane_coefficients->values[3];
  double norm = std::sqrt(a * a + b * b + c * c);
  
  double sum_squared = 0.0;
  size_t N = result.inlier_cloud->points.size();
  
  for (const auto &pt : result.inlier_cloud->points)
  {
    double distance = std::abs(a * pt.x + b * pt.y + c * pt.z + d) / norm;
    sum_squared += distance * distance;
  }
  
  return std::sqrt(sum_squared / static_cast<double>(N));
}
//===============================Calculate Roughness (OPEN3D)============================================================

inline double calculateRoughnessOpen3D(const OPEN3DResult &result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points_.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    // Unpack plane parameters: ax + by + cz + d = 0.
    double a = result.plane_model[0];
    double b = result.plane_model[1];
    double c = result.plane_model[2];
    double d = result.plane_model[3];
    double norm = std::sqrt(a*a + b*b + c*c);

    double sum_squared = 0.0;
    size_t N = result.inlier_cloud->points_.size();

    for (const auto &pt : result.inlier_cloud->points_) {
        double distance = std::abs(a * pt(0) + b * pt(1) + c * pt(2) + d) / norm;
        sum_squared += distance * distance;
    }
    
    return std::sqrt(sum_squared / static_cast<double>(N));
}


//=============================Calculate Relief (OPEN3D)==========================================================================

inline double calculateReliefOpen3D(const OPEN3DResult &result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points_.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    double z_min = std::numeric_limits<double>::max();
    double z_max = std::numeric_limits<double>::lowest();

    // Iterate through inlier points and find min and max z values.
    for (const auto &pt : result.inlier_cloud->points_) {
        double z = pt(2);
        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
    }
    
    return z_max - z_min;
}


//=============================Calculate Relief (PCL)==========================================================================
// Calculate relief from the inlier cloud (safe landing zone).
inline double calculateReliefPCL(const PCLResult &result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    double z_min = std::numeric_limits<double>::max();
    double z_max = std::numeric_limits<double>::lowest();

    // Iterate through inlier points and compute min and max z values.
    for (const auto &pt : result.inlier_cloud->points) {
        double z = pt.z;
        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
    }
    
    return z_max - z_min;
}

//=============================Calculate Data Confidence (OPEN3D)==========================================================================

// Calculate data confidence for Open3DResult.
// It computes the convex hull of the inlier cloud and returns N (number of points)
// divided by the convex hull area.
inline double calculateDataConfidenceOpen3D(const OPEN3DResult &result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points_.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }
    
    size_t N = result.inlier_cloud->points_.size();

    // Compute convex hull of the inlier cloud.
    std::shared_ptr<open3d::geometry::TriangleMesh> hull_mesh;
    std::vector<size_t> hull_indices;
    std::tie(hull_mesh, hull_indices) = result.inlier_cloud->ComputeConvexHull();

    if (!hull_mesh) {
        std::cerr << "Error: Convex hull computation failed." << std::endl;
        return -1.0;
    }
    
    double area = hull_mesh->GetSurfaceArea();
    if (area <= 0.0) {
        std::cerr << "Error: Computed hull area is non-positive." << std::endl;
        return -1.0;
    }
    
    double data_confidence = static_cast<double>(N) / area;
    return data_confidence;
}



//=============================Calculate Data Confidence (PCL)=======================================================================
// Calculate data confidence for PCLResult.
// It computes the 2D convex hull (projecting the inlier cloud) and returns N divided by the hull area.
inline double calculateDataConfidencePCL(const PCLResult &result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    size_t N = result.inlier_cloud->points.size();
    
    // Compute the convex hull of the inlier cloud projected onto a plane.
    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(result.inlier_cloud);
    chull.setDimension(2);
    
    PointCloudT::Ptr hull_points(new PointCloudT);
    std::vector<pcl::Vertices> polygons;
    chull.reconstruct(*hull_points, polygons);
    
    if (polygons.empty() || hull_points->points.empty()) {
        std::cerr << "Error: Convex hull could not be computed." << std::endl;
        return -1.0;
    }
    
    // Compute the area of the first polygon using the shoelace formula.
    double area = 0.0;
    const std::vector<int> &indices = polygons[0].vertices;
    size_t n = indices.size();
    if (n < 3) {
        std::cerr << "Error: Convex hull does not have enough points to form an area." << std::endl;
        return -1.0;
    }
    
    for (size_t i = 0; i < n; i++) {
        const auto &p1 = hull_points->points[indices[i]];
        const auto &p2 = hull_points->points[indices[(i + 1) % n]];
        area += (p1.x * p2.y - p2.x * p1.y);
    }
    area = std::abs(area) / 2.0;
    
    if (area <= 0.0) {
        std::cerr << "Error: Computed hull area is non-positive." << std::endl;
        return -1.0;
    }
    
    double data_confidence = static_cast<double>(N) / area;
    return data_confidence;
}

#endif 
