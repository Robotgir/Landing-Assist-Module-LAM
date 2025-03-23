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

#include <common.h>
#include <variant>

#include <pcl/features/normal_3d_omp.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/octree/octree_search.h>

#include<queue>
#include <unordered_set>

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

template <typename PointT>
using CloudInput = std::variant<std::string, typename pcl::PointCloud<PointT>::Ptr>;

using Open3DCloudInput = std::variant<std::string, std::shared_ptr<open3d::geometry::PointCloud>>;
 




//======================================PCA (Principle Component Analysis)(PCL)==========================================================================================================

// template <typename PointT>
inline PCLResult PrincipleComponentAnalysis(const CloudInput<PointT>& input,
                                     float voxelSize = 0.45f,
                                     float angleThreshold = 20.0f,
                                     int k = 10)
{
  PCLResult result;
  result.pcl_method = "Principal Component Analysis (using NormalEstimationOMP)";
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();
  result.downsampled_cloud = pcl::make_shared<PointCloudT>();

  // Load the cloud and determine if downsampling is needed.
  // (Assumes loadPCLCloud returns a pair: <loadedCloud, doDownsample>)
  auto [loadedCloud, doDownsample] = loadPCLCloud<PointT>(input);

  if (doDownsample) {
    downsamplePointCloudPCL<PointT>(loadedCloud, result.downsampled_cloud, voxelSize);
    std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
  } else {
    result.downsampled_cloud = loadedCloud;
  }

  // Compute normals in parallel using NormalEstimationOMP.
  pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  ne.setInputCloud(result.downsampled_cloud);
  ne.setKSearch(k);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals);

  // Classify points based on the computed normals and the angle threshold.
  for (size_t i = 0; i < normals->points.size(); i++) {
    Eigen::Vector3f normal(normals->points[i].normal_x,
                           normals->points[i].normal_y,
                           normals->points[i].normal_z);
    // Check for invalid normal values.
    if (std::isnan(normal.norm()) || normal.norm() == 0) {
      result.outlier_cloud->push_back(result.downsampled_cloud->points[i]);
      continue;
    }
    float dot_product = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
    float slope = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);
    if (slope <= angleThreshold)
      result.inlier_cloud->push_back(result.downsampled_cloud->points[i]);
    else
      result.outlier_cloud->push_back(result.downsampled_cloud->points[i]);
  }
  std::cout << "Inliers (slope ≤ " << angleThreshold << "°): " << result.inlier_cloud->size() << std::endl;
  std::cout << "Outliers (slope > " << angleThreshold << "°): " << result.outlier_cloud->size() << std::endl;

  return result;
}

//=========================================== PCA with Octree ========================================================================================




//===========================================RANSAC Plane Segmentation (OPEN3D)=======================================================================

inline OPEN3DResult RansacPlaneSegmentation(
    const Open3DCloudInput &input,
    double voxelSize,
    double distance_threshold,
    int ransac_n,
    int num_iterations)
{
    OPEN3DResult result;
    result.open3d_method ="RANSAC";


    // Load the point cloud from either the file or the provided pointer.
    auto [pcd, performDownsampling] = loadOpen3DCloud(input);

    // Downsample the point cloud.
    result.downsampled_cloud = downSamplePointCloudOpen3d(pcd, voxelSize);
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
    
    return result;
}


//================== Ransac Plane Segmentation (PCL) Accepts file path and cloud ===================================================================

inline PCLResult performRANSAC(const CloudInput<PointT> &input,
  float voxelSize = 0.05f,
  float distanceThreshold = 0.02f,
  int maxIterations = 100)
{
// Initialize the result structure with new point clouds.
PCLResult result;
result.pcl_method = "RANSAC";

result.downsampled_cloud = pcl::make_shared<PointCloudT>();
result.inlier_cloud = pcl::make_shared<PointCloudT>();
result.outlier_cloud = pcl::make_shared<PointCloudT>();

auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

// Downsample if the input was a file path.
if (performDownsampling)
{
// Assume downsamplePointCloudPCL is defined elsewhere.
downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
}
else
{
result.downsampled_cloud = cloud;
}

// Set up RANSAC segmentation for a plane model.
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

// Store the plane coefficients in the result structure.
result.plane_coefficients = coefficients;

// Extract the inlier cloud.
pcl::ExtractIndices<PointT> extract;
extract.setInputCloud(result.downsampled_cloud);
extract.setIndices(inliers);
extract.setNegative(false);
extract.filter(*result.inlier_cloud);

// Extract outliers (points not on the plane).
extract.setNegative(true);
extract.filter(*result.outlier_cloud);

return result;
}


//===========================================PROSAC Plane Segmentation (PCL)=======================================================================

inline PCLResult performPROSAC(const CloudInput<PointT> &input,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
{
  // Initialize the result structure with new point clouds
  PCLResult result;
  result.pcl_method="PROSAC";

  result.downsampled_cloud = pcl::make_shared<PointCloudT>();
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();

  auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

  // Downsample if the input was a file path.
  if (performDownsampling)
  {
  // Assume downsamplePointCloudPCL is defined elsewhere.
  downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
  std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
  }
  else
  {
  result.downsampled_cloud = cloud;
  }

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
  
  return result;
}

//===========================================RANSAC Plane Based Centroid Detection (OPEN3D)=======================================================================

// Modified RansacPlaneAndComputeCentroid accepting Open3DCloudInput.
inline std::tuple<Eigen::Vector3d, std::shared_ptr<open3d::geometry::PointCloud>>
RansacPlaneAndComputeCentroid(const Open3DCloudInput &input,
                              double voxelSize,
                              double distance_threshold,
                              int ransac_n,
                              int num_iterations) {
    // Load the point cloud from the input.
    auto [pcd, performDownsampling] = loadOpen3DCloud(input);
    
    // Run RANSAC segmentation (using our overloaded version that accepts Open3DCloudInput).
    OPEN3DResult result = RansacPlaneSegmentation(input, voxelSize, distance_threshold, ransac_n, num_iterations);
    
    Eigen::Vector3d centroid(0, 0, 0);
    if (result.inlier_cloud->points_.empty()) {
        std::cerr << "Warning: No inlier points provided. Returning zero centroid." << std::endl;
        return std::make_tuple(centroid, result.downsampled_cloud);
    }
    for (const auto &pt : result.inlier_cloud->points_) {
        centroid += pt;
    }
    centroid /= static_cast<double>(result.inlier_cloud->points_.size());
    
    return std::make_tuple(centroid, result.downsampled_cloud);
}

//====================================Least Squares Plane Fitting (OPEN3D)=======================================================================
// Modified LeastSquaresPlaneFitting accepting Open3DCloudInput.
inline OPEN3DResult LeastSquaresPlaneFitting(const Open3DCloudInput &input,
                                             double voxelSize,
                                             double distance_threshold) {
    OPEN3DResult result;
    result.open3d_method = "LEAST SQUARE PLANE FITTING";

    int ransac_n = 3;
    int num_iterations = 1000;
    auto [centroid, downsampled_pcd] = RansacPlaneAndComputeCentroid(input, voxelSize, distance_threshold, ransac_n, num_iterations);

    // Compute the covariance matrix.
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto &pt : downsampled_pcd->points_) {
        Eigen::Vector3d diff = pt - centroid;
        covariance += diff * diff.transpose();
    }

    // Compute eigen decomposition.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }
    // The eigenvector corresponding to the smallest eigenvalue is the plane normal.
    Eigen::Vector3d normal = eigen_solver.eigenvectors().col(0);
    normal.normalize();

    // Compute plane equation: n.dot(x) + d = 0.
    double d = -normal.dot(centroid);
    std::cout << "Fitted plane: normal = " << normal.transpose() << ", d = " << d << std::endl;

    // Separate points into inliers and outliers based on distance.
    auto inliers = std::make_shared<open3d::geometry::PointCloud>();
    auto outliers = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto &pt : downsampled_pcd->points_) {
        double dist = std::fabs(normal.dot(pt) + d);
        if (dist < distance_threshold) {
            inliers->points_.push_back(pt);
            inliers->colors_.push_back({0.0, 1.0, 0.0});  // Green for inliers.
        } else {
            outliers->points_.push_back(pt);
            outliers->colors_.push_back({1.0, 0.0, 0.0});  // Red for outliers.
        }
    }
    std::cout << "Found " << inliers->points_.size() << " inliers and "
              << outliers->points_.size() << " outliers." << std::endl;

    result.inlier_cloud = inliers;
    result.outlier_cloud = outliers;
    result.downsampled_cloud = downsampled_pcd;
    
    return result;
}


//=======================================Least of Median Square Plane Fitting (PCL)=======================================================================

inline PCLResult performLMEDS(const CloudInput<PointT> &input,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
{
  // Initialize the result structure with new point clouds
  PCLResult result;
  result.pcl_method="LMEDS";

  result.downsampled_cloud = pcl::make_shared<PointCloudT>();
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();

  auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

  // Downsample if the input was a file path.
  if (performDownsampling)
  {
  // Assume downsamplePointCloudPCL is defined elsewhere.
  downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
  std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
  }
  else
  {
  result.downsampled_cloud = cloud;
  }

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
  
  return result;
}

//===========================================Average 3D Gradient (PCL)=======================================================================

PCLResult Average3DGradient(const CloudInput<PointT> &input,
  float voxelSize,
  float neighborRadius,
  float gradientThreshold,
  float angleThreshold)
{
// Initialize the result struct.
PCLResult result;
result.pcl_method = "Average3DGradient";
result.plane_coefficients = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients());
result.downsampled_cloud = pcl::make_shared<PointCloudT>();

auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

// Downsample if the input was a file path.
if (performDownsampling)
{
// Assume downsamplePointCloudPCL is defined elsewhere.

downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
}
else
{
result.downsampled_cloud = cloud;
}

std::cout << "After voxel downsampling: " << result.downsampled_cloud->points.size() << " points." << std::endl;

// Build a KD-tree for neighbor search.
pcl::KdTreeFLANN<PointT> kdtree;
kdtree.setInputCloud(result.downsampled_cloud);
size_t numPoints = result.downsampled_cloud->points.size();

// Create vectors to store classification result.
std::vector<bool> isFlat(numPoints, false);
std::vector<float> avgGradients(numPoints, 0.0f);

// Loop over all points.
for (size_t i = 0; i < numPoints; i++) {
// Local vectors for neighbor search.
std::vector<int> neighborIndices;
std::vector<float> neighborDistances;
neighborIndices.reserve(32);
neighborDistances.reserve(32);

const pcl::PointXYZI &searchPoint = result.downsampled_cloud->points[i];

// Find neighbors within the specified radius.
int found = kdtree.radiusSearch(searchPoint, neighborRadius, neighborIndices, neighborDistances);
if (found > 3) {
float sumGradient = 0.0f;
int count = 0;
for (size_t j = 0; j < neighborIndices.size(); j++) {
if (neighborIndices[j] == i)
continue;
float dx = result.downsampled_cloud->points[neighborIndices[j]].x - searchPoint.x;
float dy = result.downsampled_cloud->points[neighborIndices[j]].y - searchPoint.y;
float dz = result.downsampled_cloud->points[neighborIndices[j]].z - searchPoint.z;
float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
if (distance > 0) {
float grad = std::fabs(dz) / distance;
sumGradient += grad;
count++;
}
}
float avgGradient = (count > 0) ? (sumGradient / count) : std::numeric_limits<float>::infinity();
avgGradients[i] = avgGradient;
// Convert gradient to an angle in degrees.
float slopeAngle = std::atan(avgGradient) * (180.0f / M_PI);
// Classify point as flat if both conditions are met.
if (avgGradient < gradientThreshold && slopeAngle < angleThreshold)
isFlat[i] = true;
} else {
isFlat[i] = false;
}
} // end loop

// Partition points into inliers and outliers.
result.inlier_cloud.reset(new PointCloudT);
result.outlier_cloud.reset(new PointCloudT);
for (size_t i = 0; i < numPoints; i++) {
if (isFlat[i])
result.inlier_cloud->points.push_back(result.downsampled_cloud->points[i]);
else
result.outlier_cloud->points.push_back(result.downsampled_cloud->points[i]);
}
result.inlier_cloud->width = result.inlier_cloud->points.size();
result.inlier_cloud->height = 1;
result.outlier_cloud->width = result.outlier_cloud->points.size();
result.outlier_cloud->height = 1;

return result;
}


//==========================Region growing segmentation function that can accept both cloud and file path ==================================


inline PCLResult regionGrowingSegmentation(
  const CloudInput<PointT> &input,
  float voxelSize = 0.45f,
  float angleThreshold = 15.0f,
  int min_cluster_size = 10,         // Minimum number of points per cluster.
  int max_cluster_size = 10000000,    // Maximum number of points per cluster.
  int number_of_neighbours = 30,     // Nearest neighbors used in region growing.
  int normal_k_search = 50,          // K nearest neighbors for normal estimation.
  float smoothness_threshold = 2.0f / 180.0f * M_PI,
  float curvature_threshold = 0.9f
)
{
  // Initialize the result struct.
  PCLResult result;

  // Load the pointcloud.
  auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

  // Downsample if the input was a file path.
  if (performDownsampling) {
    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
    std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
  } else {
    result.downsampled_cloud = cloud;
  }

  // Remove any NaN points.
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*result.downsampled_cloud, *result.downsampled_cloud, indices);

  // ------------------------------------------------------------------------
  // Compute normals in parallel using NormalEstimationOMP
  // ------------------------------------------------------------------------
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

  pcl::NormalEstimationOMP<PointT, pcl::Normal> normal_estimator;
  // Optionally, specify the number of threads (0 uses all available threads):
  // normal_estimator.setNumberOfThreads(4);
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(result.downsampled_cloud);
  normal_estimator.setKSearch(normal_k_search);
  normal_estimator.compute(*normals);

  // Use the indices from NaN removal as valid indices.
  pcl::IndicesPtr valid_indices(new std::vector<int>(indices));

  // Set up region growing segmentation.
  pcl::RegionGrowing<PointT, pcl::Normal> reg;
  reg.setMinClusterSize(min_cluster_size);
  reg.setMaxClusterSize(max_cluster_size);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(number_of_neighbours);
  reg.setInputCloud(result.downsampled_cloud);
  reg.setIndices(valid_indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(smoothness_threshold);
  reg.setCurvatureThreshold(curvature_threshold);

  // Extract clusters.
  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);
  std::cout << "Found " << clusters.size() << " clusters." << std::endl;

  // Containers for inliers and outliers.
  pcl::PointCloud<PointT>::Ptr inliers_cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr outliers_cloud(new pcl::PointCloud<PointT>);

  // Global vertical axis (assumed here as the Z-axis).
  Eigen::Vector3f vertical(0.0f, 0.0f, 1.0f);
  float horizontal_dot_threshold = std::cos(angleThreshold * M_PI / 180.0f);

  // Process clusters: determine whether each cluster is "horizontal" or not.
  for (const auto &cluster : clusters) {
    Eigen::Vector3f avg_normal(0.0f, 0.0f, 0.0f);
    for (const auto &idx : cluster.indices) {
      const auto &n = normals->points[idx];
      avg_normal += Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
    }
    if (!cluster.indices.empty())
      avg_normal /= static_cast<float>(cluster.indices.size());
    if (avg_normal.norm() != 0)
      avg_normal.normalize();

    float dot = std::fabs(avg_normal.dot(vertical));
    if (dot >= horizontal_dot_threshold) {
      // Cluster is horizontal.
      for (const auto &idx : cluster.indices) {
        inliers_cloud->points.push_back(result.downsampled_cloud->points[idx]);
      }
    } else {
      // Cluster is inclined.
      for (const auto &idx : cluster.indices) {
        outliers_cloud->points.push_back(result.downsampled_cloud->points[idx]);
      }
    }
  }

  // Add any downsampled points not part of any cluster to outliers.
  std::set<int> cluster_indices;
  for (const auto &cluster : clusters) {
    for (const auto &idx : cluster.indices) {
      cluster_indices.insert(idx);
    }
  }
  for (size_t i = 0; i < result.downsampled_cloud->points.size(); ++i) {
    if (cluster_indices.find(static_cast<int>(i)) == cluster_indices.end()) {
      outliers_cloud->points.push_back(result.downsampled_cloud->points[i]);
    }
  }

  // Propagate header information.
  inliers_cloud->header = result.downsampled_cloud->header;
  outliers_cloud->header = result.downsampled_cloud->header;

  // Build and return the result.
  result.inlier_cloud = inliers_cloud;
  result.outlier_cloud = outliers_cloud;
  result.pcl_method = "REGION GROWING SEGMENTATION (OMP Normals)";
  return result;
}

inline PCLResult regionGrowingSegmentation2(
  const CloudInput<PointT> &input,
  float voxelSize = 0.45f,
  float angleThreshold = 15.0f,     // Angle threshold for smoothness
  int min_cluster_size = 10,        // Minimum number of points per cluster
  int max_cluster_size = 10000000,  // Maximum number of points per cluster
  int number_of_neighbours = 30,    // Nearest neighbors used for PCA curvature
  int normal_k_search = 10,         // K nearest neighbors for normal estimation
  float curvature_threshold = 0.9f,
  float height_threshold = 0.5f,    // Threshold for low height points
  float smoothness_threshold = 2.0f / 180.0f * M_PI
)
{
  // Initialize the result structure.
  PCLResult result;
  result.inlier_cloud = pcl::make_shared<PointCloudT>();
  result.outlier_cloud = pcl::make_shared<PointCloudT>();
  result.downsampled_cloud = pcl::make_shared<PointCloudT>();

  // Log: Loading the point cloud.
  std::cout << "Loading point cloud..." << std::endl;
  auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);

  // Downsample if needed.
  if (!performDownsampling) {
    downsamplePointCloudPCL<PointT>(cloud, result.downsampled_cloud, voxelSize);
    std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
  } else {
    result.downsampled_cloud = cloud;
    std::cout << "No downsampling performed." << std::endl;
  }

  // ------------------------------------------------------------------------
  // Compute normals using NormalEstimationOMP.
  // ------------------------------------------------------------------------
  std::cout << "Computing normals..." << std::endl;
  // Create a KD-tree for the downsampled cloud.
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(result.downsampled_cloud);

  pcl::NormalEstimationOMP<PointT, pcl::Normal> normal_estimator;
  normal_estimator.setInputCloud(result.downsampled_cloud);
  normal_estimator.setKSearch(normal_k_search);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  normal_estimator.compute(*normals);

  // Log the first few normals.
  std::cout << "First 5 normals (NormalEstimationOMP):" << std::endl;
  for (size_t i = 0; i < 5 && i < normals->points.size(); ++i) {
    std::cout << "Normal " << i << ": ("
              << normals->points[i].normal_x << ", "
              << normals->points[i].normal_y << ", "
              << normals->points[i].normal_z << ")\n";
  }

  // ------------------------------------------------------------------------
  // Compute PCA-based curvature using eigenvalue decomposition.
  // ------------------------------------------------------------------------
  std::vector<float> pca_curvature_values(result.downsampled_cloud->points.size(), 0.0f);
  for (size_t i = 0; i < result.downsampled_cloud->points.size(); ++i) {
    std::vector<int> point_indices(number_of_neighbours);
    std::vector<float> point_squared_distances(number_of_neighbours);
    if (tree->nearestKSearch(result.downsampled_cloud->points[i],
                             number_of_neighbours,
                             point_indices,
                             point_squared_distances) > 0)
    {
      // Build matrix M using the relative coordinates of the neighbors.
      Eigen::MatrixXf M(number_of_neighbours, 3);
      for (int j = 0; j < number_of_neighbours; ++j) {
        const PointT &neighbor = result.downsampled_cloud->points[point_indices[j]];
        M(j, 0) = neighbor.x - result.downsampled_cloud->points[i].x;
        M(j, 1) = neighbor.y - result.downsampled_cloud->points[i].y;
        M(j, 2) = neighbor.z - result.downsampled_cloud->points[i].z;
      }
      // Compute the covariance matrix.
      Eigen::Matrix3f covariance_matrix = M.transpose() * M;
      // Perform eigenvalue decomposition.
      Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
      Eigen::Vector3f eigenvalues = solver.eigenvalues().real();
      int min_eigenvalue_index;
      eigenvalues.minCoeff(&min_eigenvalue_index);
      // Compute curvature (you may use lambda_min / sqrt(sum of squares) if preferred).
      float curvature_value = eigenvalues(min_eigenvalue_index) / eigenvalues.sum();
      pca_curvature_values[i] = curvature_value;
    } else {
      pca_curvature_values[i] = 0.0f;
    }
  }

  // Log PCA curvature for the first 50 points.
  std::cout << "PCA Curvature values for the first 50 points:" << std::endl;
  for (size_t i = 0; i < 50 && i < pca_curvature_values.size(); ++i) {
    std::cout << "Curvature " << i << ": " << pca_curvature_values[i] << std::endl;
  }

  // ------------------------------------------------------------------------
  // Sort the point cloud indices by height and then by curvature.
  // ------------------------------------------------------------------------
  std::vector<int> sorted_indices(result.downsampled_cloud->points.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::cout << "Sorting points by height..." << std::endl;
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&result](int i1, int i2) {
              return result.downsampled_cloud->points[i1].z < result.downsampled_cloud->points[i2].z;
            });
  std::cout << "Sorted points by height: " << result.downsampled_cloud->points.size() << " points." << std::endl;

  std::cout << "Sorting points by curvature..." << std::endl;
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&pca_curvature_values](int i1, int i2) {
              return pca_curvature_values[i1] < pca_curvature_values[i2];
            });
  std::cout << "Sorting by curvature completed." << std::endl;

  // ------------------------------------------------------------------------
  // Select seed points based on height and curvature thresholds.
  // ------------------------------------------------------------------------
  std::queue<int> seedQueue;
  pcl::PointCloud<PointT>::Ptr seed_points(new pcl::PointCloud<PointT>());
  std::cout << "Selecting seed points..." << std::endl;
  for (const auto &idx : sorted_indices) {
    const auto &point = result.downsampled_cloud->points[idx];
    float curvature = pca_curvature_values[idx];
    if (point.z < height_threshold && curvature < curvature_threshold) {
      seed_points->points.push_back(point);
      seedQueue.push(idx);
    }
  }
  std::cout << "Selected " << seed_points->points.size() << " seed points." << std::endl;

  // ------------------------------------------------------------------------
  // Region growing: search neighbors of seed points.
  // ------------------------------------------------------------------------
  pcl::PointCloud<PointT>::Ptr inliers_cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr outliers_cloud(new pcl::PointCloud<PointT>);
  while (!seedQueue.empty()) {
    int seedIdx = seedQueue.front();
    seedQueue.pop();
    const auto &seedPoint = result.downsampled_cloud->points[seedIdx];
    const auto &seedNormal = normals->points[seedIdx];

    std::vector<int> k_indices;
    std::vector<float> k_sqr_distances;
    tree->nearestKSearch(seedPoint, normal_k_search, k_indices, k_sqr_distances);
    for (size_t i = 0; i < k_indices.size(); ++i) {
      int neighborIdx = k_indices[i];
      const auto &neighborPoint = result.downsampled_cloud->points[neighborIdx];
      const auto &neighborNormal = normals->points[neighborIdx];

      // Compute the angle between the normals.
      float dotProduct = seedNormal.normal_x * neighborNormal.normal_x +
                         seedNormal.normal_y * neighborNormal.normal_y +
                         seedNormal.normal_z * neighborNormal.normal_z;
      float angle = std::acos(dotProduct);
      if (angle < smoothness_threshold) {
        inliers_cloud->points.push_back(neighborPoint);
        float neighborCurvature = pca_curvature_values[neighborIdx];
        if (neighborCurvature < curvature_threshold) {
          seedQueue.push(neighborIdx);
        }
      }
    }
  }

  // ------------------------------------------------------------------------
  // Finalize the result: determine outlier points.
  // ------------------------------------------------------------------------
  std::set<int> cluster_indices;
  for (size_t i = 0; i < result.downsampled_cloud->points.size(); ++i) {
    if (cluster_indices.find(static_cast<int>(i)) == cluster_indices.end()) {
      outliers_cloud->points.push_back(result.downsampled_cloud->points[i]);
    }
  }
  inliers_cloud->header = result.downsampled_cloud->header;
  outliers_cloud->header = result.downsampled_cloud->header;
  result.inlier_cloud = inliers_cloud;
  result.outlier_cloud = outliers_cloud;
  result.pcl_method = "IMPROVED REGION GROWING SEGMENTATION WITH SEED POINT QUEUE";

  return result;
}


// void calculateNormalsAndCurvature(const CloudInput<PointT> &input) {
//   // Load the point cloud
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//   if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
//       PCL_ERROR("Couldn't read the file \n");
//       return;
//   }

//   // Create a KD-Tree for the point cloud
//   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//   tree->setInputCloud(cloud);

//   // Create the normal estimation object
//   pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//   ne.setSearchMethod(tree);
//   ne.setKSearch(number_of_neighbours); // Number of neighbors for normal estimation

//   pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//   ne.setInputCloud(cloud);
//   ne.compute(*normals);

//   // Extract normal vectors and curvature
//   Eigen::Vector3f normal;
//   float curvature;

//   std::cout << "Normal Vectors and Curvatures of the first 5 points:" << std::endl;
//   for (size_t i = 0; i < 5; ++i) {
//       // Normal vector: nx, ny, nz
//       normal << normals->points[i].normal_x, normals->points[i].normal_y, normals->points[i].normal_z;
//       // Curvature: Using eigenvalue of covariance matrix, curvature is the smallest eigenvalue
//       curvature = normals->points[i].curvature;

//       std::cout << "Point " << i + 1 << ":" << std::endl;
//       std::cout << "Normal Vector: (" << normal[0] << ", " << normal[1] << ", " << normal[2] << ")" << std::endl;
//       std::cout << "Curvature: " << curvature << std::endl;
//   }

//   // Now implement the PCA for the eigenvalue calculation to get the exact normal and curvature

//   for (size_t i = 0; i < cloud->points.size(); ++i) {
//       // Get the k nearest neighbors using the KD-tree
//       std::vector<int> point_indices(k_neighbors);
//       std::vector<float> point_squared_distances(k_neighbors);

//       if (tree->nearestKSearch(cloud->points[i], k_neighbors, point_indices, point_squared_distances) > 0) {
//           // Build matrix M with the relative coordinates
//           Eigen::MatrixXf M(k_neighbors, 3);
//           for (int j = 0; j < k_neighbors; ++j) {
//               const pcl::PointXYZ& neighbor = cloud->points[point_indices[j]];
//               M(j, 0) = neighbor.x - cloud->points[i].x;
//               M(j, 1) = neighbor.y - cloud->points[i].y;
//               M(j, 2) = neighbor.z - cloud->points[i].z;
//           }

//           // Calculate the covariance matrix
//           Eigen::Matrix3f covariance_matrix = M.transpose() * M;

//           // Perform eigenvalue decomposition
//           Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
//           Eigen::Vector3f eigenvalues = solver.eigenvalues().real();
//           Eigen::Matrix3f eigenvectors = solver.eigenvectors().real();

//           // Get the smallest eigenvalue and its corresponding eigenvector (normal vector)
//           int min_eigenvalue_index;
//           eigenvalues.minCoeff(&min_eigenvalue_index);

//           Eigen::Vector3f normal_vector = eigenvectors.col(min_eigenvalue_index);
//           float curvature_value = eigenvalues(min_eigenvalue_index) / eigenvalues.sum();

//           // Print results
//           std::cout << "Point " << i + 1 << ":" << std::endl;
//           std::cout << "Normal Vector: (" << normal_vector[0] << ", " << normal_vector[1] << ", " << normal_vector[2] << ")" << std::endl;
//           std::cout << "Curvature: " << curvature_value << std::endl;
//       }
//   }
// }

//============================== Check SLZ -r ================================================================================================



inline PCLResult octreeNeighborhoodPCAFilter1(const CloudInput<PointT>& input,
                                               double radius,
                                               float voxelSize,
                                               int k,
                                               float angleThreshold,
                                               int landingZoneNumber,
                                               int maxAttempts) {
    // Seed the random number generator to ensure unique random values each time
    srand(time(0));
    
    std::cout << "Loading point cloud..." << std::endl;
    auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);
    if (!cloud || cloud->points.empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        PCLResult emptyResult;
        return emptyResult;
    }

    // Build the KD-tree using the entire cloud
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    // We'll collect accepted candidate patches and track their point indices.
    std::vector<PCLResult> candidatePatches;
    std::unordered_set<int> acceptedIndices;  // Indices from accepted (flat) patches
    std::unordered_set<int> rejectedIndices;  // Indices from rejected (non-flat) patches
    int attempts = 0;

    while (candidatePatches.size() < static_cast<size_t>(landingZoneNumber) && attempts < maxAttempts) {
        attempts++;
        int randomIndex = rand() % cloud->points.size();
        pcl::PointXYZI searchPoint = (*cloud)[randomIndex];

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        std::cout << "Attempt " << attempts << ": Searching neighbors within radius at ("
                  << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
                  << ") with radius = " << radius << std::endl;

        int neighborsFound = kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        if (neighborsFound > 0) {
            std::cout << "Found " << neighborsFound << " neighbors for this patch." << std::endl;

            // Create a patch cloud from the found neighbors.
            typename pcl::PointCloud<PointT>::Ptr patchCloud(new pcl::PointCloud<PointT>());
            for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
                patchCloud->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
            }

            // Apply PCA to the patch.
            // PrincipleComponentAnalysis is assumed to return a PCLResult,
            // where the outlier_cloud field is non-empty if the patch is not flat.
            PCLResult pcaResult = PrincipleComponentAnalysis(patchCloud, 0.45f, 20.0f, 10);

            if (pcaResult.outlier_cloud->points.empty()) {
                std::cout << "Patch is flat. Saving as candidate patch." << std::endl;
                candidatePatches.push_back(pcaResult);
                // Mark the indices as accepted.
                for (int idx : pointIdxRadiusSearch) {
                    acceptedIndices.insert(idx);
                }
            } else {
                std::cout << "Patch has outliers. Discarding this patch." << std::endl;
                // Optionally, store rejected indices.
                for (int idx : pointIdxRadiusSearch) {
                    rejectedIndices.insert(idx);
                }
            }
        } else {
            std::cout << "No neighbors found within radius." << std::endl;
        }
    }

    // Merge candidate patches into a single inlier cloud.
    PCLResult finalResult;
    finalResult.inlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.outlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.downsampled_cloud = cloud; // keep the original cloud

    for (const auto& patch : candidatePatches) {
        finalResult.inlier_cloud->points.insert(finalResult.inlier_cloud->points.end(),
                                                  patch.inlier_cloud->points.begin(),
                                                  patch.inlier_cloud->points.end());
    }

    // Add remaining points (i.e. those not used in accepted candidate patches) to outlier cloud.
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (acceptedIndices.find(i) == acceptedIndices.end()) {
            finalResult.outlier_cloud->points.push_back(cloud->points[i]);
        }
    }

    std::cout << "Found " << candidatePatches.size() 
              << " candidate landing zones after " << attempts << " attempts." << std::endl;
    std::cout << "Final inlier cloud size (candidate patches): " << finalResult.inlier_cloud->points.size() << " points." << std::endl;
    std::cout << "Final outlier cloud size (remaining points): " << finalResult.outlier_cloud->points.size() << " points." << std::endl;

    return finalResult;
}

//Working perfectly but find patches on the corner and form semi circle
inline PCLResult octreeNeighborhoodPCAFilter2(const CloudInput<PointT>& input,
  double initRadius,
  float voxelSize,
  int k,
  float angleThreshold,
  int landingZoneNumber,
  int maxAttempts) {
  // Seed the random number generator to ensure unique random values each time
  srand(time(0));

  std::cout << "Loading point cloud..." << std::endl;
  auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);
  if (!cloud || cloud->points.empty()) {
  std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
  PCLResult emptyResult;
  return emptyResult;
  }

  // Build the KD-tree using the entire cloud
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(cloud);

  // Containers to collect accepted candidate patches and track indices.
  std::vector<PCLResult> candidatePatches;
  std::unordered_set<int> acceptedIndices;  // Indices from accepted (flat) patches
  std::unordered_set<int> rejectedIndices;  // Indices from rejected (non-flat) patches
  int attempts = 0;
  double radiusIncrement = 0.5;  // adjustable increment in meters

  // Main loop: keep trying until we have enough candidate patches or reach maxAttempts.
  while (candidatePatches.size() < static_cast<size_t>(landingZoneNumber) && attempts < maxAttempts) {
  attempts++;
  int randomIndex = rand() % cloud->points.size();
  PointT searchPoint = (*cloud)[randomIndex];

  double currentRadius = initRadius;
  PCLResult bestFlatPatch;  // to store the last flat patch result
  bool foundFlat = false;   // flag indicating we have at least one flat patch

  std::cout << "Attempt " << attempts << ": Searching neighbors at ("
  << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
  << ") with initial radius = " << currentRadius << std::endl;

  while (true) {
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  int neighborsFound = kdtree.radiusSearch(searchPoint, currentRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

  std::cout << "   Radius = " << currentRadius << " m: Found " << neighborsFound << " neighbors." << std::endl;

  if (neighborsFound <= 0) {
  // If no neighbors are found at current radius, break the loop.
  break;
  }

  // Create a patch cloud from the found neighbors.
  typename pcl::PointCloud<PointT>::Ptr patchCloud(new pcl::PointCloud<PointT>());
  for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
  patchCloud->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
  }

  // Apply PCA to the patch.
  // Here, PrincipleComponentAnalysis is assumed to return a PCLResult.
  // It is assumed that if the patch is flat, pcaResult.outlier_cloud->points is empty.
  PCLResult pcaResult = PrincipleComponentAnalysis(patchCloud, voxelSize, angleThreshold, k);

  if (pcaResult.outlier_cloud->points.empty()) {
  // The patch is flat.
  std::cout << "   Patch is flat at radius " << currentRadius << " m." << std::endl;
  bestFlatPatch = pcaResult;  // store this as the best flat patch so far
  foundFlat = true;
  // Increase the search radius and try to get a larger flat area.
  currentRadius += radiusIncrement;
  continue;  // try with the increased radius
  } else {
  std::cout << "   Patch is not flat at radius " << currentRadius << " m." << std::endl;
  // The current patch is not flat. Break out of the loop.
  break;
  }
  }  // end inner loop

  // If we have found a flat patch before the patch turned non-flat, use it.
  if (foundFlat) {
  std::cout << "Saving candidate patch from previous flat search (radius < " 
  << currentRadius << " m)." << std::endl;
  candidatePatches.push_back(bestFlatPatch);
  // Mark the indices of this candidate patch as accepted.
  // For simplicity, we add all neighbor indices from the best flat patch.
  // (Assuming bestFlatPatch stores the indices or you can derive them from its inlier_cloud.)
  for (const auto& pt : bestFlatPatch.inlier_cloud->points) {
  // In a real implementation, you might need to map the point back to its index.
  // Here we simply note that these points belong to an accepted patch.
  // (This part may be adjusted based on your data structure.)
  // acceptedIndices.insert(idx);
  }
  } else {
  std::cout << "No flat patch found in this attempt." << std::endl;
  }
  }  // end main loop

  // Merge candidate patches into a single inlier cloud.
  PCLResult finalResult;
  finalResult.inlier_cloud = pcl::make_shared<PointCloudT>();
  finalResult.outlier_cloud = pcl::make_shared<PointCloudT>();
  finalResult.downsampled_cloud = cloud; // keep the original cloud

  for (const auto& patch : candidatePatches) {
  finalResult.inlier_cloud->points.insert(finalResult.inlier_cloud->points.end(),
      patch.inlier_cloud->points.begin(),
      patch.inlier_cloud->points.end());
  }

  // Add remaining points (i.e. those not used in accepted candidate patches) to the outlier cloud.
  for (size_t i = 0; i < cloud->points.size(); ++i) {
  if (acceptedIndices.find(i) == acceptedIndices.end()) {
  finalResult.outlier_cloud->points.push_back(cloud->points[i]);
  }
  }

  std::cout << "Found " << candidatePatches.size() 
  << " candidate landing zones after " << maxAttempts << " attempts." << std::endl;
  std::cout << "Final inlier cloud size (candidate patches): " << finalResult.inlier_cloud->points.size() << " points." << std::endl;
  std::cout << "Final outlier cloud size (remaining points): " << finalResult.outlier_cloud->points.size() << " points." << std::endl;

  return finalResult;
}
//================================================================================================================================================

// The following function is your octreeNeighborhood PCAFilter modified to check for edge points.
inline PCLResult octreeNeighborhoodPCAFilter(const CloudInput<PointT>& input,
                                               double initRadius,
                                               float voxelSize,
                                               int k,
                                               float angleThreshold,
                                               int landingZoneNumber,
                                               int maxAttempts) {
    // Seed the random number generator to ensure unique random values each time
    srand(time(0));
    
    std::cout << "Loading point cloud..." << std::endl;
    auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);
    if (!cloud || cloud->points.empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        PCLResult emptyResult;
        return emptyResult;
    }

    // Compute the 2D bounding box in the XY plane for the entire cloud
    double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }
    std::cout << "Cloud bounding box (XY): x=[" << minX << ", " << maxX 
              << "], y=[" << minY << ", " << maxY << "]" << std::endl;
    
    // Downsample the cloud if needed.
    if (performDownsampling) {
        downsamplePointCloudPCL<PointT>(cloud, cloud, voxelSize);
        std::cout << "Downsampled cloud has " << cloud->points.size() << " points." << std::endl;
    } else {
        std::cout << "No downsampling performed." << std::endl;
    }

    // Build the KD-tree using the entire cloud.
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    // Containers for candidate patches.
    std::vector<PCLResult> candidatePatches;
    std::unordered_set<int> acceptedIndices;  // (optional: to track accepted indices)
    int attempts = 0;

    // Adjustable parameters.
    double currentRadius = initRadius;
    double radiusIncrement = 0.5;           // Increase search radius by 0.5 m each time
    double circularityThreshold = 0.8;        // Minimum circularity (using a metric like (4π·area)/(perimeter²)) for a full circle

    while (candidatePatches.size() < static_cast<size_t>(landingZoneNumber) && attempts < maxAttempts) {
        attempts++;
        int randomIndex = rand() % cloud->points.size();
        PointT searchPoint = (*cloud)[randomIndex];

        // Check if the random point is sufficiently away from the edge
        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius)) {
            std::cout << "Attempt " << attempts << ": Random point at (" << searchPoint.x << ", " 
                      << searchPoint.y << ", " << searchPoint.z 
                      << ") is too close to the edge. Skipping." << std::endl;
            continue;
        }

        std::cout << "Attempt " << attempts << ": Searching neighbors at ("
                  << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
                  << ") with initial radius = " << currentRadius << " m" << std::endl;

        bool foundFlat = false;
        PCLResult bestFlatPatch;  // to store the last flat, circular patch
        double bestCircularity = 0.0;

        // Increase search radius iteratively until a non-flat patch is detected.
        while (true) {
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            int neighborsFound = kdtree.radiusSearch(searchPoint, currentRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            
            std::cout << "   Radius = " << currentRadius << " m: Found " << neighborsFound << " neighbors." << std::endl;
            
            if (neighborsFound <= 0) {
                break;  // No neighbors found.
            }
            
            // Create a patch cloud from the found neighbors.
            typename pcl::PointCloud<PointT>::Ptr patchCloud(new pcl::PointCloud<PointT>());
            for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
                patchCloud->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
            }
            
            // Apply PCA to the patch.
            // PrincipleComponentAnalysis is assumed to return a PCLResult,
            // where pcaResult.outlier_cloud is empty if the patch is flat.
            PCLResult pcaResult = PrincipleComponentAnalysis(patchCloud, voxelSize, angleThreshold, k);
            
            if (pcaResult.outlier_cloud->points.empty()) {
                // Patch is flat; now check its circularity.
                // Compute the convex hull of the patch projected onto XY.
                pcl::ConvexHull<PointT> chull;
                chull.setInputCloud(patchCloud);
                chull.setDimension(2);
                typename pcl::PointCloud<PointT>::Ptr hull(new pcl::PointCloud<PointT>());
                chull.reconstruct(*hull);

                // Compute area and perimeter using the hull.
                double area = 0.0, perimeter = 0.0;
                if (hull->points.size() >= 3) {
                    for (size_t i = 0; i < hull->points.size(); ++i) {
                        size_t j = (i + 1) % hull->points.size();
                        double xi = hull->points[i].x, yi = hull->points[i].y;
                        double xj = hull->points[j].x, yj = hull->points[j].y;
                        area += (xi * yj - xj * yi);
                        perimeter += std::hypot(xj - xi, yj - yi);
                    }
                    area = std::abs(area) * 0.5;
                }
                double circularity = (perimeter > 0) ? (4 * M_PI * area) / (perimeter * perimeter) : 0.0;
                std::cout << "   Patch is flat. Circularity = " << circularity << std::endl;
                if (circularity >= circularityThreshold) {
                    std::cout << "   Patch is flat and circular at radius " << currentRadius << " m." << std::endl;
                    bestFlatPatch = pcaResult;
                    bestCircularity = circularity;
                    foundFlat = true;
                } else {
                    std::cout << "   Patch is flat but incomplete (circularity = " << circularity << ")." << std::endl;
                }
                // Increase the search radius to attempt a larger patch.
                currentRadius += radiusIncrement;
                continue;
            } else {
                std::cout << "   Patch is not flat at radius " << currentRadius << " m." << std::endl;
                break;  // Exit the inner loop if patch is non-flat.
            }
        }  // End inner while
        
        if (foundFlat) {
            std::cout << "Saving candidate patch from previous flat search (radius < " 
                      << currentRadius << " m, circularity = " << bestCircularity << ")." << std::endl;
            candidatePatches.push_back(bestFlatPatch);
        } else {
            std::cout << "No suitable full flat patch found in this attempt." << std::endl;
        }
        
        // Reset currentRadius for the next attempt.
        currentRadius = initRadius;
    }  // End main while

    // Merge candidate patches into a single inlier cloud.
    PCLResult finalResult;
    finalResult.inlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.outlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.downsampled_cloud = cloud;  // keep original cloud

    for (const auto& patch : candidatePatches) {
        finalResult.inlier_cloud->points.insert(finalResult.inlier_cloud->points.end(),
                                                  patch.inlier_cloud->points.begin(),
                                                  patch.inlier_cloud->points.end());
    }

    // Add remaining points to outlier cloud.
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        // (If you track indices, you can avoid duplicates)
        finalResult.outlier_cloud->points.push_back(cloud->points[i]);
    }

    std::cout << "Found " << candidatePatches.size() 
              << " candidate landing zones after " << maxAttempts << " attempts." << std::endl;
    std::cout << "Final inlier cloud size (candidate patches): " << finalResult.inlier_cloud->points.size() << " points." << std::endl;
    std::cout << "Final outlier cloud size (remaining points): " << finalResult.outlier_cloud->points.size() << " points." << std::endl;

    return finalResult;
}

//================================================================================================================================================
// The following function is your octreeNeighborhood PCAFilter modified to avoid seed points which are close to edges and seed points which are already in a candidate point found
inline PCLResult octreeNeighborhoodPCAFilter3(const CloudInput<PointT>& input,
                                               double initRadius,
                                               float voxelSize,
                                               int k,
                                               float angleThreshold,
                                               int landingZoneNumber,
                                               int maxAttempts) {
    // Seed the random number generator for unique values each time.
    srand(time(0));
    
    std::cout << "Loading point cloud..." << std::endl;
    auto [cloud, performDownsampling] = loadPCLCloud<PointT>(input);
    if (!cloud || cloud->points.empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        PCLResult emptyResult;
        return emptyResult;
    }

    // Compute 2D bounding box (optional, if you want to avoid edge points)
    double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }
    std::cout << "Cloud bounding box (XY): x=[" << minX << ", " << maxX 
              << "], y=[" << minY << ", " << maxY << "]" << std::endl;
    
    // Downsample if needed.
    if (performDownsampling) {
        downsamplePointCloudPCL<PointT>(cloud, cloud, voxelSize);
        std::cout << "Downsampled cloud has " << cloud->points.size() << " points." << std::endl;
    } else {
        std::cout << "No downsampling performed." << std::endl;
    }

    // Build KD-tree for the cloud.
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    // This set will track indices already included in a candidate flat patch.
    std::unordered_set<int> acceptedIndices;
    
    // Containers for candidate patches.
    std::vector<PCLResult> candidatePatches;
    int attempts = 0;

    // Adjustable parameters.
    double currentRadius = initRadius;
    double radiusIncrement = 0.5;           // Increase search radius by 0.5 m each iteration.
    double circularityThreshold = 0.8;        // Circularity threshold (e.g., (4π·area)/(perimeter²) must be >= 0.8.

    while (candidatePatches.size() < static_cast<size_t>(landingZoneNumber) && attempts < maxAttempts) {
        attempts++;
        int randomIndex = rand() % cloud->points.size();
        
        // Skip this random point if it is already in an accepted candidate patch.
        if (acceptedIndices.find(randomIndex) != acceptedIndices.end()) {
            std::cout << "Attempt " << attempts << ": Random point " << randomIndex 
                      << " is already part of an accepted flat patch. Skipping." << std::endl;
            continue;
        }
        
        PointT searchPoint = (*cloud)[randomIndex];

        // Optionally, check if the seed is too near the edge.
        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius)) {
            std::cout << "Attempt " << attempts << ": Point " << randomIndex << " is too near the edge. Skipping." << std::endl;
            continue;
        }

        std::cout << "Attempt " << attempts << ": Searching neighbors at ("
                  << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
                  << ") with initial radius = " << currentRadius << " m" << std::endl;

        bool foundFlat = false;
        PCLResult bestFlatPatch;  // store last flat patch that is complete.
        std::vector<int> bestPatchIndices;  // store indices from the best flat patch.
        double bestCircularity = 0.0;

        // Increase search radius iteratively until patch becomes non-flat.
        while (true) {
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            int neighborsFound = kdtree.radiusSearch(searchPoint, currentRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            
            std::cout << "   Radius = " << currentRadius << " m: Found " << neighborsFound << " neighbors." << std::endl;
            
            if (neighborsFound <= 0) {
                break;
            }
            
            // Create a patch cloud from the found neighbors.
            typename pcl::PointCloud<PointT>::Ptr patchCloud(new pcl::PointCloud<PointT>());
            for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
                patchCloud->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
            }
            
            // Apply PCA to the patch.
            // PrincipleComponentAnalysis (PCA) is assumed to return a PCLResult.
            // We assume that if the patch is flat, pcaResult.outlier_cloud->points is empty.
            PCLResult pcaResult = PrincipleComponentAnalysis(patchCloud, voxelSize, angleThreshold, k);
            
            if (pcaResult.outlier_cloud->points.empty()) {
                // Patch is flat; now compute circularity.
                pcl::ConvexHull<PointT> chull;
                chull.setInputCloud(patchCloud);
                chull.setDimension(2);
                typename pcl::PointCloud<PointT>::Ptr hull(new pcl::PointCloud<PointT>());
                chull.reconstruct(*hull);
                
                double area = 0.0, perimeter = 0.0;
                if (hull->points.size() >= 3) {
                    for (size_t i = 0; i < hull->points.size(); ++i) {
                        size_t j = (i + 1) % hull->points.size();
                        double xi = hull->points[i].x, yi = hull->points[i].y;
                        double xj = hull->points[j].x, yj = hull->points[j].y;
                        area += (xi * yj - xj * yi);
                        perimeter += std::hypot(xj - xi, yj - yi);
                    }
                    area = std::abs(area) * 0.5;
                }
                double circularity = (perimeter > 0) ? (4 * M_PI * area) / (perimeter * perimeter) : 0.0;
                std::cout << "   Patch is flat. Circularity = " << circularity << std::endl;
                
                if (circularity >= circularityThreshold) {
                    std::cout << "   Patch is flat and circular at radius " << currentRadius << " m." << std::endl;
                    bestFlatPatch = pcaResult;
                    bestPatchIndices = pointIdxRadiusSearch;
                    bestCircularity = circularity;
                    foundFlat = true;
                } else {
                    std::cout << "   Patch is flat but incomplete (circularity = " << circularity << ")." << std::endl;
                }
                // Increase radius and try again.
                currentRadius += radiusIncrement;
                continue;
            } else {
                std::cout << "   Patch is not flat at radius " << currentRadius << " m." << std::endl;
                break;
            }
        }  // End inner while
        
        if (foundFlat) {
            std::cout << "Saving candidate patch from previous flat search (radius < " 
                      << currentRadius << " m, circularity = " << bestCircularity << ")." << std::endl;
            candidatePatches.push_back(bestFlatPatch);
            // Mark all indices from bestPatchIndices as accepted.
            for (int idx : bestPatchIndices) {
                acceptedIndices.insert(idx);
            }
        } else {
            std::cout << "No suitable full flat patch found in this attempt." << std::endl;
        }
        
        // Reset currentRadius for the next attempt.
        currentRadius = initRadius;
    }  // End main while

    // Merge candidate patches into a single inlier cloud.
    PCLResult finalResult;
    finalResult.inlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.outlier_cloud = pcl::make_shared<PointCloudT>();
    finalResult.downsampled_cloud = cloud;  // Keep original cloud

    for (const auto& patch : candidatePatches) {
        finalResult.inlier_cloud->points.insert(finalResult.inlier_cloud->points.end(),
                                                  patch.inlier_cloud->points.begin(),
                                                  patch.inlier_cloud->points.end());
    }

    // Add remaining points to outlier cloud.
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (acceptedIndices.find(i) == acceptedIndices.end()) {
            finalResult.outlier_cloud->points.push_back(cloud->points[i]);
        }
    }

    std::cout << "Found " << candidatePatches.size() 
              << " candidate landing zones after " << maxAttempts << " attempts." << std::endl;
    std::cout << "Final inlier cloud size (candidate patches): " << finalResult.inlier_cloud->points.size() << " points." << std::endl;
    std::cout << "Final outlier cloud size (remaining points): " << finalResult.outlier_cloud->points.size() << " points." << std::endl;

    return finalResult;
}

//================================================================================================================================================
inline PCLResult findSafeLandingZones(const CloudInput<PointT>& flatInliersInput,
  const CloudInput<PointT>& originalCloudInput,
  double landingRadius,
  double removalThreshold = 0.01f,
  double clusterTolerance =0.02){

  PCLResult result;
  // Initialize the struct clouds.
  result.downsampled_cloud.reset(new PointCloudT);
  result.inlier_cloud.reset(new PointCloudT);   // Initially holds safe landing candidates.
  result.outlier_cloud.reset(new PointCloudT);    // Will store obstacles.
  result.plane_coefficients.reset(new pcl::ModelCoefficients); // Populate if available.
  result.pcl_method = "SafeLandingZoneDetection";

  // Load flat inlier cloud.
  auto [flatInliers, performDownsamplingFlat] = loadPCLCloud<PointT>(flatInliersInput);
  if (!flatInliers) {
  std::cerr << "Failed to load flat inlier cloud." << std::endl;
  return result;
  }
  std::cout << "Loaded flat inlier cloud with " << flatInliers->size() << " points." << std::endl;
  int maxClusterSize = flatInliers->size();

  // Load original point cloud.
  auto [originalCloud, performDownsamplingOrig] = loadPCLCloud<PointT>(originalCloudInput);
  if (!originalCloud) {
  std::cerr << "Failed to load original point cloud." << std::endl;
  return result;
  }
  std::cout << "Loaded original cloud with " << originalCloud->size() << " points." << std::endl;
  result.downsampled_cloud = originalCloud; // For visualization.

  // Build a KD-tree for the flat inlier cloud to remove flat points from the original cloud.
  pcl::KdTreeFLANN<PointT> flatKdTree;
  flatKdTree.setInputCloud(flatInliers);

  // Create an obstacle cloud by removing points from originalCloud that lie in the flat area.
  for (const auto& pt : originalCloud->points)
  {
  std::vector<int> flatIndices;
  std::vector<float> flatDistances;
  // Use a small radius to check if the point is part of the flat area.
  if (flatKdTree.radiusSearch(pt, removalThreshold, flatIndices, flatDistances) == 0)
  {
  // Not found in flatInliers; add to obstacle cloud.
  result.outlier_cloud->points.push_back(pt);
  }
  }
  result.outlier_cloud->width = static_cast<uint32_t>(result.outlier_cloud->points.size());
  result.outlier_cloud->height = 1;
  std::cout << "Created obstacle cloud with " << result.outlier_cloud->size() << " points." << std::endl;

  // Build a KD-tree for the obstacle cloud for fast radius queries.
  pcl::KdTreeFLANN<PointT> obstacleKdTree;
  obstacleKdTree.setInputCloud(result.outlier_cloud);

  // For each candidate point in the flat area, check if obstacles exist within the landing radius.
  for (const auto& candidate : flatInliers->points)
  {
  std::vector<int> obstacleIndices;
  std::vector<float> obstacleDistances;
  int found = obstacleKdTree.radiusSearch(candidate, landingRadius, obstacleIndices, obstacleDistances);
  // If no obstacles are found within landingRadius, mark this candidate as a safe landing zone.
  if (found == 0)
  {
  result.inlier_cloud->points.push_back(candidate);
  }
  }
  result.inlier_cloud->width = static_cast<uint32_t>(result.inlier_cloud->points.size());
  result.inlier_cloud->height = 1;
  std::cout << "Found " << result.inlier_cloud->size() << " safe landing zone candidate points." << std::endl;

  // ---------------------------------------------------------------
  // Cluster the safe landing candidates and filter clusters by area.
  // ---------------------------------------------------------------

  // Create a KD-tree for clustering.
  pcl::search::KdTree<PointT>::Ptr clusterTree(new pcl::search::KdTree<PointT>);
  clusterTree->setInputCloud(result.inlier_cloud);

  // Euclidean clustering.
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<PointT> ec;
  // Parameters for clustering:
  //  // clusterTolerance Maximum distance between points in a cluster (meters).
  int minClusterSize = 80;       // Minimum number of points to consider a valid cluster.
  ec.setClusterTolerance(clusterTolerance);
  ec.setMinClusterSize(minClusterSize);
  ec.setMaxClusterSize(maxClusterSize);
  ec.setSearchMethod(clusterTree);
  ec.setInputCloud(result.inlier_cloud);
  ec.extract(clusters);
  std::cout << "Identified " << clusters.size() << " candidate landing clusters." << std::endl;

  // Calculate the required landing area (area of a circle with radius = landingRadius).
  double requiredArea = M_PI * landingRadius * landingRadius;

  // Create a new point cloud to store valid landing zones.
  PointCloudT::Ptr validLandingZones(new PointCloudT);
  for (const auto& cluster : clusters)
  {
  PointCloudT::Ptr clusterCloud(new PointCloudT);
  for (int idx : cluster.indices)
  {
  clusterCloud->points.push_back(result.inlier_cloud->points[idx]);
  }
  clusterCloud->width = static_cast<uint32_t>(clusterCloud->points.size());
  clusterCloud->height = 1;

  // Estimate the cluster area using a rough approximation.
  // Each point is assumed to represent an area of (clusterTolerance)^2.
  double clusterArea = clusterCloud->points.size() * clusterTolerance * clusterTolerance;
  if (clusterArea >= requiredArea)
  {
  std::cout << "Valid landing cluster found with estimated area: " << clusterArea << " m²." << std::endl;
  *validLandingZones += *clusterCloud;
  }
  }
  std::cout << "Final valid landing zones contain " << validLandingZones->points.size() << " points." << std::endl;

  // Update the inlier cloud with valid landing zones.
  result.inlier_cloud = validLandingZones;

  return result;
}

PCLResult findSafeLandingZonesConvexHull(const CloudInput<PointT>& flatInliersInput,
  const CloudInput<PointT>& originalCloudInput,
  double landingRadius,
  double removalThreshold = 0.01f,
  double clusterTolerance = 0.02){
PCLResult result;
// Initialize the struct clouds.
result.downsampled_cloud.reset(new PointCloudT);
result.inlier_cloud.reset(new PointCloudT);   // Initially holds safe landing candidates.
result.outlier_cloud.reset(new PointCloudT);    // Will store obstacles.
result.plane_coefficients.reset(new pcl::ModelCoefficients); // Populate if available.
result.pcl_method = "SafeLandingZoneDetection";

// Load flat inlier cloud.
auto [flatInliers, performDownsamplingFlat] = loadPCLCloud<PointT>(flatInliersInput);
if (!flatInliers) {
std::cerr << "Failed to load flat inlier cloud." << std::endl;
return result;
}
std::cout << "Loaded flat inlier cloud with " << flatInliers->size() << " points." << std::endl;
int maxClusterSize = flatInliers->size();

// Load original point cloud.
auto [originalCloud, performDownsamplingOrig] = loadPCLCloud<PointT>(originalCloudInput);
if (!originalCloud) {
std::cerr << "Failed to load original point cloud." << std::endl;
return result;
}
std::cout << "Loaded original cloud with " << originalCloud->size() << " points." << std::endl;
result.downsampled_cloud = originalCloud; // For visualization.

// Build a KD-tree for the flat inlier cloud to remove flat points from the original cloud.
pcl::KdTreeFLANN<PointT> flatKdTree;
flatKdTree.setInputCloud(flatInliers);

// Create an obstacle cloud by removing points from originalCloud that lie in the flat area.
for (const auto& pt : originalCloud->points)
{
std::vector<int> flatIndices;
std::vector<float> flatDistances;
// Use a small radius to check if the point is part of the flat area.
if (flatKdTree.radiusSearch(pt, removalThreshold, flatIndices, flatDistances) == 0)
{
// Not found in flatInliers; add to obstacle cloud.
result.outlier_cloud->points.push_back(pt);
}
}
result.outlier_cloud->width = static_cast<uint32_t>(result.outlier_cloud->points.size());
result.outlier_cloud->height = 1;
std::cout << "Created obstacle cloud with " << result.outlier_cloud->size() << " points." << std::endl;

// Build a KD-tree for the obstacle cloud for fast radius queries.
pcl::KdTreeFLANN<PointT> obstacleKdTree;
obstacleKdTree.setInputCloud(result.outlier_cloud);

// For each candidate point in the flat area, check if obstacles exist within the landing radius.
for (const auto& candidate : flatInliers->points)
{
std::vector<int> obstacleIndices;
std::vector<float> obstacleDistances;
int found = obstacleKdTree.radiusSearch(candidate, landingRadius, obstacleIndices, obstacleDistances);
// If no obstacles are found within landingRadius, mark this candidate as a safe landing zone.
if (found == 0)
{
result.inlier_cloud->points.push_back(candidate);
}
}
result.inlier_cloud->width = static_cast<uint32_t>(result.inlier_cloud->points.size());
result.inlier_cloud->height = 1;
std::cout << "Found " << result.inlier_cloud->size() << " safe landing zone candidate points." << std::endl;

// ---------------------------------------------------------------
// Cluster the safe landing candidates and filter clusters by area.
// ---------------------------------------------------------------

// Create a KD-tree for clustering.
pcl::search::KdTree<PointT>::Ptr clusterTree(new pcl::search::KdTree<PointT>);
clusterTree->setInputCloud(result.inlier_cloud);

// Euclidean clustering.
std::vector<pcl::PointIndices> clusters;
pcl::EuclideanClusterExtraction<PointT> ec;
ec.setClusterTolerance(clusterTolerance); // Maximum distance between points (meters)
int minClusterSize = 20;                  // Minimum number of points to consider a valid cluster.
ec.setMinClusterSize(minClusterSize);
ec.setMaxClusterSize(maxClusterSize);
ec.setSearchMethod(clusterTree);
ec.setInputCloud(result.inlier_cloud);
ec.extract(clusters);
std::cout << "Identified " << clusters.size() << " candidate landing clusters." << std::endl;

// Calculate the required landing area (area of a circle with radius = landingRadius).
double requiredArea = M_PI * landingRadius * landingRadius;

// Create a new point cloud to store valid landing zones.
PointCloudT::Ptr validLandingZones(new PointCloudT);
for (const auto& cluster : clusters)
{
// Create a cloud for the current cluster.
PointCloudT::Ptr clusterCloud(new PointCloudT);
for (int idx : cluster.indices)
{
clusterCloud->points.push_back(result.inlier_cloud->points[idx]);
}
clusterCloud->width = static_cast<uint32_t>(clusterCloud->points.size());
clusterCloud->height = 1;

// Project the cluster onto the XY plane by creating a new cloud of pcl::PointXYZ.
pcl::PointCloud<pcl::PointXYZ>::Ptr projectedCloud(new pcl::PointCloud<pcl::PointXYZ>);
for (const auto& pt : clusterCloud->points)
{
pcl::PointXYZ proj_pt;
proj_pt.x = pt.x;
proj_pt.y = pt.y;
proj_pt.z = 0.0; // projection onto XY plane
projectedCloud->points.push_back(proj_pt);
}
projectedCloud->width = static_cast<uint32_t>(projectedCloud->points.size());
projectedCloud->height = 1;

// Compute the convex hull of the projected points.
pcl::ConvexHull<pcl::PointXYZ> hull;
pcl::PointCloud<pcl::PointXYZ>::Ptr hullPoints(new pcl::PointCloud<pcl::PointXYZ>);
hull.setInputCloud(projectedCloud);
hull.reconstruct(*hullPoints);

// Use getTotalArea() to compute the area of the convex hull.
double hullArea = hull.getTotalArea();
std::cout << "Cluster convex hull area: " << hullArea << " m²." << std::endl;

if (hullArea >= requiredArea)
{
std::cout << "Valid landing cluster found with area: " << hullArea << " m²." << std::endl;
*validLandingZones += *clusterCloud;
}
}
std::cout << "Final valid landing zones contain " << validLandingZones->points.size() << " points." << std::endl;

// Update the inlier cloud with valid landing zones.
result.inlier_cloud = validLandingZones;

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


