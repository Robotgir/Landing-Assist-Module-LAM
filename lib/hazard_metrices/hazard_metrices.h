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

#include <../common/common.h>
#include <variant>

#include <pcl/features/normal_3d_omp.h>

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

template <typename PointT>
using CloudInput = std::variant<std::string, typename pcl::PointCloud<PointT>::Ptr>;

using Open3DCloudInput = std::variant<std::string, std::shared_ptr<open3d::geometry::PointCloud>>;
 




//======================================PCA (Principle Component Analysis)(PCL)==========================================================================================================


// inline PCLResult PrincipleComponentAnalysis(const CloudInput<PointT>& input,
//                                               float voxelSize = 0.45f,
//                                               float angleThreshold = 20.0f,
//                                               int k = 10) {
//   PCLResult result;
//   result.pcl_method = "Principal Component Analysis";
//   result.inlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
//   result.outlier_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();
//   result.downsampled_cloud = pcl::make_shared<typename pcl::PointCloud<PointT>>();

//   // Remove previous declarations and use structured binding with new variable names.
//   auto [loadedCloud, doDownsample] = loadPCLCloud<PointT>(input);

//   // Downsample if the input was a file path.
//   if (doDownsample) {
//     downsamplePointCloudPCL<PointT>(loadedCloud, result.downsampled_cloud, voxelSize);
//     std::cout << "Downsampled cloud has " << result.downsampled_cloud->points.size() << " points." << std::endl;
//   } else {
//     result.downsampled_cloud = loadedCloud;
//   }

//   pcl::KdTreeFLANN<PointT> tree;
//   tree.setInputCloud(result.downsampled_cloud);

//   // Compute normals.
//   pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//   normals->points.resize(result.downsampled_cloud->points.size());
//   normals->width = result.downsampled_cloud->width;
//   normals->height = result.downsampled_cloud->height;
//   normals->is_dense = result.downsampled_cloud->is_dense;

//   std::vector<int> neighbor_indices(k);
//   std::vector<float> sqr_distances(k);

//   for (size_t i = 0; i < result.downsampled_cloud->points.size(); i++) {
//     if (tree.nearestKSearch(result.downsampled_cloud->points[i], k, neighbor_indices, sqr_distances) > 0) {
//       Eigen::Vector4f local_centroid;
//       pcl::compute3DCentroid(*result.downsampled_cloud, neighbor_indices, local_centroid);
//       Eigen::Matrix3f covariance;
//       pcl::computeCovarianceMatrixNormalized(*result.downsampled_cloud, neighbor_indices, local_centroid, covariance);
//       Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
//       Eigen::Vector3f normal = solver.eigenvectors().col(0);

//       normals->points[i].normal_x = normal.x();
//       normals->points[i].normal_y = normal.y();
//       normals->points[i].normal_z = normal.z();

//       float dot_product = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
//       float slope = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);

//       if (slope <= angleThreshold)
//         result.inlier_cloud->push_back(result.downsampled_cloud->points[i]);
//       else
//         result.outlier_cloud->push_back(result.downsampled_cloud->points[i]);
//     } else {
//       normals->points[i].normal_x = std::numeric_limits<float>::quiet_NaN();
//       normals->points[i].normal_y = std::numeric_limits<float>::quiet_NaN();
//       normals->points[i].normal_z = std::numeric_limits<float>::quiet_NaN();
//     }
//   }
//   std::cout << "Inliers (slope ≤ " << angleThreshold << "°): " << result.inlier_cloud->size() << std::endl;
//   std::cout << "Outliers (slope > " << angleThreshold << "°): " << result.outlier_cloud->size() << std::endl;

//   return result;
// }

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
