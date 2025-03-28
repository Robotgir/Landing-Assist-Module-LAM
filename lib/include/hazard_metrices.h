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
auto cloud = loadPCLCloud<PointT>(input);

result.downsampled_cloud = cloud;


// Compute normals in parallel using NormalEstimationOMP.
pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
ne.setInputCloud(result.downsampled_cloud);
ne.setKSearch(k);

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
ne.compute(*normals);


// Compute PCA on the entire cloud.
pcl::PCA<PointT> pca;
pca.setInputCloud(result.downsampled_cloud);
Eigen::Matrix3f eigenvectors = pca.getEigenVectors(); // Eigenvectors sorted by descending eigenvalues.
Eigen::Vector4f mean = pca.getMean();

// Use the eigenvector with the smallest eigenvalue (typically the 3rd column) as the global plane normal.
Eigen::Vector3f global_normal = eigenvectors.col(2);

// Compute and output the global slope (angle between global_normal and the vertical (0,0,1)).
float global_dot = std::fabs(global_normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
float global_slope = std::acos(global_dot) * 180.0f / static_cast<float>(M_PI);
std::cout << "Global PCA plane slope: " << global_slope << " degrees" << std::endl;

// Compute plane coefficients: Ax + By + Cz + D = 0.
float A = global_normal(0);
float B = global_normal(1);
float C = global_normal(2);
float D = -(A * mean(0) + B * mean(1) + C * mean(2));
result.plane_coefficients = std::make_shared<pcl::ModelCoefficients>();
result.plane_coefficients->values.push_back(A);
result.plane_coefficients->values.push_back(B);
result.plane_coefficients->values.push_back(C);
result.plane_coefficients->values.push_back(D);
std::cout << "Plane coefficients (A, B, C, D): " << A << ", " << B << ", " << C << ", " << D << std::endl;

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


//====================================================================================================================================================
inline PCLResult PrincipleComponentAnalysis1(const CloudInput<PointT>& input,
  float voxelSize = 0.45f,
  float angleThreshold = 20.0f,
  int k = 10)
{
PCLResult result;
result.pcl_method = "Principal Component Analysis (using PCA global plane as reference)";
result.inlier_cloud = pcl::make_shared<PointCloudT>();
result.outlier_cloud = pcl::make_shared<PointCloudT>();
result.downsampled_cloud = pcl::make_shared<PointCloudT>();

// Load the cloud.
auto cloud = loadPCLCloud<PointT>(input);
result.downsampled_cloud = cloud;

// Compute normals in parallel using NormalEstimationOMP.
pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
ne.setInputCloud(result.downsampled_cloud);
ne.setKSearch(k);
pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
ne.compute(*normals);


// Compute PCA on the entire cloud.
pcl::PCA<PointT> pca;
pca.setInputCloud(result.downsampled_cloud);
Eigen::Matrix3f eigenvectors = pca.getEigenVectors(); // Eigenvectors sorted by descending eigenvalues.
Eigen::Vector4f mean = pca.getMean();

// Use the eigenvector with the smallest eigenvalue (typically the 3rd column) as the global plane normal.
Eigen::Vector3f global_normal = eigenvectors.col(2);

// Compute and output the global slope (angle between global_normal and the vertical (0,0,1)).
float global_dot = std::fabs(global_normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
float global_slope = std::acos(global_dot) * 180.0f / static_cast<float>(M_PI);
std::cout << "Global PCA plane slope: " << global_slope << " degrees" << std::endl;

// Compute plane coefficients: Ax + By + Cz + D = 0.
float A = global_normal(0);
float B = global_normal(1);
float C = global_normal(2);
float D = -(A * mean(0) + B * mean(1) + C * mean(2));
result.plane_coefficients = std::make_shared<pcl::ModelCoefficients>();
result.plane_coefficients->values.push_back(A);
result.plane_coefficients->values.push_back(B);
result.plane_coefficients->values.push_back(C);
result.plane_coefficients->values.push_back(D);
std::cout << "Plane coefficients (A, B, C, D): " << A << ", " << B << ", " << C << ", " << D << std::endl;

// Now, classify each point based on how its normal compares to the global PCA normal.
// (This ensures that the slope used for classification comes from the PCA result.)
for (size_t i = 0; i < normals->points.size(); i++) {
Eigen::Vector3f point_normal(normals->points[i].normal_x,
normals->points[i].normal_y,
normals->points[i].normal_z);
// Skip invalid normals.
if (std::isnan(point_normal.norm()) || point_normal.norm() == 0) {
result.outlier_cloud->push_back(result.downsampled_cloud->points[i]);
continue;
}
// Compute the angle between the point's normal and the global plane normal.
float dot_product = std::fabs(point_normal.dot(global_normal));
float angle_diff = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);

if (angle_diff <= angleThreshold)
result.inlier_cloud->push_back(result.downsampled_cloud->points[i]);
else
result.outlier_cloud->push_back(result.downsampled_cloud->points[i]);
}

std::cout << "Inliers (angle diff ≤ " << angleThreshold << "°): " << result.inlier_cloud->size() << std::endl;
std::cout << "Outliers (angle diff > " << angleThreshold << "°): " << result.outlier_cloud->size() << std::endl;

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
    auto cloud= loadOpen3DCloud(input);

    // Downsample the point cloud.
    result.downsampled_cloud = cloud;
  
    // Ensure the downsampled point cloud has color information.
    if (!result.downsampled_cloud->HasColors()) {
        result.downsampled_cloud->colors_.resize(result.downsampled_cloud->points_.size(), Eigen::Vector3d(1, 1, 1));
    }

    // Perform RANSAC plane segmentation.
    // 'SegmentPlane' returns a pair: the plane model and the vector of inlier indices.
    double probability = 0.9999;
    Eigen::Vector4d plane_coefficients;
    auto [plane, indices] = result.downsampled_cloud->SegmentPlane(distance_threshold, ransac_n, num_iterations, probability);
    plane_coefficients = plane;
    result.plane_coefficients = plane_coefficients;

    std::cout << "Plane model: " << plane_coefficients.transpose() << std::endl;
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

auto cloud= loadPCLCloud<PointT>(input);
result.downsampled_cloud = cloud;


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

  auto cloud = loadPCLCloud<PointT>(input);
  result.downsampled_cloud = cloud;
 

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
    // pcd = loadOpen3DCloud(input);
    
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

  auto cloud = loadPCLCloud<PointT>(input);
  result.downsampled_cloud = cloud;
 

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

auto cloud = loadPCLCloud<PointT>(input);
result.downsampled_cloud = cloud;

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
  auto cloud= loadPCLCloud<PointT>(input);
  result.downsampled_cloud = cloud;

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
  auto cloud= loadPCLCloud<PointT>(input);
  result.downsampled_cloud = cloud;

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
  auto flatInliers= loadPCLCloud<PointT>(flatInliersInput);
  if (!flatInliers) {
  std::cerr << "Failed to load flat inlier cloud." << std::endl;
  return result;
  }
  std::cout << "Loaded flat inlier cloud with " << flatInliers->size() << " points." << std::endl;
  int maxClusterSize = flatInliers->size();

  // Load original point cloud.
  auto originalCloud = loadPCLCloud<PointT>(originalCloudInput);
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
auto flatInliers = loadPCLCloud<PointT>(flatInliersInput);
if (!flatInliers) {
std::cerr << "Failed to load flat inlier cloud." << std::endl;
return result;
}
std::cout << "Loaded flat inlier cloud with " << flatInliers->size() << " points." << std::endl;
int maxClusterSize = flatInliers->size();

// Load original point cloud.
auto originalCloud = loadPCLCloud<PointT>(originalCloudInput);
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

inline double calculateRoughnessPCL(PCLResult& result)
{
  // Check if the plane coefficients are valid
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
  
  // Calculate the plane normal's magnitude for normalization
  double norm = std::sqrt(a * a + b * b + c * c);
  
  // Variable to accumulate the squared distance of each point from the plane
  double sum_squared = 0.0;
  size_t N = result.inlier_cloud->points.size();
  
  // Loop over each point in the result.er cloud to calculate the roughness
  for (const auto &pt : result.inlier_cloud->points)
  {
    // Calculate the distance from the point to the plane
    double distance = std::abs(a * pt.x + b * pt.y + c * pt.z + d) / norm;
    sum_squared += distance * distance;
  }
  
  // Return the square root of the average squared distance (roughness)
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
    double a = result.plane_coefficients[0];
    double b = result.plane_coefficients[1];
    double c = result.plane_coefficients[2];
    double d = result.plane_coefficients[3];
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
inline double calculateReliefPCL(PCLResult& result)
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
inline double calculateDataConfidencePCL(PCLResult& result)
{
    if (!result.inlier_cloud || result.inlier_cloud->points.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    size_t N = result.inlier_cloud->points.size();

    // Compute the convex hull of the inlier cloud projected onto a plane.
    pcl::ConvexHull<pcl::PointXYZI> chull;
    chull.setInputCloud(result.inlier_cloud);
    chull.setDimension(2);

    pcl::PointCloud<pcl::PointXYZI>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<pcl::Vertices> polygons;
    chull.reconstruct(*hull_points, polygons);

    if (polygons.empty() || hull_points->points.empty()) {
        std::cerr << "Error: Convex hull could not be computed." << std::endl;
        return -1.0;
    }

    // Compute the area of the first polygon using the shoelace formula.
    double area = 0.0;
    const std::vector<int>& indices = polygons[0].vertices;
    size_t n = indices.size();
    if (n < 3) {
        std::cerr << "Error: Convex hull does not have enough points to form an area." << std::endl;
        return -1.0;
    }

    for (size_t i = 0; i < n; i++) {
        const auto& p1 = hull_points->points[indices[i]];
        const auto& p2 = hull_points->points[indices[(i + 1) % n]];
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

enum MetricType {
  DATA_CONFIDENCE = 1,  // 0001
  RELIEF = 2,           // 0010
  ROUGHNESS = 4,        // 0100
  ALL = 7               // 0111 (all metrics)
};

inline SLZDCandidatePoints rankCandidatePatchFromPCLResult(PCLResult& result, const std::string& metrics = "ALL") {
  // Create an SLZDCandidatePoints object for the result
  SLZDCandidatePoints candidate;

  // Clear previous metrics
  candidate.dataConfidences.clear();
  candidate.reliefs.clear();
  candidate.roughnesses.clear();
  candidate.score.clear();

  // Check if the result contains a valid inlier cloud
  if (result.inlier_cloud && !result.inlier_cloud->points.empty()) {
      PCLResult surfResult;
      surfResult.inlier_cloud = result.inlier_cloud;
      surfResult.plane_coefficients = result.plane_coefficients;

      // Calculate all metrics if "ALL" is selected
      if (metrics == "ALL") {
          double dataConfidence = calculateDataConfidencePCL(surfResult);
          candidate.dataConfidences.push_back(dataConfidence);

          double relief = calculateReliefPCL(surfResult);
          candidate.reliefs.push_back(relief);

          double roughness = calculateRoughnessPCL(surfResult);
          candidate.roughnesses.push_back(roughness);
      }
      // Calculate data confidence only if specified
      else if (metrics == "DATA_CONFIDENCE") {
          double dataConfidence = calculateDataConfidencePCL(surfResult);
          candidate.dataConfidences.push_back(dataConfidence);
      }
      // Calculate relief only if specified
      else if (metrics == "RELIEF") {
          double relief = calculateReliefPCL(surfResult);
          candidate.reliefs.push_back(relief);
      }
      // Calculate roughness only if specified
      else if (metrics == "ROUGHNESS") {
          double roughness = calculateRoughnessPCL(surfResult);
          candidate.roughnesses.push_back(roughness);
      }
      else {
          std::cerr << "[rankCandidatePatchFromPCLResult] Error: Invalid metric specified." << std::endl;
      }

      // Compute the surface score if any of the metrics are calculated
      double surfaceScore = 0.0;
      if (metrics == "ALL" || metrics == "DATA_CONFIDENCE") {
          surfaceScore += candidate.dataConfidences.back();
      }
      if (metrics == "ALL" || metrics == "RELIEF") {
          surfaceScore -= candidate.reliefs.back();
      }
      if (metrics == "ALL" || metrics == "ROUGHNESS") {
          surfaceScore -= candidate.roughnesses.back();
      }

      candidate.score.push_back(surfaceScore);

      // Print details of the candidate patch
      std::cout << "Candidate Patch Details:" << std::endl;
      if (metrics == "ALL" || metrics == "DATA_CONFIDENCE") {
          std::cout << "  Data Confidence: " << candidate.dataConfidences.back() << std::endl;
      }
      if (metrics == "ALL" || metrics == "RELIEF") {
          std::cout << "  Relief: " << candidate.reliefs.back() << std::endl;
      }
      if (metrics == "ALL" || metrics == "ROUGHNESS") {
          std::cout << "  Roughness: " << candidate.roughnesses.back() << std::endl;
      }
  } else {
      std::cerr << "[rankCandidatePatchFromPCLResult] Error: Inlier cloud is empty." << std::endl;
  }

  // Compute the final candidate score as the average (if more than one patch is in the result)
  double total = 0.0;
  for (double s : candidate.score) {
      total += s;
  }
  double averageScore = candidate.score.empty() ? 0.0 : total / candidate.score.size();

  // Set the final average score
  candidate.score.clear();
  candidate.score.push_back(averageScore);

  // Print the final score
  std::cout << "  Final Average Score: " << averageScore << std::endl;

  // Return the candidate object
  return candidate;
}




#endif 


