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
//======================================PCA (Principle Component Analysis)(PCL)==========================================================================================================

// template <typename Point_pcl>
// inline processingResult PrincipleComponentAnalysis(const PointCloudPcl& input_cloud,
//   float angleThreshold = 20.0f,
//   int k = 10)
// {
// processingResult result;
// result.processing_method = "Principal Component Analysis (using NormalEstimationOMP)";
// result.hazardMetric_type = "Flatness";

// auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
// auto result_outlier_cloud = std::get<PointCloudPcl>(result.outlier_cloud);

// // Compute normals in parallel using NormalEstimationOMP.
// pcl::NormalEstimationOMP<PointPcl, pcl::Normal> ne;
// ne.setInputCloud(input_cloud);
// ne.setKSearch(k);

// pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
// ne.compute(*normals);


// // Compute PCA on the entire cloud.
// pcl::PCA<PointPcl> pca;
// pca.setInputCloud(input_cloud);
// Eigen::Matrix3f eigenvectors = pca.getEigenVectors(); // Eigenvectors sorted by descending eigenvalues.
// Eigen::Vector4f mean = pca.getMean();

// // Use the eigenvector with the smallest eigenvalue (typically the 3rd column) as the global plane normal.
// Eigen::Vector3f global_normal = eigenvectors.col(2);

// // Compute and output the global slope (angle between global_normal and the vertical (0,0,1)).
// float global_dot = std::fabs(global_normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
// float global_slope = std::acos(global_dot) * 180.0f / static_cast<float>(M_PI);
// std::cout << "Global PCA plane slope: " << global_slope << " degrees" << std::endl;

// // Compute plane coefficients: Ax + By + Cz + D = 0.
// float A = global_normal(0);
// float B = global_normal(1);
// float C = global_normal(2);
// float D = -(A * mean(0) + B * mean(1) + C * mean(2));
// std::cout << "[INFO] here_3: " <<std::endl;
// auto result_plane_coefficients =  std::make_shared<pcl::ModelCoefficients>();
// std::cout << "[INFO] here__4: " <<std::endl;
// result_plane_coefficients->values.push_back(A);
// std::cout << "[INFO] here_5: " <<std::endl;
// result_plane_coefficients->values.push_back(B);
// result_plane_coefficients->values.push_back(C);
// result_plane_coefficients->values.push_back(D);
// std::cout << "Plane coefficients (A, B, C, D): " << A << ", " << B << ", " << C << ", " << D << std::endl;

// // Classify points based on the computed normals and the angle threshold.
// for (size_t i = 0; i < normals->points.size(); i++) {
// Eigen::Vector3f normal(normals->points[i].normal_x,
// normals->points[i].normal_y,
// normals->points[i].normal_z);
// // Check for invalid normal values.
// if (std::isnan(normal.norm()) || normal.norm() == 0) {
// result_outlier_cloud->push_back(input_cloud->points[i]);
// continue;
// }
// float dot_product = std::fabs(normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
// float slope = std::acos(dot_product) * 180.0f / static_cast<float>(M_PI);
// if (slope <= angleThreshold)
// result_inlier_cloud->push_back(input_cloud->points[i]);
// else
// result_outlier_cloud->push_back(input_cloud->points[i]);
// }
// std::cout << "Inliers (slope ≤ " << angleThreshold << "°): " << result_inlier_cloud->size() << std::endl;
// std::cout << "Outliers (slope > " << angleThreshold << "°): " << result_outlier_cloud->size() << std::endl;

// return result;
// }
inline processingResult PrincipleComponentAnalysis(
  const PointCloudPcl& input_cloud,
  float angleThreshold = 20.0f,
  int k = 10)
{
  processingResult result;
  result.processing_method = "Principal Component Analysis (using NormalEstimationOMP)";
  result.hazardMetric_type = "Flatness";

  // Initialize outputs
  auto result_inlier_cloud = std::make_shared<pcl::PointCloud<PointPcl>>();
  auto result_outlier_cloud = std::make_shared<pcl::PointCloud<PointPcl>>();
  auto result_plane_coefficients = std::make_shared<pcl::ModelCoefficients>();
  result.plane_coefficients = result_plane_coefficients;

  // Validate input
  if (!input_cloud || input_cloud->empty() || input_cloud->size() < static_cast<size_t>(k))
  {
      std::cerr << "PCA: Invalid or insufficient input cloud (size=" << (input_cloud ? input_cloud->size() : 0) << ", required=" << k << ")\n";
      result.inlier_cloud = result_inlier_cloud;
      result.outlier_cloud = result_outlier_cloud;
      return result;
  }

  // Compute normals
  pcl::NormalEstimationOMP<PointPcl, pcl::Normal> ne;
  ne.setInputCloud(input_cloud);
  ne.setKSearch(k);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals);

  if (normals->size() != input_cloud->size())
  {
      std::cerr << "PCA: Normal computation failed (normals=" << normals->size() << ", expected=" << input_cloud->size() << ")\n";
      result.inlier_cloud = result_inlier_cloud;
      result.outlier_cloud = result_outlier_cloud;
      return result;
  }

  // Compute PCA
  pcl::PCA<PointPcl> pca;
  pca.setInputCloud(input_cloud);
  Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
  Eigen::Vector4f mean = pca.getMean();

  // Use smallest eigenvalue's eigenvector as plane normal
  Eigen::Vector3f global_normal = eigenvectors.col(2);

  // Compute slope
  float global_dot = std::fabs(global_normal.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
  float global_slope = std::acos(global_dot) * 180.0f / static_cast<float>(M_PI);
  std::cout << "Global PCA plane slope: " << global_slope << " degrees\n";

  // Compute plane coefficients: Ax + By + Cz + D = 0
  float A = global_normal(0);
  float B = global_normal(1);
  float C = global_normal(2);
  float D = -(A * mean(0) + B * mean(1) + C * mean(2));
  std::cout << "[INFO] here_3: " << std::endl;

  result_plane_coefficients->values.push_back(A);
  std::cout << "[INFO] here__4: " << std::endl;
  result_plane_coefficients->values.push_back(B);
  result_plane_coefficients->values.push_back(C);
  result_plane_coefficients->values.push_back(D);
  std::cout << "[INFO] here_5: " << std::endl;
  std::cout << "Plane coefficients (A, B, C, D): " << A << ", " << B << ", " << C << ", " << D << "\n";

  // Filter points
  result_inlier_cloud->reserve(input_cloud->size());
  result_outlier_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); ++i)
  {
      const auto& point = input_cloud->points[i];
      const auto& normal = normals->points[i];
      float dot = std::fabs(Eigen::Vector3f(normal.normal_x, normal.normal_y, normal.normal_z)
                                .dot(global_normal));
      float angle = std::acos(dot) * 180.0f / static_cast<float>(M_PI);
      if (angle <= angleThreshold)
      {
          result_inlier_cloud->push_back(point);
      }
      else
      {
          result_outlier_cloud->push_back(point);
      }
  }

  // Set cloud properties
  result_inlier_cloud->width = result_inlier_cloud->size();
  result_inlier_cloud->height = 1;
  result_inlier_cloud->is_dense = true;
  result_outlier_cloud->width = result_outlier_cloud->size();
  result_outlier_cloud->height = 1;
  result_outlier_cloud->is_dense = true;

  result.inlier_cloud = result_inlier_cloud;
  result.outlier_cloud = result_outlier_cloud;

  // Log cloud sizes for debugging
  std::cout << "PCA: Inliers (slope ≤ " << angleThreshold << "°): " << result_inlier_cloud->size() << "\n";
  std::cout << "PCA: Outliers (slope > " << angleThreshold << "°): " << result_outlier_cloud->size() << "\n";

  return result;
}
//===============================Calculate Roughness====================================================================================

inline double calculateRoughness(processingResult& result)
{auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
auto result_plane_coefficients = std::get<pcl::ModelCoefficients::Ptr>(result.plane_coefficients);
  // Check if the plane coefficients are valid
  if (result_plane_coefficients->values.size() < 4 || result_inlier_cloud->points.empty())
  {
    std::cerr << "Invalid plane coefficients or empty inlier cloud. Cannot compute roughness." << std::endl;
    return -1.0;
  }
  
  // Extract plane parameters (ax + by + cz + d = 0).
  double a = result_plane_coefficients->values[0];
  double b = result_plane_coefficients->values[1];
  double c = result_plane_coefficients->values[2];
  double d = result_plane_coefficients->values[3];
  
  // Calculate the plane normal's magnitude for normalization
  double norm = std::sqrt(a * a + b * b + c * c);
  
  // Variable to accumulate the squared distance of each point from the plane
  double sum_squared = 0.0;
  size_t N = result_inlier_cloud->points.size();
  
  // Loop over each point in the result.er cloud to calculate the roughness
  for (const auto &pt : result_inlier_cloud->points)
  {
    // Calculate the distance from the point to the plane
    double distance = std::abs(a * pt.x + b * pt.y + c * pt.z + d) / norm;
    sum_squared += distance * distance;
  }
  
  // Return the square root of the average squared distance (roughness)
  return std::sqrt(sum_squared / static_cast<double>(N));
}
//=============================Calculate Relief==========================================================================
// Calculate relief from the inlier cloud (safe landing zone).
inline double calculateRelief(processingResult& result)
{auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
    if (!result_inlier_cloud || result_inlier_cloud->points.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    double z_min = std::numeric_limits<double>::max();
    double z_max = std::numeric_limits<double>::lowest();

    // Iterate through inlier points and compute min and max z values.
    for (const auto &pt : result_inlier_cloud->points) {
        double z = pt.z;
        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
    }
    
    return z_max - z_min;
}
//=============================Calculate Data Confidence=======================================================================
// Calculate data confidence for PCLResult.
// It computes the 2D convex hull (projecting the inlier cloud) and returns N divided by the hull area.
inline double calculateDataConfidence(processingResult& result)
{auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
    if (!result_inlier_cloud || result_inlier_cloud->points.empty()) {
        std::cerr << "Error: Inlier cloud is empty." << std::endl;
        return -1.0;
    }

    size_t N = result_inlier_cloud->points.size();

    // Compute the convex hull of the inlier cloud projected onto a plane.
    pcl::ConvexHull<pcl::PointXYZI> chull;
    chull.setInputCloud(result_inlier_cloud);
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


#endif 


