#include "hazard_metrices.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>

// Global flag to control visualization in tests.
bool g_skipVisualization = false;

// Input file path (adjust if needed)
static const std::string file_path = "/home/airsim_user/Landing-Assist-Module-LAM/lib/hazard_metrices/test.pcd";

//--------------------------------------------------------------------------
// Test 1: PCA / Normal Estimation / Classification (PCL)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestPCA_NormalEstimation) {
    float voxel_size = 0.45f;
    float slope_threshold = 10.0f;
    int k = 10;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // Call the PCA-based classification function.
    PCLResult result = computeNormalsAndClassifyPoints<PointT>(file_path,
                                                               normals,
                                                               voxel_size,
                                                               slope_threshold,
                                                               k);
    if (!g_skipVisualization) {
        // Visualize the result; press 'q' to close the viewer.
        visualizePCL(result);
    } else {
        std::cout << "[TestPCA_NormalEstimation] Inliers: " << result.inlier_cloud->size() 
                  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Check that some points have been classified.
    EXPECT_GT(result.inlier_cloud->size() + result.outlier_cloud->size(), 0);
}

//--------------------------------------------------------------------------
// Test 2: Open3D-Based RANSAC Segmentation
//-------------------------------------------------------------------------- 
TEST(HazardMetricesTest, TestOpen3D_RANSAC) {
    double voxel_size = 0.1;
    double distance_threshold = 1.9;
    int ransac_n = 3;
    int num_iterations = 1000;

    // Call the Open3D RANSAC segmentation function.
    OPEN3DResult result = RansacPlaneSegmentation(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);

    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        VisualizeOPEN3D(result);
    } else {
        std::cout << "[TestOpen3D_RANSAC] Inliers: " << result.inlier_cloud->points_.size()
                  << ", Outliers: " << result.outlier_cloud->points_.size() << std::endl;
    }
    // Verify that inliers and outliers were produced.
    EXPECT_GT(result.inlier_cloud->points_.size(), 0);
    EXPECT_GT(result.outlier_cloud->points_.size(), 0);
}

//--------------------------------------------------------------------------
// Test 3: PCL-Based RANSAC Segmentation
//-------------------------------------------------------------------------- 
TEST(HazardMetricesTest, TestPCL_RANSAC) {
    float voxelSize = 0.15f;
    float distanceThreshold = 1.9f;
    int maxIterations = 1000;

    // Call the PCL RANSAC segmentation function.
    PCLResult result = performRANSAC(file_path, voxelSize, distanceThreshold, maxIterations);

    if (!g_skipVisualization) {
        visualizePCL(result);
    } else {
        std::cout << "[TestPCL_RANSAC] Inliers: " << result.inlier_cloud->size()
                  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Verify that both inliers and outliers were found.
    EXPECT_GT(result.inlier_cloud->size(), 0);
    EXPECT_GT(result.outlier_cloud->size(), 0);
}

//--------------------------------------------------------------------------
// Test 4: PROSAC Segmentation (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestPROSAC) {
    float voxel_size = 0.15f;
    float distanceThreshold = 1.9f;
    int maxIterations = 200;

    // Call the PROSAC segmentation function.
    PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold, maxIterations);

    if (!g_skipVisualization) {
        visualizePCL(result);
    } else {
        std::cout << "[TestPROSAC] Inliers: " << result.inlier_cloud->size()
                  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Check that segmentation produced inliers and outliers.
    EXPECT_GT(result.inlier_cloud->size(), 0);
    EXPECT_GT(result.outlier_cloud->size(), 0);
}

//--------------------------------------------------------------------------
// Test 5: Least Squares Plane Fitting (OPEN3D-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestLeastSquaresPlaneFitting) {
    double voxel_size = 0.15;
    double distance_threshold = 1.85;

    // Call the Least Squares plane fitting function.
    OPEN3DResult result = LeastSquaresPlaneFitting(file_path, voxel_size, distance_threshold);

    if (!g_skipVisualization) {
        VisualizeOPEN3D(result);
    } else {
        std::cout << "[TestLeastSquaresPlaneFitting] Inliers: " << result.inlier_cloud->points_.size()
                  << ", Outliers: " << result.outlier_cloud->points_.size() << std::endl;
    }
    // Verify that the function produced inliers and outliers.
    EXPECT_GT(result.inlier_cloud->points_.size(), 0);
    EXPECT_GT(result.outlier_cloud->points_.size(), 0);
}

//--------------------------------------------------------------------------
// Test 6: LMEDS Plane Fitting (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestLMEDS) {
    float voxelSize = 0.15f;
    float distanceThreshold = 1.9f;
    int maxIterations = 100;

    // Call the LMEDS segmentation function.
    PCLResult result = performLMEDS(file_path, voxelSize, distanceThreshold, maxIterations);

    if (!g_skipVisualization) {
        visualizePCL(result);
    } else {
        std::cout << "[TestLMEDS] Inliers: " << result.inlier_cloud->size()
                  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Ensure that inliers and outliers are found.
    EXPECT_GT(result.inlier_cloud->size(), 0);
    EXPECT_GT(result.outlier_cloud->size(), 0);
}

//--------------------------------------------------------------------------
// Test 7: Region Growing Segmentation (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestRegionGrowing) {
    double voxel_size = 0.45;

    // Call the region growing segmentation function.
    PCLResult result = regionGrowingSegmentation(file_path, voxel_size);

    if (!g_skipVisualization) {
        visualizePCL(result);
    } else {
        std::cout << "[TestRegionGrowing] Inliers: " << result.inlier_cloud->size() << std::endl;
    }
    // Check that some inliers are detected.
    EXPECT_GT(result.inlier_cloud->size(), 0);
}

//--------------------------------------------------------------------------
// main() for Google Test
//--------------------------------------------------------------------------
int main(int argc, char **argv) {
    // Process additional command-line arguments.
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--no-vis") {
            g_skipVisualization = true;
        }
    }
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
