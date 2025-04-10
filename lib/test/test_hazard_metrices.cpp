#include "hazard_metrices.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "hazard_metrices.h"
#include <gtest/gtest.h>
#include <string>
#include <iostream>

// Global flag to control visualization in tests.
bool g_skipVisualization = false;

// Input file path (adjust if needed)
static const std::string filePath = "/home/airsim_user/Landing-Assist-Module-LAM/test.pcd";

//--------------------------------------------------------------------------
// Test 1: PCA / Normal Estimation / Classification (PCL)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestPCA_NormalEstimation) {
    float slope_threshold = 5.0f;
    int k = 10;
    // Call the PCA-based classification function.
    PCLResult result =  PrincipleComponentAnalysis(filePath,
                                                
                                                    slope_threshold,
                                                    k);
    if (!g_skipVisualization) {
        // Visualize the result; press 'q' to close the viewer.
        visualizePCL(result);
    } else {

        std::cout << "[TestPCA_NormalEstimation] Inliers: " << result.inlier_cloud->size()  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Check that some points have been classified.
    EXPECT_GT(result.inlier_cloud->size() + result.outlier_cloud->size(), 0);

}
//--------------------------------------------------------------------------
// Test 2: Open3D-Based RANSAC Segmentation
//-------------------------------------------------------------------------- 
TEST(HazardMetricesTest, TestOpen3D_RANSAC) {
    double distance_threshold = 1.9;
    int ransac_n = 3;
    int num_iterations = 1000;

    // Call the Open3D RANSAC segmentation function.
    OPEN3DResult result = RansacPlaneSegmentation(filePath, distance_threshold, ransac_n, num_iterations);

    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        visualizeOPEN3D(result);
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
    float distanceThreshold = 1.9f;
    int maxIterations = 1000;

    // Call the PCL RANSAC segmentation function.
    PCLResult result = performRANSAC(filePath, distanceThreshold, maxIterations);

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
    float distanceThreshold = 1.9f;
    int maxIterations = 200;

    // Call the PROSAC segmentation function.
    PCLResult result = performPROSAC(filePath, distanceThreshold, maxIterations);

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
TEST(HazardMetricesTest, TestLeastSquaresPlaneFitting) {;
    double distance_threshold = 1.85;

    // Call the Least Squares plane fitting function.
    OPEN3DResult result = LeastSquaresPlaneFitting(filePath, distance_threshold);

    if (!g_skipVisualization) {
        visualizeOPEN3D(result);
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
    float distanceThreshold = 1.9f;
    int maxIterations = 100;

    // Call the LMEDS segmentation function.
    PCLResult result = performLMEDS(filePath, distanceThreshold, maxIterations);

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
// Test 7: Average 3d Gradient (PCL-Based)
//--------------------------------------------------------------------------

TEST(HazardMetricesTest, TestAverage3DGradient) {
    // Set default values for parameters.
    float neighborRadius = 0.5f;
    float gradientThreshold = 0.15f;
    float angleThreshold = 10.0f;

    // Call the Average3DGradient flatness detection function.
    PCLResult result = Average3DGradient(filePath, neighborRadius, gradientThreshold, angleThreshold);

    if (!g_skipVisualization) {
        // Visualize the result using visualizePCL, with "inlier_cloud" tag.
        visualizePCL(result);
    } else {
        std::cout << "[TestAverage3DGradient] Inliers: " << result.inlier_cloud->size() 
                  << ", Outliers: " << result.outlier_cloud->size() << std::endl;
    }
    // Ensure that some points have been classified.
    EXPECT_GT(result.inlier_cloud->size() + result.outlier_cloud->size(), 0);
}


//--------------------------------------------------------------------------
// Test 8: Region Growing Segmentation (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestRegionGrowing) {;

    // Call the region growing segmentation function.
    PCLResult result = regionGrowingSegmentation(filePath);

    if (!g_skipVisualization) {
        visualizePCL(result);
    } else {
        std::cout << "[TestRegionGrowing] Inliers: " << result.inlier_cloud->size() << std::endl;
    }
    // Check that some inliers are detected.
    EXPECT_GT(result.inlier_cloud->size(), 0);
}


//--------------------------------------------------------------------------
// Test 9: Calculate Roughness (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestRoughnessPCL) {
    float distanceThreshold = 1.9f;
    int maxIterations = 200;

    // Perform PROSAC segmentation (PCL-based)
    PCLResult result = performPROSAC(filePath, distanceThreshold, maxIterations);

    // Calculate roughness using the PCL-based method.
    double roughness = calculateRoughnessPCL(result);
    std::cout << "[TestRoughnessPCL] Roughness of the point cloud: " << roughness << std::endl;
    
    // Verify that a valid roughness value was calculated.
    EXPECT_GE(roughness, 0);
}

//--------------------------------------------------------------------------
// Test 10: Calculate Roughness (Open3D-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestRoughnessOpen3D) {
    double distance_threshold = 1.9;
    int ransac_n = 3;
    int num_iterations = 1000;

    // Perform RANSAC segmentation (Open3D-based)
    OPEN3DResult segmentation_result = RansacPlaneSegmentation(filePath, distance_threshold, ransac_n, num_iterations);

    // Calculate roughness using the Open3D-based method.
    double roughness = calculateRoughnessOpen3D(segmentation_result);
    std::cout << "[TestRoughnessOpen3D] Roughness of the safe landing zone: " << roughness << std::endl;
    
    // Verify that a valid roughness value was calculated.
    EXPECT_GE(roughness, 0);
}

//--------------------------------------------------------------------------
// Test 11: Calculate Relief (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestReliefPCL) {
    float distanceThreshold = 1.9f;
    int maxIterations = 200;

    // Perform PROSAC segmentation (PCL-based)
    PCLResult result = performPROSAC(filePath, distanceThreshold, maxIterations);

    // Calculate relief using the PCL-based method.
    double relief = calculateReliefPCL(result);
    std::cout << "[TestReliefPCL] Relief of the landing zone (PCL-based): " << relief << std::endl;
    
    // Verify that a valid relief value was calculated.
    EXPECT_GE(relief, 0);
}

//--------------------------------------------------------------------------
// Test 12: Calculate Relief (Open3D-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestReliefOpen3D) {
    double distance_threshold = 1.9;
    int ransac_n = 3;
    int num_iterations = 1000;

    // Perform RANSAC segmentation (Open3D-based)
    OPEN3DResult segmentation_result = RansacPlaneSegmentation(filePath, distance_threshold, ransac_n, num_iterations);

    // Calculate relief using the Open3D-based method.
    double relief = calculateReliefOpen3D(segmentation_result);
    std::cout << "[TestReliefOpen3D] Relief of the safe landing zone (Open3D-based): " << relief << std::endl;
    
    // Verify that a valid relief value was calculated.
    EXPECT_GE(relief, 0);
}

//--------------------------------------------------------------------------
// Test 13: Calculate Data Confidence (PCL-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestDataConfidencePCL) {
    float distanceThreshold = 1.9f;
    int maxIterations = 200;

    // Perform PROSAC segmentation (PCL-based)
    PCLResult result = performPROSAC(filePath, distanceThreshold, maxIterations);

    // Calculate data confidence using the PCL-based method.
    double data_confidence = calculateDataConfidencePCL(result);
    std::cout << "[TestDataConfidencePCL] Data confidence (PCL-based): " << data_confidence << std::endl;
    
    // Verify that a valid data confidence value was calculated.
    EXPECT_GE(data_confidence, 0);
}

//--------------------------------------------------------------------------
// Test 14: Calculate Data Confidence (Open3D-Based)
//--------------------------------------------------------------------------
TEST(HazardMetricesTest, TestDataConfidenceOpen3D) {;
    double distance_threshold = 1.9;
    int ransac_n = 3;
    int num_iterations = 1000;

    // Perform RANSAC segmentation (Open3D-based)
    OPEN3DResult segmentation_result = RansacPlaneSegmentation(filePath, distance_threshold, ransac_n, num_iterations);

    // Calculate data confidence using the Open3D-based method.
    double data_confidence = calculateDataConfidenceOpen3D(segmentation_result);
    std::cout << "[TestDataConfidenceOpen3D] Data confidence (Open3D-based): " << data_confidence << std::endl;
    
    // Verify that a valid data confidence value was calculated.
    EXPECT_GE(data_confidence, 0);
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

