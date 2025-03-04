#include "hazard_metrices.h"
#include "pointcloud_preprocessing.h"

int main(int argc, char **argv)
{
    // Input file (same for all blocks)
    static const std::string file_path = "/home/airsim_user/Landing-Assist-Module-LAM/lib/hazard_metrices/test.pcd";

    //=============================================================================================
    //Block 1: PCA / Normal Estimation / Classification (PCL) TESTED BUT SPEED SHOULD BE INCREASED
    //=============================================================================================

    {

        float voxel_size =0.35f;
        float slope_threshold = 5.0f;
        int k =10;
        
      // Get classification results including the downsampled point cloud.
        PCLResult result = PrincipleComponentAnalysis(file_path,    
                                                      voxel_size ,
                                                      slope_threshold,  // slope threshold
                                                      k);    // k-nearest neighbors

      // Use the stored downsampled point cloud in the visualization function.
        visualizePCL(result);

    }

    //======================================================
    // Block 2: Open3D-Based RANSAC Segmentation 
    //======================================================

    {
      // Parameters for downsampling and RANSAC.
      double voxel_size = 0.1;          // Example voxel size.
      double distance_threshold = 1.9;   // Threshold for plane segmentation.
      int ransac_n = 3;                  // Number of points to sample for plane fitting.
      int num_iterations = 1000;         // Number of RANSAC iterations.

      // Perform RANSAC plane segmentation.
      OPEN3DResult result = RansacPlaneSegmentation(
          file_path, voxel_size, distance_threshold, ransac_n, num_iterations);

      // Visualize the segmentation result.
      //VisualizeRansacSegmentationOpen3d(result.inlier_cloud, result.outlier_cloud);
      VisualizeOPEN3D(result);
    }

    //======================================================
    // Block 3: PCL-Based RANSAC Segmentation 
    //======================================================

    {
        float voxelSize = 0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 1000;

        PCLResult result = performRANSAC(file_path, voxelSize, distanceThreshold, maxIterations);
        visualizePCL(result,"inlier_cloud");
        
    }

    //======================================================
    // Block 4: PROSAC Segmentation (PCL-Based) 
    //======================================================

    {
        float voxel_size =0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 200;

        PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold,maxIterations);
        visualizePCL(result,"inlier_cloud");
    }

    //===========================================================
    // Block 5: Least Squares Plane Fitting (OPEN3D-Based)
    //===========================================================

       {

        OPEN3DResult result;

        double voxel_size = 0.15;         // Voxel size for downsampling (adjust as needed)
        double distance_threshold = 1.85; // Threshold for inlier classification (in same units as your point cloud)

        result = LeastSquaresPlaneFitting(file_path, voxel_size, distance_threshold);
        VisualizeOPEN3D(result);
  
       }

    //==================================================================
    // Block 6: Least of Median Squares Plane Fitting (PCL-Based) 
    //==================================================================

      {
        float voxelSize = 0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 100;


        // Perform plane fitting using LMedS.
        PCLResult result = performLMEDS(file_path, voxelSize, distanceThreshold,maxIterations);
        
        // Visualize the result.
        visualizePCL(result);
      }
    // ===============================================================
    // Block 7: Average 3d Gradients (PCL-Based) not completed
    // ===============================================================

      {

        // Set default values for parameters
        float voxelSize = 0.1f;
        float neighborRadius = 0.5f;
        float gradientThreshold = 0.15f;

    

        // Call the flatness detection function
        PCLResult result = Average3DGradient(file_path, voxelSize, neighborRadius, gradientThreshold);
        visualizePCL(result,"inlier_cloud");


      }

    // ===============================================================
    // Block 8: Region growing segmentation (PCL-Based) 
    // ===============================================================

      {
        double voxel_size = 0.45;
    
        PCLResult result = regionGrowingSegmentation(file_path, voxel_size);
        visualizePCL(result,"inlier_cloud");
      
      }
    
    //===============================================================
    // Block 9: Calculate Roughness (PCL-Based) 
    //===============================================================

    { 
      // Find the potential plane.
      float voxel_size =0.15f;
      float distanceThreshold = 1.9f;
      int maxIterations = 200;

      
      PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold,maxIterations);
      
      double roughness = calculateRoughnessPCL(result);
      if (roughness >= 0)
        std::cout << "Roughness of the point cloud: " << roughness << std::endl;
      else
        std::cerr << "Failed to calculate roughness." << std::endl;
    }

    //===============================================================
    // Block 10: Calculate Roughness (OPEN3D-Based) 
    //===============================================================

     {
      
      double voxel_size = 0.1;
      double distance_threshold = 1.9;
      int ransac_n = 3;
      int num_iterations = 1000;
    
      // Perform RANSAC segmentation.
      OPEN3DResult segmentation_result = RansacPlaneSegmentation(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);
    
      // Calculate roughness using the inlier cloud and the stored plane model.
      double roughness = calculateRoughnessOpen3D(segmentation_result);
    
      std::cout << "Stored plane model: " << segmentation_result.plane_model.transpose() << std::endl;
      std::cout << "Roughness of the safe landing zone: " << roughness << std::endl;

     }

    //===============================================================
    // Block 11: Calculate Relief (PCL-Based) 
    //===============================================================

    {
            // Find the potential plane.
      float voxel_size =0.15f;
      float distanceThreshold = 1.9f;
      int maxIterations = 200;

      
      PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold,maxIterations);
      
      double relief = calculateReliefPCL(result);
      if (relief >= 0)
        std::cout << "relief of the landing zone pcl based : " << relief << std::endl;
      else
        std::cerr << "Failed to calculate relief." << std::endl;
    }


    //===============================================================
    // Block 12: Calculate Relief (OPEN3D-Based) 
    //===============================================================

    {
      double voxel_size = 0.1;
      double distance_threshold = 1.9;
      int ransac_n = 3;
      int num_iterations = 1000;
    
      // Perform RANSAC segmentation.
      OPEN3DResult segmentation_result = RansacPlaneSegmentation(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);
    
      // Calculate relief using the inlier cloud and the stored plane model.
      double relief = calculateReliefOpen3D(segmentation_result);
    
      std::cout << "Stored plane model: " << segmentation_result.plane_model.transpose() << std::endl;
      std::cout << "relief of the safe landing zone open3d based : " << relief << std::endl;
    }

    // ===============================================================
    // Block 13: Calculate Data Confidence (PCL-Based) 
    // ===============================================================

    {
            // Find the potential plane.
      float voxel_size =0.15f;
      float distanceThreshold = 1.9f;
      int maxIterations = 200;

      
      PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold,maxIterations);
      
      double data_confidence= calculateDataConfidencePCL(result);
      if (data_confidence>= 0)
        std::cout << "data confidence of the landing zone pcl based : " << data_confidence<< std::endl;
      else
        std::cerr << "Failed to calculate relief." << std::endl;
    }


    // ===============================================================
    // Block 14: Calculate Data Confidence (OPEN3D-Based) 
    // ===============================================================

    {
      double voxel_size = 0.01;
      double distance_threshold = 1.9;
      int ransac_n = 3;
      int num_iterations = 1000;
    
      // Perform RANSAC segmentation.
      OPEN3DResult segmentation_result = RansacPlaneSegmentation(file_path, voxel_size, distance_threshold, ransac_n, num_iterations);
    
      // Calculate data_confidenceusing the inlier cloud and the stored plane model.
      double data_confidence= calculateDataConfidenceOpen3D(segmentation_result);
    
      std::cout << "Stored plane model: " << segmentation_result.plane_model.transpose() << std::endl;
      std::cout << "data confidence of the safe landing zone open3d based : " << data_confidence<< std::endl;
    }

    

    return 0;
}
