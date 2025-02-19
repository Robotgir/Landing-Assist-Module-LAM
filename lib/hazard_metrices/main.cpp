
#include "hazard_metrices.h"

int main(int argc, char **argv)
{
    // Input file (same for all blocks)
    static const std::string file_path = "/home/airsim_user/Landing-Assist-Module-LAM/lib/hazard_metrices/test.pcd";

    // pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *original_cloud) == -1)
    // {
    //     PCL_ERROR("Couldn't read file %s\n", file_path.c_str());
    //     return -1;
    // }
    // std::cout << "Loaded " << original_cloud->size() << " points from " << file_path << std::endl;

    //=============================================================================================
    //Block 1: PCA / Normal Estimation / Classification (PCL) TESTED BUT SPEED SHOULD BE INCREASED
    //=============================================================================================
    {

        float voxel_size =0.45f;
        float slope_threshold = 10.0f;
        int k =10;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

      // Get classification results including the downsampled point cloud.
        PCLResult classification = computeNormalsAndClassifyPoints<PointT>(file_path,
                                                                                           normals,
                                                                                           voxel_size ,
                                                                                           slope_threshold,  // slope threshold
                                                                                           k);    // k-nearest neighbors

      // Use the stored downsampled point cloud in the visualization function.
        int point_size = 3;
        visualizePCL(classification);
 
    }

    //======================================================
    // Block 2: Open3D-Based RANSAC Segmentation TESTED
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
    // Block 3: PCL-Based RANSAC Segmentation TESTED
    //======================================================
    {
        float voxelSize = 0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 1000;

        PCLResult result = performRANSAC(file_path, voxelSize, distanceThreshold, maxIterations);
        visualizePCL(result);
        
    }

    //======================================================
    // Block 4: PROSAC Segmentation (PCL-Based) TESTED
    //======================================================
    {
        float voxel_size =0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 200;

        PCLResult result = performPROSAC(file_path, voxel_size, distanceThreshold,maxIterations);
        visualizePCL(result);
    }

    //======================================================
    // Block 5: Least Squares Plane Fitting (OPEN3D-Based) TESTED
    //======================================================
       {

        OPEN3DResult result;

        double voxel_size = 0.15;         // Voxel size for downsampling (adjust as needed)
        double distance_threshold = 1.85; // Threshold for inlier classification (in same units as your point cloud)

        result = LeastSquaresPlaneFitting(file_path, voxel_size, distance_threshold);
        VisualizeOPEN3D(result);
  
       }
    //===============================================================
    // Block 6: Least of Median Squares Plane Fitting (PCL-Based) TESTED
    //===============================================================

      {
        float voxelSize = 0.15f;
        float distanceThreshold = 1.9f;
        int maxIterations = 100;


        // Perform plane fitting using LMedS.
        PCLResult result = performLMEDS(file_path, voxelSize, distanceThreshold,maxIterations);
        
        // Visualize the result.
        visualizePCL(result);
      }
    //===============================================================
    // Block 7: Average 3d Gradients (PCL-Based) not completed
    //===============================================================
      // {
      //   float slope_threshold = 20.0f;

      //   // Compute normals and classify the points.
      //   auto classified_clouds = computeNormalsUsingAverage3DGradient(original_cloud, slope_threshold);

      //   // Visualize the inliers and outliers.
      //   visualizeClassifiedCloud(classified_clouds.first, classified_clouds.second);
      // }

    //===============================================================
    // Block 8: Region growing segmentation (PCL-Based) TESTED
    //===============================================================
      {
        double voxel_size = 0.45;
    
        PCLResult result = regionGrowingSegmentation(file_path, voxel_size);
        visualizePCL(result);
      
      }
    return 0;
}
