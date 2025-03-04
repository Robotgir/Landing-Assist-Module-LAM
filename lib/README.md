# 3D Point Cloud Processing and Visualization

This project demonstrates how to process and visualize point clouds using multiple libraries:
- **Open3D** for 3D point cloud processing and visualization.
- **OctoMap** for generating 3D occupancy maps.
- **PCL (Point Cloud Library)** for filtering, downsampling, and visualizing point clouds.

The code includes examples for:
- Loading point clouds.
- Downsampling using a voxel grid filter.
- Creating 2D and 3D grid maps.
- Building KD-Trees and Octrees.
- Applying filters like Statistical Outlier Removal (SOR), Radius Outlier Removal, and Bilateral filtering.
- Visualizing the results using both Open3D and PCL visualizers.
- Converting point clouds to OctoMap format.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [References](#references)

## Dependencies

This project requires:
- [Open3D](http://www.open3d.org/) (for point cloud processing and visualization)
- [OctoMap](https://octomap.github.io/) (for 3D occupancy mapping)
- [PCL (Point Cloud Library)](http://pointclouds.org/documentation/) (for filtering and advanced visualization)
- A modern C++ compiler supporting C++11 or later.

## Download sample pointcloud from this link https://drive.google.com/file/d/1gQqce4ZqsO59hb1R1lowTViAK7js4DPa/view?usp=sharing
## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/athulkrishnaaei/Landing-Assist-Module-LAM
   ```
   change branch 

   ```bash
   git checkout pointcloud_preprocessing
   ```


2. **Change directory and build the project:**
    
   ```bash
   cd Landing-Assist-Module-LAM/lib/preprocessing

   mkdir build
   
   cd build
   ```

   ```bash
   sudo cmake ..
   ```

   ```bash
   sudo make -j
   ```
3. **Run the project:**

   ```bash
   ./main
   ```

## Refrences
## RUNNING THE TEST
 For running test_hazard_metrices with visualization you need to run the following command
   ./test_pointcloud_preprocessing
 WIthout visualization you need to run the following command 
   ./test_pointcloud_preprocessing --no-vis 
    or change line 7 bool g_skipVisualization = false; to bool g_skipVisualization = true;

## Data Structuring Grid based
 1. create 3d grid

    Function name : VoxelGridResult create_3d_grid(const std::string& filename, double voxel_size)
    Link          : https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_voxel_grid.html#ae9239348e9de069abaf5912b92dd2b84

 2. create 2d grid
    
    Function name : VoxelGridResult create_2d_grid(const std::string& filename, double voxel_size)
    Link          : http://pointclouds.org/documentation/classpcl_1_1_grid_minimum.html#details
                    https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/grid_minimum.h

## Data Structuring Tree based

 3. create kdtree
    Function name : KDTreeResult create_kdtree(const std::string& filename, int K)
    Link          : https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_k_d_tree_flann.html#ae9239348e9de069abaf5912b92dd2b84

 4. create octree   
    Function name : OctreeResult create_octree(const std::string& filename, double voxel_size)
    Link          : https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_octree.html#ae9239348e9de069abaf5912b92dd2b84

## Filtering Downsampling

 5. apply voxel grid filter(voxel downsample)
    Function name : std::shared_ptr<open3d::geometry::PointCloud> apply_voxel_grid_filter(const std::string& filename,double voxel_size) 
    Link          : https://www.open3d.org/docs/0.11.0/cpp_api/classopen3d_1_1geometry_1_1_point_cloud.html#a50efddf2d460dccf3de46cd0d38071af

## Filtering Outlier Removal

 6. apply statistical outlier removal
    Function name :SORFilterResult apply_sor_filter(const std::string& filename,int nb_neighbors,double std_ratio)
    Reference Link: https://www.open3d.org/docs/0.6.0/cpp_api/namespaceopen3d_1_1geometry.html#add56e2ec673de3b9289a25095763af6d
                    https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.cpp#L602

 7. radius outlier removal
    Function name :pcl::PointCloud<PointT>::Ptr applyRadiusFilter(const typename pcl::PointCloud<PointT>::Ptr &cloud,double radius_search,int min_neighbors)
    Reference Link:  http://pointclouds.org/documentation/classpcl_1_1_radius_outlier_removal.html
                     https://github.com/PointCloudLibrary/pcl/blob/master/filters/src/radius_outlier_removal.cpp#L47

## Filtering Smoothing

 8. bilateral filter
    Function name : pcl::PointCloud<PointT>::Ptr applyBilateralFilter(const typename pcl::PointCloud<PointT>::Ptr &cloud,double sigma_s,double sigma_r)
    Reference Link: https://pointclouds.org/documentation/classpcl_1_1_bilateral_filter.html
                    https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/bilateral.h#L56


## HAZARD METRICES

## RUNNING THE TEST
 For running test_hazard_metrices with visualization you need to run the following command
   ./test_hazard_metrices 

 Without visualization you need to run the following command 
    ./test_hazard_metrices --no-vis 
    or change line 7 bool g_skipVisualization = false; to bool g_skipVisualization = true;

 For running specific test for eg :PROSAC you can run it with 
   ./test_hazard_metrices --gtest_filter=HazardMetricesTest.TestPROSAC

 1. Principal Component Analysis
    Function name : PCLResult PrincipleComponentAnalysis(const std::string& file_path,
                                          pcl::PointCloud<pcl::Normal>::Ptr normals,
                                          float voxel_size = 0.45f,
                                          float slope_threshold = 20.0f,  // degrees
                                          int k = 10)  // k-nearest neighbors
   Reference Link : https://pointclouds.org/documentation/classpcl_1_1_p_c_a.html

 2. RANSAC Plane Segmentation
    OPEN3D 
    Function name : OPEN3DResult RansacPlaneSegmentation(const std::string& file_path,
                                                         double voxel_size,
                                                         double distance_threshold,
                                                         int ransac_n,
                                                         int num_iterations)      
    Reference Link: https://github.com/isl-org/Open3D/blob/main/examples/cpp/RegistrationRANSAC.cpp 
                    https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html         
    PCL 
    Function name : PCLResult performRANSAC(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
    Reference Link : https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation.html
 
 3. PROSAC Plane Segmentation
    PCL 
    Function name : PCLResult performPROSAC(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100) 
    Reference Link : https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation.html
                     https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html change method type to SAC_PROSAC
 
 4. Least Squares Plane Fitting
    OPEN3D
    Function name : OPEN3DResult LeastSquaresPlaneFitting(const std::string &file_path, double voxel_size, double distance_threshold)
    Reference Link : 
 
 5. Least Of Median Squares Plane Fitting
    PCL
    Function name : PCLResult performLMEDS(const std::string &file_path,
                           float voxelSize = 0.05f,
                           float distanceThreshold = 0.02f,
                           int maxIterations = 100)
    Reference Link : https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation.html
                     https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html change method type to SAC_LMEDS
 
 6. Region Growing Segmentation
    PCL
    Function name : inline PCLResult regionGrowingSegmentation(
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
                           float horizontal_angle_threshold_deg = 10.0f)
    Reference Link : https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html
