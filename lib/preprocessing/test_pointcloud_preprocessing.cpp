#include "pointcloud_preprocessing.h"
#include <gtest/gtest.h>
#include <fstream>
#include <common.h>


// Global flag to control visualization in tests.
bool g_skipVisualization = false;

// Change this file path if needed.
static const std::string filePath = "/home/airsim_user/Landing-Assist-Module-LAM/test.pcd";

// Helper function to check if a file exists.
bool fileExists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

////////////////////////////////////////////////////////////
// TESTS for DATA STRUCTURING functions
////////////////////////////////////////////////////////////

TEST(DataStructuring, Create2DGridMap) {
    float gridmap_resolution = 0.1f;
    pcl::PointCloud<pcl::PointXYZI>::Ptr grid_map = create2DGridMap(filePath, gridmap_resolution);

    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        visualize2DGridMap(grid_map);
    }else{
        std::cout << "Grid map is null or visualization is skipped." << std::endl;
    } 

    // Check that the returned pointer is not null and has points.
    EXPECT_NE(grid_map, nullptr);
    if(grid_map)
    {
        EXPECT_GT(grid_map->size(), 0);
    }
}

TEST(DataStructuring, Create3DGridMap) {
    double voxel_size = 0.55;
    VoxelGridResult voxelgrid_result = create_3d_grid(filePath, voxel_size);

    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        Visualize3dGridMap(voxelgrid_result.voxel_grid_ptr);
    } 
    
    EXPECT_NE(voxelgrid_result.cloud_ptr, nullptr);
    EXPECT_NE(voxelgrid_result.voxel_grid_ptr, nullptr);
    if(voxelgrid_result.cloud_ptr)
    {
        EXPECT_GT(voxelgrid_result.cloud_ptr->points_.size(), 0);
    }
    if(voxelgrid_result.voxel_grid_ptr)
    {
        EXPECT_GT(voxelgrid_result.voxel_grid_ptr->voxels_.size(), 0);
    }
}

TEST(DataStructuring, CreateKDTree) {
    float K = 0.1f;
    KDTreeResult kdtree_result = create_kdtree(filePath, K);
    EXPECT_NE(kdtree_result.cloud_ptr, nullptr);
    EXPECT_NE(kdtree_result.kdtree, nullptr);
    if(kdtree_result.cloud_ptr)
    {
        EXPECT_GT(kdtree_result.cloud_ptr->points_.size(), 0);
    }
}

TEST(DataStructuring, CreateOctree) {
    int max_depth = 10;
    OctreeResult octree_result = create_octree(filePath, max_depth);
    EXPECT_NE(octree_result.cloud_ptr, nullptr);
    EXPECT_NE(octree_result.octree, nullptr);
    if(octree_result.cloud_ptr)
    {
        EXPECT_GT(octree_result.cloud_ptr->points_.size(), 0);
    }
}

TEST(DataStructuring, ConvertPointCloudToOctomap) {
    std::string octomap_filePath = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/pointcloud.bt";
    double octomap_resolution = 0.05;
    // Call the conversion function.
    convertPointCloudToOctomap(filePath, octomap_filePath, octomap_resolution);
    // Check that the octomap file was created.
    EXPECT_TRUE(fileExists(octomap_filePath));
}

////////////////////////////////////////////////////////////
// TESTS for FILTERING functions
////////////////////////////////////////////////////////////

TEST(OPEN3DFiltering, ApplyVoxelGridFilter) {
    double voxel_downsample_size = 0.15;
    auto downsampled_cloud = apply_voxel_grid_filter(filePath, voxel_downsample_size);
    
    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        VisualizeGeometry(downsampled_cloud);
    }   
    EXPECT_NE(downsampled_cloud, nullptr);
    if(downsampled_cloud)
    {
        EXPECT_GT(downsampled_cloud->points_.size(), 0);
    }
}

TEST(OPEN3DFiltering, ApplySORFilter) {
    int nb_neighbors = 15;
    double std_ratio = 0.1;
    OPEN3DResult result = apply_sor_filter(filePath, nb_neighbors, std_ratio);
    

    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        visualizeOPEN3D(result);
    } 
    EXPECT_NE(result.inlier_cloud, nullptr);
    EXPECT_NE(result.inlier_cloud, nullptr);
    if(result.inlier_cloud)
    {
        EXPECT_GT(result.inlier_cloud->points_.size(), 0);
    }

}

////////////////////////////////////////////////////////////
// TESTS for PCL-based filtering functions
////////////////////////////////////////////////////////////


TEST(PCLFiltering, ApplyRadiusFilter) {
    // Load original cloud.
    double voxel_size=0.05;
    double radius_search = 0.9; //0.1 to 0.3,0.3 to 0.7,0.7 to 0.15
    int min_neighbors = 50;      // 5 to 15,10 to 30,20 to 50
    PCLResult result = applyRadiusFilter(filePath, voxel_size, radius_search, min_neighbors);
    
    if (!g_skipVisualization) {
        // Visualize the segmentation result; press 'q' to close the window.
        visualizePCL(result);
    } 
    
    // Check that the filtered cloud is not empty.
    EXPECT_FALSE(result.inlier_cloud->empty());
}

TEST(PCLFiltering, ApplyBilateralFilter) {

    double voxelSize = 0.05;   
    double sigma_s = 15.0; // Small point clouds or detailed structures: sigma_s = 1.0 - 5.0 ,Noisy or dense point clouds: sigma_s = 5.0 - 10.0,Large or very noisy point clouds: sigma_s = 10.0 - 15.0
    double sigma_r = 0.3;  //Preserve edges and details: sigma_r = 0.05 - 0.1, Moderate smoothing: sigma_r = 0.1 - 0.2, Heavy denoising (risk of over-smoothing): sigma_r = 0.2 - 0.3

    PCLResult result = applyBilateralFilter(filePath, voxelSize, sigma_s, sigma_r);

    
    if (!g_skipVisualization) {
        visualizePCL(result);
    } 

    // Check that the bilateral filtered cloud is not empty.
    EXPECT_FALSE(result.inlier_cloud->empty());
}

////////////////////////////////////////////////////////////
// Main function for Google Test
////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--no-vis") {
            g_skipVisualization = true;
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    // Note: We do not call visualization functions here in order not to block the tests.
    return RUN_ALL_TESTS();
}
