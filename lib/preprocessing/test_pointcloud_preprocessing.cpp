#include "pointcloud_preprocessing.h"
#include <gtest/gtest.h>
#include <fstream>

// Change this file path if needed.
static const std::string filename = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/3_7_2024_dense.pcd";

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
    auto grid_map = create2DGridMap(filename, gridmap_resolution);
    // Check that the returned pointer is not null and has points.
    EXPECT_NE(grid_map, nullptr);
    if(grid_map)
    {
        EXPECT_GT(grid_map->size(), 0);
    }
}

TEST(DataStructuring, Create3DGridMap) {
    double voxel_size = 0.1;
    VoxelGridResult voxelgrid_result = create_3d_grid(filename, voxel_size);
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
    KDTreeResult kdtree_result = create_kdtree(filename, K);
    EXPECT_NE(kdtree_result.cloud_ptr, nullptr);
    EXPECT_NE(kdtree_result.kdtree, nullptr);
    if(kdtree_result.cloud_ptr)
    {
        EXPECT_GT(kdtree_result.cloud_ptr->points_.size(), 0);
    }
}

TEST(DataStructuring, CreateOctree) {
    int max_depth = 10;
    OctreeResult octree_result = create_octree(filename, max_depth);
    EXPECT_NE(octree_result.cloud_ptr, nullptr);
    EXPECT_NE(octree_result.octree, nullptr);
    if(octree_result.cloud_ptr)
    {
        EXPECT_GT(octree_result.cloud_ptr->points_.size(), 0);
    }
}

TEST(DataStructuring, ConvertPointCloudToOctomap) {
    std::string octomap_filename = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/pointcloud.bt";
    double octomap_resolution = 0.05;
    // Call the conversion function.
    convertPointCloudToOctomap(filename, octomap_filename, octomap_resolution);
    // Check that the octomap file was created.
    EXPECT_TRUE(fileExists(octomap_filename));
}

////////////////////////////////////////////////////////////
// TESTS for FILTERING functions
////////////////////////////////////////////////////////////

TEST(Filtering, ApplyVoxelGridFilter) {
    double voxel_downsample_size = 0.15;
    auto downsampled_cloud = apply_voxel_grid_filter(filename, voxel_downsample_size);
    EXPECT_NE(downsampled_cloud, nullptr);
    if(downsampled_cloud)
    {
        EXPECT_GT(downsampled_cloud->points_.size(), 0);
    }
}

TEST(Filtering, ApplySORFilter) {
    int nb_neighbors = 15;
    double std_ratio = 0.1;
    SORFilterResult sor_result = apply_sor_filter(filename, nb_neighbors, std_ratio);
    EXPECT_NE(sor_result.original_cloud, nullptr);
    EXPECT_NE(sor_result.filtered_cloud, nullptr);
    if(sor_result.original_cloud)
    {
        EXPECT_GT(sor_result.original_cloud->points_.size(), 0);
    }
    if(sor_result.filtered_cloud)
    {
        // In many cases the filtered cloud should be smaller than the original.
        EXPECT_LT(sor_result.filtered_cloud->points_.size(),
                  sor_result.original_cloud->points_.size());
    }
}

////////////////////////////////////////////////////////////
// TESTS for PCL-based filtering functions
////////////////////////////////////////////////////////////

TEST(PCLFiltering, DownsamplePointCloud) {
    // Load the PCL point cloud.
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
    int load_ret = pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud_original);
    EXPECT_NE(load_ret, -1) << "Failed to load original PCD file.";
    EXPECT_GT(cloud_original->size(), 0);

    float leaf_size = 0.1f;
    auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, leaf_size);
    EXPECT_NE(cloud_downsampled, nullptr);
    if(cloud_downsampled)
    {
        EXPECT_GT(cloud_downsampled->size(), 0);
    }
}

TEST(PCLFiltering, ApplyRadiusFilter) {
    // Load original cloud.
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
    int load_ret = pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud_original);
    EXPECT_NE(load_ret, -1) << "Failed to load original PCD file.";
    EXPECT_GT(cloud_original->size(), 0);

    float leaf_size = 0.1f;
    auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, leaf_size);
    double radius_search = 0.1;
    int min_neighbors = 4;
    auto cloud_radius_filtered = applyRadiusFilter<pcl::PointXYZI>(cloud_downsampled, radius_search, min_neighbors);
    // Check that the filtered cloud is not empty.
    EXPECT_FALSE(cloud_radius_filtered->empty());
}

TEST(PCLFiltering, ApplyBilateralFilter) {
    // Load original cloud.
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
    int load_ret = pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud_original);
    EXPECT_NE(load_ret, -1) << "Failed to load original PCD file.";
    EXPECT_GT(cloud_original->size(), 0);

    float leaf_size = 0.1f;
    auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, leaf_size);
    double sigma_s = 15.0;
    double sigma_r = 0.3;
    auto cloud_bilateral_filtered = applyBilateralFilter<pcl::PointXYZI>(cloud_downsampled, sigma_s, sigma_r);
    // Check that the bilateral filtered cloud is not empty.
    EXPECT_FALSE(cloud_bilateral_filtered->empty());
}

////////////////////////////////////////////////////////////
// Main function for Google Test
////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Note: We do not call visualization functions here in order not to block the tests.
    return RUN_ALL_TESTS();
}
