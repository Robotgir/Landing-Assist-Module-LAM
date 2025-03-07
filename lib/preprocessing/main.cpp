#include "pointcloud_preprocessing.h"
#include "common.h"

using PointT = pcl::PointXYZI;

int main(int argc, char **argv)
{
    //// Define the input PCD file path.
    std::string filePath = "/home/airsim_user/Landing-Assist-Module-LAM/test.pcd";

    //============================= DATA STRUCTURING ==================================================

    // Create and visualize a 2D Grid Map.
    float gridmap_resolution = 0.1f;
    pcl::PointCloud<PointT>::Ptr grid_map = create2DGridMap(filePath, gridmap_resolution);
    if (!grid_map) {
        return -1;
    }
    visualize2DGridMap(grid_map);

    //// Create and visualize a 3D Grid Map.
    double voxel_size = 0.55; // Adjust voxel size as needed.
    VoxelGridResult voxelgrid_result = create_3d_grid(filePath, voxel_size);
    if (!voxelgrid_result.voxel_grid_ptr || !voxelgrid_result.cloud_ptr) {
        std::cerr << "Failed to create 3d Grid." << std::endl;
        return 1;
    }
    Visualize3dGridMap(voxelgrid_result.voxel_grid_ptr);

    //// Create KDTree.
    float K = 0.1f; // (Parameter can be tuned as needed.)
    KDTreeResult kdtree_result = create_kdtree(filePath, K);
    if (!kdtree_result.kdtree || !kdtree_result.cloud_ptr) {
        std::cerr << "Failed to create KDTree." << std::endl;
        return 1;
    }

    //// Create Octree.
    int max_depth = 10; // Example maximum depth.
    OctreeResult octree_result = create_octree(filePath, max_depth);
    if (!octree_result.octree || !octree_result.cloud_ptr) {
        std::cerr << "Failed to create Octree." << std::endl;
        return 1;
    }

    //// Create and save Octomap.
    std::string octomap_filePath = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/pointcloud.bt";
    double octomap_resolution = 0.05;
    convertPointCloudToOctomap(filePath, octomap_filePath, octomap_resolution);

    //======================================= FILTERING ===============================================

    //// Apply voxel grid filter using Open3D and visualize.
    {
        double voxel_downsample_size = 0.15;
        auto downsampled_cloud = apply_voxel_grid_filter(filePath, voxel_downsample_size);
        if (!downsampled_cloud) {
            std::cerr << "Failed to apply voxel grid filter." << std::endl;
            return 1;
        }
        VisualizeGeometry(downsampled_cloud);
    }
    //================ Apply Statistical Outlier Removal (SOR) filter using Open3D and visualize.===============
    {
        int nb_neighbors = 15;
        double std_ratio = 0.1;
        OPEN3DResult result = apply_sor_filter(filePath, nb_neighbors, std_ratio);
        visualizeOPEN3D(result);
    }
    // ============================= FILTERING USING PCL ======================================================

    // ================= Radius Outlier Removal Filter (PCL) ==================================================================
    {   
        double voxelSize = 0.05;
        double radius_search = 0.9; //0.1 to 0.3,0.3 to 0.7,0.7 to 0.15
        int min_neighbors = 50;      // 5 to 15,10 to 30,20 to 50
        float translation_offset_radius_filter = 0.0f; //change this value to visualize the filtered cloud in a different position along x axis if it 0 filtered and original pointcloud will be in the same position
        PCLResult result = applyRadiusFilter(filePath,voxelSize, radius_search, min_neighbors);
        visualizePCL(result);
    }
    //============================= Bilateral Filter (PCL) =================================================================================================================================================================
    {
        double voxelSize = 0.05;    
        double sigma_s   = 15.0;  
        double sigma_r   = 0.3;
        
        // Before (three args):
        // applyBilateralFilter(filePath, sigma_s, sigma_r);
        
        // After (four args):
        PCLResult result = applyBilateralFilter(filePath, voxelSize, sigma_s, sigma_r);
            
        visualizePCL(result);

    }
    return 0;
}
