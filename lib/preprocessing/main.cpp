#include "pointcloud_preprocessing.h"

int main(int argc, char **argv)
{
    //// Define the input PCD file path.
    std::string filename = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/3_7_2024_dense.pcd";

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // DATA STRUCTURING
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // Create and visualize a 2D Grid Map.
    float gridmap_resolution = 0.1f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr grid_map = create2DGridMap(filename, gridmap_resolution);
    if (!grid_map) {
        return -1;
    }
    visualize2DGridMap(grid_map);

    //// Create and visualize a 3D Grid Map.
    double voxel_size = 0.1; // Adjust voxel size as needed.
    VoxelGridResult voxelgrid_result = create_3d_grid(filename, voxel_size);
    if (!voxelgrid_result.voxel_grid_ptr || !voxelgrid_result.cloud_ptr) {
        std::cerr << "Failed to create 3d Grid." << std::endl;
        return 1;
    }
    Visualize3dGridMap(voxelgrid_result.voxel_grid_ptr);

    //// Create KDTree.
    float K = 0.1f; // (Parameter can be tuned as needed.)
    KDTreeResult kdtree_result = create_kdtree(filename, K);
    if (!kdtree_result.kdtree || !kdtree_result.cloud_ptr) {
        std::cerr << "Failed to create KDTree." << std::endl;
        return 1;
    }

    //// Create Octree.
    int max_depth = 10; // Example maximum depth.
    OctreeResult octree_result = create_octree(filename, max_depth);
    if (!octree_result.octree || !octree_result.cloud_ptr) {
        std::cerr << "Failed to create Octree." << std::endl;
        return 1;
    }

    //// Create and save Octomap.
    std::string octomap_filename = "/home/airsim_user/Landing-Assist-Module-LAM/lib/preprocessing/pointcloud.bt";
    double octomap_resolution = 0.05;
    convertPointCloudToOctomap(filename, octomap_filename, octomap_resolution);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // FILTERING
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    //// Apply voxel grid filter using Open3D and visualize.
    double voxel_downsample_size = 0.15;
    auto downsampled_cloud = apply_voxel_grid_filter(filename, voxel_downsample_size);
    if (!downsampled_cloud) {
        std::cerr << "Failed to apply voxel grid filter." << std::endl;
        return 1;
    }
    VisualizeGeometry(downsampled_cloud);

    //// Apply Statistical Outlier Removal (SOR) filter using Open3D and visualize.
    int nb_neighbors = 15;
    double std_ratio = 0.1;
    SORFilterResult sor_result = apply_sor_filter(filename, nb_neighbors, std_ratio);
    if (!sor_result.filtered_cloud) {
        std::cerr << "Error: SOR filtering failed." << std::endl;
        return 1;
    }
    visualize_sor_filtered_point_cloud(sor_result.original_cloud, sor_result.filtered_cloud);

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // /////////////////////////FILTERING USING PCL///////////////////////////////////////////////////////////
    // //////////////////////////////////////////////////////////////////////////////////////////////////////

    // //// For further filtering with PCL, load the point cloud as PointXYZI.
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud_original) == -1) {
        std::cerr << "[ERROR] Failed to load original PCD file: " << filename << std::endl;
        return -1;
    }
    std::cout << "[INFO] Loaded " << cloud_original->size() << " points from " << filename << std::endl;

    //// Downsample the PCL point cloud.
    float leaf_size = 0.1f;
    auto cloud_downsampled = downsamplePointCloud<pcl::PointXYZI>(cloud_original, leaf_size);

    // // /////////////////////////////////////////////////////////////////////////////////////////////////////
    // // // WHile using this code comment line 78 to 87 ,this function use voxeldownsampled pointcloud
    // // //// --- Radius Outlier Removal Filter (PCL) ---
    double radius_search = 0.1; //0.1 to 0.3,0.3 to 0.7,0.7 to 0.15
    int min_neighbors = 4;      // 5 to 15,10 to 30,20 to 50
    float translation_offset = 0.0f; //change this value to visualize the filtered cloud in a different position along x axis if it 0 filtered and original pointcloud will be in the same position
    std::cout << "[SETTINGS] filename: " << filename
              << ", radius_search: " << radius_search
              << ", min_neighbors: " << min_neighbors << std::endl;
    auto cloud_radius_filtered = applyRadiusFilter<pcl::PointXYZI>(cloud_downsampled, radius_search, min_neighbors);
    if (!cloud_radius_filtered->empty()) {
        visualizeClouds<pcl::PointXYZI>(cloud_downsampled, cloud_radius_filtered,
                                         "Radius Outlier Removal",
                                         "original cloud",
                                         "radius_filtered cloud",
                                         2, translation_offset);
    } else {
        std::cerr << "[ERROR] Radius Outlier Removal resulted in an empty cloud. Skipping visualization." << std::endl;
    }

    // // ////// --- Bilateral Filter (PCL) ---
    // // //// WHile using this code comment line 78 to 87 ,this function use voxeldownsampled pointcloud
    double sigma_s = 15.0; // Small point clouds or detailed structures: sigma_s = 1.0 - 5.0 ,Noisy or dense point clouds: sigma_s = 5.0 - 10.0,Large or very noisy point clouds: sigma_s = 10.0 - 15.0
    double sigma_r = 0.3;  //Preserve edges and details: sigma_r = 0.05 - 0.1, Moderate smoothing: sigma_r = 0.1 - 0.2, Heavy denoising (risk of over-smoothing): sigma_r = 0.2 - 0.3
    float translation_offset = 0.0f; //change this value to visualize the filtered cloud in a different position along x axis if it 0 filtered and original pointcloud will be in the same position
    std::cout << "[SETTINGS] filename: " << filename
              << ", sigma_s: " << sigma_s
              << ", sigma_r: " << sigma_r << std::endl;
    auto cloud_bilateral_filtered = applyBilateralFilter<pcl::PointXYZI>(cloud_downsampled, sigma_s, sigma_r);
    if (!cloud_bilateral_filtered->empty()) {
        visualizeClouds<pcl::PointXYZI>(cloud_downsampled, cloud_bilateral_filtered,
                                         "Bilateral Filter",
                                         "original cloud",
                                         "bilateral_filtered cloud",
                                         2, translation_offset);
    } else {
        std::cerr << "[ERROR] Bilateral Filter resulted in an empty cloud. Skipping visualization." << std::endl;
    }

    return 0;
}
