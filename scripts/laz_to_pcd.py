import laspy
import open3d as o3d
import numpy as np

def laz_to_pcd(input_laz_path, output_pcd_path):
    # Read the .laz file
    las = laspy.read(input_laz_path)
    
    # Extract point coordinates (x, y, z)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optionally, include colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0  # Normalize to [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to .pcd file
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Converted {input_laz_path} to {output_pcd_path}")

# Example usage
input_file = "input.laz"
output_file = "output.pcd"
laz_to_pcd(input_file, output_file)
