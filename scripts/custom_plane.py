import open3d as o3d
import numpy as np

# Parameters
width = 100    # meters
height = 100   # meters
resolution = 0.01  # spacing between points in meters (e.g., 0.01 = 1 cm)

if resolution <= 0:
    raise ValueError("Resolution must be a positive number.")

# Create coordinate arrays using np.arange for fixed spacing
x = np.arange(-width / 2, width / 2 + resolution, resolution)
y = np.arange(-height / 2, height / 2 + resolution, resolution)
xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)

# Stack to get (N, 3) points
points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Save to PCD
output_path = "/home/giri/Documents/robotspace/pcds/plane_100x100m_001.pcd"
o3d.io.write_point_cloud(output_path, pcd)
print(f"PCD file saved as {output_path}")
