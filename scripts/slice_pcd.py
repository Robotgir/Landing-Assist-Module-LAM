import open3d as o3d
import numpy as np
import os

def crop_pcd(input_path, output_path, center_x, center_y, size_x, size_y):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        raise ValueError("❌ Loaded point cloud is empty!")

    # Convert to NumPy array
    points = np.asarray(pcd.points)
    print(f"✅ Loaded {points.shape[0]} points from {input_path}")

    # Show bounds to guide slicing
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    print(f"📦 Point Cloud Bounds:")
    print(f"   X: {min_bounds[0]:.2f} to {max_bounds[0]:.2f}")
    print(f"   Y: {min_bounds[1]:.2f} to {max_bounds[1]:.2f}")
    print(f"   Z: {min_bounds[2]:.2f} to {max_bounds[2]:.2f}")

    # Define slice box boundaries
    min_x = center_x - size_x / 2
    max_x = center_x + size_x / 2
    min_y = center_y - size_y / 2
    max_y = center_y + size_y / 2

    # Filter points inside the bounding box
    in_bounds = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    )
    sliced_points = points[in_bounds]

    # Handle empty slices
    if sliced_points.size == 0:
        print("⚠️ No points found in the selected crop region.")
        return

    # Create new point cloud
    sliced_pcd = o3d.geometry.PointCloud()
    sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)

    # Save the cropped point cloud
    success = o3d.io.write_point_cloud(output_path, sliced_pcd)
    if success:
        print(f"✅ Sliced point cloud saved to: {output_path}")
        print(f"🟢 Sliced points: {sliced_points.shape[0]}")
    else:
        print("❌ Failed to write the output PCD file.")

# === 🔧 User-configurable parameters ===
input_pcd_path = "/home/giri/Documents/robotspace/pcds/test.pcd"
output_pcd_path = "/home/giri/Documents/robotspace/pcds/test_100x100m.pcd"
center_x = 300.0      # Center X of slice
center_y = 300.0      # Center Y of slice
size_x = 100.0       # Width in meters
size_y = 100.0       # Height in meters

# Run the crop
crop_pcd(input_pcd_path, output_pcd_path, center_x, center_y, size_x, size_y)
