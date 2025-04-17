import numpy as np
import open3d as o3d
import random
import os

def generate_plane_points(x_range, y_range, z_height, density):
    """Generate points for a plane with given dimensions and point density."""
    x = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) * density))
    y = np.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) * density))
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_height)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points

def generate_block_points(x_range, y_range, z_range, density):
    """Generate points for a 3D block with given dimensions and point density."""
    x = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) * density))
    y = np.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) * density))
    z = np.linspace(z_range[0], z_range[1], int((z_range[1] - z_range[0]) * density))
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points

def check_overlap(new_block, existing_blocks):
    """Check if a new block overlaps with existing blocks."""
    x1, y1, z1 = new_block[0][0], new_block[1][0], new_block[2][0]
    x2, y2, z2 = new_block[0][1], new_block[1][1], new_block[2][1]
    for block in existing_blocks:
        bx1, by1, bz1 = block[0][0], block[1][0], block[2][0]
        bx2, by2, bz2 = block[0][1], block[1][1], block[2][1]
        if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2 or z2 < bz1 or z1 > bz2):
            return True
    return False

def generate_point_cloud(output_path, plane_size=100.0, num_blocks=5, density=10.0):
    """Generate a point cloud with a plane and random blocks, save as .pcd."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate plane points (10m x 10m = 100m², z=0)
    plane_points = generate_plane_points(
        x_range=(-plane_size / 2, plane_size / 2),
        y_range=(-plane_size / 2, plane_size / 2),
        z_height=0.0,
        density=density
    )
    
    # Generate random blocks
    block_points_list = []
    existing_blocks = []
    
    for _ in range(num_blocks):
        # Random block dimensions (1m to 10m)
        width = random.uniform(1.0, 10.0)
        length = random.uniform(1.0, 10.0)
        height = random.uniform(1.0, 10.0)
        
        # Random position for block base within plane bounds
        max_x = plane_size / 2 - width / 2
        max_y = plane_size / 2 - length / 2
        x_center = random.uniform(-max_x, max_y)
        y_center = random.uniform(-max_y, max_y)
        
        # Define block ranges
        block_x_range = (x_center - width / 2, x_center + width / 2)
        block_y_range = (y_center - length / 2, y_center + length / 2)
        block_z_range = (0.0, height)  # Block starts at z=0 (on plane)
        
        # Check for overlap
        block_ranges = (block_x_range, block_y_range, block_z_range)
        if not check_overlap(block_ranges, existing_blocks):
            # Generate block points
            block_points = generate_block_points(block_x_range, block_y_range, block_z_range, density)
            block_points_list.append(block_points)
            existing_blocks.append(block_ranges)
    
    # Combine all points
    all_points = np.vstack([plane_points] + block_points_list)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Optional: Add random colors for visualization
    colors = np.random.rand(len(all_points), 3)  # Random RGB colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to .pcd file
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    print(f"Total points: {len(all_points)}")

# Parameters
output_file = "/home/giri/Documents/robotspace/pcds/custom_point_cloud_100m.pcd"
plane_size = 10.0  # 10m x 10m = 100m²
num_blocks = 5     # Number of blocks
density = 10.0     # Points per meter

# Generate and save point cloud
generate_point_cloud(output_file, plane_size, num_blocks, density)