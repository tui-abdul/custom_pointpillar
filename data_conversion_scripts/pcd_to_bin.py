import os
import numpy as np
import open3d as o3d

def convert_pcd_to_bin(input_folder, output_folder):
    """
    Convert all .pcd files in input_folder to .bin files in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all .pcd files sorted
    pcd_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pcd')])

    for pcd_file in pcd_files:
        input_path = os.path.join(input_folder, pcd_file)
        output_path = os.path.join(output_folder, os.path.splitext(pcd_file)[0] + '.bin')

        print("Converting:", input_path)

        # Read point cloud using Open3D
        pcd = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Optional: add a dummy intensity column if you need 4D points
        if points.shape[1] == 3:
            points_4d = np.zeros((points.shape[0], 4), dtype=np.float32)
            points_4d[:, :3] = points
            points = points_4d

        # Save to .bin
        points.tofile(output_path)
        print(f"Saved: {output_path} ({points.shape[0]} points)")

# Example usage
input_folder = '../../dataset_local/intersection_data_21/sunny/sequence_3/training/velodyne/'
output_folder = '../../dataset_local/intersection_data_21/sunny/sequence_3/training/velodyne/'

convert_pcd_to_bin(input_folder, output_folder)
