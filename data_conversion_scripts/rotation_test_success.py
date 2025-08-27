import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ---------- User Inputs ----------
pcd_file = "/mnt/sda/Abdul_Haq/intersection_dataset/sunny/sequence_1/lidar_point_cloud_0/1.pcd"


# Bounding box parameters
bbox_position = np.array([11.305885822912918, -18.217732301430257, 1.6990770818492196])
bbox_scale    = np.array([3.9612818032346797, 1.9517761240434666, 1.4908074495563661])  # [L, W, H]
bbox_rotation = np.array([
    -0.026999180933650745,  # pitch  (x-axis, rad)
    -0.4114445651660462,    # yaw (y-axis, rad)
     0.5257318950733301     # roll   (z-axis, rad)
])

# -----------------------------
# Step 1: Load point cloud
# -----------------------------
pcd = o3d.io.read_point_cloud(pcd_file)
pcd.paint_uniform_color([0.2, 0.7, 0.3])  # green

# -----------------------------
# Step 2: Create oriented bounding box
# -----------------------------
# Rotation matrices
roll, pitch, yaw = bbox_rotation

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll),  np.cos(roll)]
])

Ry = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])

Rz = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0, 0, 1]
])

R_bbox = Ry @ Rz @ Rx  # pitch -> yaw -> roll

obb = o3d.geometry.OrientedBoundingBox()
obb.center = bbox_position
obb.extent = bbox_scale
obb.R = R_bbox
obb.color = [1, 0, 0]  # red

# -----------------------------
# Step 3: Add coordinate axes
# -----------------------------
world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

box_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
box_axis.rotate(obb.R, center=[0, 0, 0])
box_axis.translate(obb.center)

# -----------------------------
# Step 4: Visualize
# -----------------------------
o3d.visualization.draw_geometries([pcd, obb, world_axis, box_axis])