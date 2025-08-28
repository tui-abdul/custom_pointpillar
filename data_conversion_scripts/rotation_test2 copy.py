import numpy as np
import open3d as o3d

# -----------------------------
# User inputs
# -----------------------------
pcd_file = "/mnt/sda/Abdul_Haq/intersection_dataset/sunny/sequence_1/lidar_point_cloud_0/1.pcd"

bbox_position = np.array([11.305885822912918, -18.217732301430257, 1.6990770818492196])
bbox_scale    = np.array([3.9612818032346797, 1.9517761240434666, 1.4908074495563661])
bbox_rotation = np.array([
    -0.026999180933650745,  # roll  (x-axis, rad)
    -0.4114445651660462,    # pitch    (y-axis, rad)
     0.5257318950733301     # yaw  (z-axis, rad)
])

rotation_matrix = np.array([
    [ 0.95148668,  0.00724281, -0.30760468],
    [ 0.00724281,  0.99891868,  0.04592391],
    [ 0.30760468, -0.04592391,  0.95040536]
])

# -----------------------------
# Step 1: Compute inverse rotation
# -----------------------------
R_inv = rotation_matrix.T  # inverse of a rotation matrix

# -----------------------------
# Step 2: Load and rotate point cloud
# -----------------------------
pcd = o3d.io.read_point_cloud(pcd_file)
pcd_points = np.asarray(pcd.points)

# Apply inverse rotation
rotated_points = pcd_points @ R_inv.T
pcd_rotated = o3d.geometry.PointCloud()
pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)
pcd_rotated.paint_uniform_color([0.2, 0.7, 0.3])  # green

# -----------------------------
# Step 3: Create oriented bounding box
# -----------------------------
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

R_bbox = Ry @ Rz @ Rx

# Apply inverse rotation to bbox rotation and center
R_bbox = R_inv @ R_bbox
rotated_center = R_inv @ bbox_position
print("final rotation", R_bbox)
obb = o3d.geometry.OrientedBoundingBox()
obb.center = rotated_center
obb.extent = bbox_scale
obb.R = R_bbox
obb.color = [1, 0, 0]  # red

yaw = 0.51673786  # radians

R_yaw = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0,            0,           1]
])
# -----------------------------
# Step 4: Add axes
# -----------------------------
world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

box_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
box_axis.rotate(R_yaw, center=[0, 0, 0])
box_axis.translate(rotated_center)

# -----------------------------
# Step 5: Visualize
# -----------------------------
o3d.visualization.draw_geometries([pcd_rotated, obb, world_axis, box_axis])
