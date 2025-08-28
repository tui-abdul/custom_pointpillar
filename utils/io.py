import numpy as np
import os
import pickle
import open3d as o3d

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply',".pcd"]
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    
    elif suffix in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        # if dim > 3 and colors exist, append them
        if dim > 3:
            if np.asarray(pcd.colors).size != 0:
                colors = np.asarray(pcd.colors)
                points = np.hstack([points, colors])
            else:
                # pad with zeros if no color
                points = np.hstack([points, np.zeros((points.shape[0], dim-3))])
        return points
    else:
        raise NotImplementedError(f"Unsupported file format: {suffix}")

import os
import numpy as np

def read_points_with_inverse_rotation(file_path, rotation_matrix, dim=4):
    """
    Read point cloud data from a .bin file and apply the inverse 
    of a predefined rotation matrix.

    Args:
        file_path (str): Path to the .bin file.
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
        dim (int): Number of dimensions to return (default 3: x, y, z).

    Returns:
        np.ndarray: Transformed array of points with shape (N, dim)
    """
    suffix = os.path.splitext(file_path)[1].lower()
    assert suffix == '.bin', f"Unsupported file type: {suffix}"

    # Read binary file as float32
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)

    # Inverse of rotation matrix (transpose since it's orthogonal)
    inv_rotation = rotation_matrix.T

    # Apply inverse rotation only on xyz
    points_rotated = points[:, :3] @ inv_rotation.T

    # If dim > 3, keep extra channels or pad with zeros
    if dim > 3:
        if points.shape[1] >= dim:
            points_rotated = np.hstack([points_rotated, points[:, 3:dim]])
        else:
            points_rotated = np.hstack([points_rotated, np.zeros((points_rotated.shape[0], dim - 3))])

    return points_rotated





def write_points(lidar_points, file_path):
    """
    Write point cloud data to a file, supporting .bin, .ply, .pcd.
    Ensures points are float32 to avoid type issues.
    """
    # Convert to float32
    lidar_points = lidar_points.astype(np.float32)

    suffix = os.path.splitext(file_path)[1].lower()
    assert suffix in ['.bin', '.ply', '.pcd']

    if suffix == '.bin':
        # write as binary float32
        lidar_points.tofile(file_path)

    elif suffix in ['.ply', '.pcd']:
        # create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        points = np.asarray(pcd.points, dtype=np.float32)
        points.tofile(file_path)

        #o3d.io.write_point_cloud(file_path, pcd)

    else:
        raise NotImplementedError(f"Unsupported file format: {suffix}")



def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float32)#[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float32)
    
    return annotation


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
        result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
        result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)
