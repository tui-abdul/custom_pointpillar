import numpy as np



import numpy as np
import math

def rotation_matrix_to_euler_zyx(R, degrees=False):
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw)
    using the ZYX convention:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        R (array-like): 3x3 rotation matrix
        degrees (bool): if True, return angles in degrees (default False -> radians)

    Returns:
        (roll, pitch, yaw): tuple of floats (in radians by default)
            roll  - rotation about x-axis
            pitch - rotation about y-axis
            yaw   - rotation about z-axis
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")

    # sy = sqrt(R00^2 + R10^2)  (avoid singularities when sy ~ 0)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch ~= +-90 deg
        # set yaw = 0 and compute roll from remaining entries
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)  # sy is ~0 here
        yaw   = 0.0

    if degrees:
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    else:
        return (roll, pitch, yaw)





def transform_bbox(rotation_matrix, bbox_rotation, bbox_position):
    """
    Apply the inverse of `rotation_matrix` to a bounding box.

    Args:
        rotation_matrix (np.ndarray): 3x3 world rotation matrix
        bbox_rotation (np.ndarray): 3x3 rotation of bounding box
        bbox_position (np.ndarray): 3-element center position of bbox

    Returns:
        R_bbox_new (np.ndarray): rotated bounding box rotation matrix
        center_new (np.ndarray): rotated bounding box center position
    """
    #print(bbox_rotation)
    # Step 1: inverse rotation
    R_inv = rotation_matrix.T

    roll = bbox_rotation["x"]  
    pitch = bbox_rotation["y"] 
    yaw = bbox_rotation["z"] 

    bbox_position = np.array([bbox_position["x"], bbox_position["y"], bbox_position["z"]])
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

    _,_,yaw = rotation_matrix_to_euler_zyx(R_bbox, degrees=False)

    return yaw, rotated_center



