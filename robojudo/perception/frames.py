from __future__ import annotations

import math

import numpy as np


G1_PELVIS_TO_WAIST_ROLL_OFFSET = np.array([-0.0039635, 0.0, 0.044], dtype=np.float32)


def quat_wxyz_to_matrix(quat: tuple[float, float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def get_joint_angle_by_name(joint_pos: np.ndarray, joint_names: list[str], joint_name: str) -> float:
    return float(joint_pos[joint_names.index(joint_name)])


def g1_torso_point_to_pelvis(point_torso: np.ndarray, joint_pos: np.ndarray, joint_names: list[str]) -> np.ndarray:
    yaw = get_joint_angle_by_name(joint_pos, joint_names, "waist_yaw_joint")
    roll = get_joint_angle_by_name(joint_pos, joint_names, "waist_roll_joint")
    pitch = get_joint_angle_by_name(joint_pos, joint_names, "waist_pitch_joint")
    r_yaw = rotation_z(yaw)
    rotation = r_yaw @ rotation_x(roll) @ rotation_y(pitch)
    translation = r_yaw @ G1_PELVIS_TO_WAIST_ROLL_OFFSET
    return (translation + rotation @ np.asarray(point_torso, dtype=np.float32)).astype(np.float32)


def default_g1_realsense_depth_link_transform() -> tuple[np.ndarray, np.ndarray]:
    translation = np.array(
        [
            0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
            0.015,
            0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
        ],
        dtype=np.float32,
    )
    rotation_wxyz = np.array(
        [
            math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
            math.sin(math.radians(0.5) / 2),
            math.sin(math.radians(48) / 2),
            0.0,
        ],
        dtype=np.float32,
    )
    return translation, quat_wxyz_to_matrix(rotation_wxyz)
