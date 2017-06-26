"""Functions for coordinate transformations."""
import numpy as np


def inertial_to_body_frame(points, translation_vec, rotations):
    points = np.array(points)
    translation_vec = np.array(translation_vec)
    rotations = np.array(rotations)

    rot_mat_acc = np.eye(2)
    for phi in np.nditer(rotations):
        rot_mat_acc = _rotation_matrix(phi) @ rot_mat_acc
    return rot_mat_acc @ points + translation_vec


def body_to_inertial_frame(points, translation_vec, rotations):
    points = np.array(points)
    translation_vec = np.array(translation_vec)
    rotations = -np.array(rotations)

    rot_mat_acc = np.eye(2)
    for phi in np.nditer(rotations):
        rot_mat_acc = _rotation_matrix(phi) @ rot_mat_acc
    return rot_mat_acc @ (points - translation_vec)


def _rotation_matrix(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])
