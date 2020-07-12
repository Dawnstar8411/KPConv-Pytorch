# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from utils.helper_data_processing import DataProcessing as DP


# vertical rotation
def rotate_point_cloud(points, normals=None, args=None):
    rotation_matrix = np.eye(points.shape[1])

    if points.shape[1] == 3:
        if args.augment_rotation == 'vertical':
            theta = np.random.rand() * 2 * np.pi
            cosval = np.cos(theta)
            sinval = np.sin(theta)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]], dtype=np.float32)
        elif args.augment_rotation == 'all':
            theta = np.random.rand() * 2 * np.pi
            phi = (np.random.rand() - 0.5) * np.pi
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            alpha = np.random.rand() * 2 * np.pi
            rotation_matrix = DP.create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
    rotation_matrix = rotation_matrix.astype(np.float32)
    # Do not use np.dot because it is multi-threaded
    # rotated_data = np.dot(points.reshape((-1, 3), rotation_matrix))
    rotated_data = np.sum(np.expand_dims(points, 2) * rotation_matrix, axis=1)
    if normals is None:
        return rotation_matrix, rotated_data
    else:
        rotated_normals = np.dot(normals, rotation_matrix)
        return rotation_matrix, rotated_data, rotated_normals


def scale_point_cloud(points, normals=None, args=None):
    min_s = args.augment_scale_min
    max_s = args.augment_scale_max

    if args.augment_scale_anisotropic:
        s = np.random.uniform(low=min_s, high=max_s, size=3)
    else:
        s = np.random.uniform(low=min_s, high=max_s, size=1)

    symmetries = []
    for i in range(3):
        if args.augment_symmetries[i]:
            symmetries.append(np.round(np.random.uniform()) * 2 - 1)
        else:
            symmetries.append(1.0)
    scales = s * symmetries

    noise = (np.random.randn(points.shape[0], points.shape[1]) * args.augment_noise).astype(np.float32)

    scaled_points = points * scales + noise
    if normals is None:
        return scales, scaled_points
    else:
        normal_scale = scales[[1, 2, 0]] * scales[[2, 0, 1]]
        augmented_normals = normals * normal_scale
        augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)
        return scales, scaled_points, augmented_normals
