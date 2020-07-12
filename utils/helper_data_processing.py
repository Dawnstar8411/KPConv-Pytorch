import os
from os.path import join

import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling  # 网格采样
import utils.cpp_wrappers.cpp_neighbors.neighbors as cpp_neighbors  # 计算某个半径范围内的邻居点
import utils.cpp_wrappers.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors  # 计算k近邻的邻居点

import numpy as np
import pandas as pd


def load_pc_semantic3d(filename):
    pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
    pc = pc_pd.values
    return pc


def load_label_semantic3d(filename):
    label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
    cloud_labels = label_pd.values
    return cloud_labels


def load_pc_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # get xyz
    return points


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def get_file_list(dataset_path, test_scan_num):
    seq_list = np.sort(os.listdir(dataset_path))

    train_file_list = []
    test_file_list = []
    val_file_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        if seq_id == '08':
            val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            if seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif int(seq_id) >= 11 and seq_id == test_scan_num:
            test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
        elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
            train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)
    test_file_list = np.concatenate(test_file_list, axis=0)
    return train_file_list, val_file_list, test_file_list


def knn_search(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """

    neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)


def data_aug(xyz, color, labels, idx, num_out):
    num_in = len(xyz)
    dup = np.random.choice(num_in, num_out - num_in)
    xyz_dup = xyz[dup, ...]
    xyz_aug = np.concatenate([xyz, xyz_dup], 0)
    color_dup = color[dup, ...]
    color_aug = np.concatenate([color, color_dup], 0)
    idx_dup = list(range(num_in)) + list(dup)
    idx_aug = idx[idx_dup]
    label_aug = labels[idx_dup]
    return xyz_aug, color_aug, idx_aug, label_aug


def shuffle_idx(x):
    # random shuffle the index
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx]


def shuffle_list(data_list):
    indices = np.arange(np.shape(data_list)[0])
    np.random.shuffle(indices)
    data_list = data_list[indices]
    return data_list


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32 [N,3]
    :param angle: float32 [N,]
    :return: float32 [N,3,3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))


def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    """
    CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param grid_size: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: sub_sampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points, sampleDl=grid_size, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.subsample(points, features=features, sampleDl=grid_size, verbose=verbose)
    elif features is None:
        return cpp_subsampling.subsample(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    else:
        return cpp_subsampling.subsample(points, features=features, classes=labels, sampleDl=grid_size,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None, grid_size=0.1, max_p=0, verbose=0,
                           random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param grid_size: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)  # batch_size

    if random_grid_orient:
        # 为batch中的每一个点云生成一个随机旋转矩阵

        # 生成两个随机角度，并构造极坐标系中的第一个向量
        theta = np.random.rand(B) * 2 * np.pi  # 0 ~ 2 pi
        phi = (np.random.rand(B) - 0.5) * np.pi  # -1/2 pi ~ 1/2 pi
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        alpha = np.random.rand(B) * 2 * np.pi  # 0 ~ 2 pi

        # 由向量u和角度alpha构造旋转矩阵
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        # 对原始点云进行旋转操作
        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length
    # 对点云数据进行下采样操作
    if features in None and labels is None:
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=grid_size,
                                                          max_p=max_p,
                                                          verbost=verbose)
        # 下采样之后的点云也要做相应的旋转变换
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                                                     axis=1)
                i0 += length
        return s_points, s_len
    elif labels is None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=grid_size,
                                                                      max_p=max_p,
                                                                      verbost=verbose)
        # 下采样之后的点云也要做相应的旋转变换
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                                                     axis=1)
                i0 += length
        return s_points, s_len, s_features
    elif features is None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=grid_size,
                                                                    max_p=max_p,
                                                                    verbost=verbose)
        # 下采样之后的点云也要做相应的旋转变换
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                                                     axis=1)
                i0 += length
        return s_points, s_len, s_labels
    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=grid_size,
                                                                                max_p=max_p,
                                                                                verbost=verbose)
        # 下采样之后的点云也要做相应的旋转变换
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                                                     axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


def get_class_weights(dataset_name):
    # pre-calculate the number of points in each category
    num_per_class = []
    if dataset_name is 'S3DIS':
        num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                  650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    elif dataset_name is 'Semantic3D':
        num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                 dtype=np.int32)
    elif dataset_name is 'SemanticKITTI':
        num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                  240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                  9833174, 129609852, 4506626, 1168181])
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    return np.expand_dims(ce_label_weight, axis=0)


def rasterize_mesh(vertices, faces, dl, verbose=False):
    """
    通过光栅化由三角面片生成点云 All models are rescaled to fit in a 1 meter radius sphere
    :param vertices: 点数组  N1*3
    :param faces: 面片数组   N2*3
    :param dl: 控制点云密度的参数，表示两个点之间的距离
    :param verbose: 是否打印信息
    :return: 点云
    """
    ######################################
    # 消除无用的面片和点
    ######################################
    faces3d = vertices[faces, :]  # 获取每个三角面片的三个顶点，N2*3*3
    sides = np.stack([faces3d[:, i, :] - faces3d[:, i - 1, :] for i in [2, 0, 1]], axis=1)  # 任意两个顶点的坐标差
    keep_bool = np.min(np.linalg.norm(sides, axis=-1), axis=-1) > 1e-9  # 计算每个三角面片两两顶点之间的欧氏距离
    faces = faces[keep_bool]  # 筛选出符合条件的面片，太小的面片就不要了

    ##################################
    # 在每个面片上随机放置一个点
    ##################################

    faces3d = vertices[faces, :]  # 面片顶点的坐标 N2*3*3
    opposite_sides = np.stack([faces3d[:, i, :] - faces3d[:, i - 1, :] for i in [2, 0, 1]], axis=1)  # 面片中顶点之间的坐标差
    lengths = np.linalg.norm(opposite_sides, axis=-1)  # 面片中顶点之间的欧式距离  N2*3

    all_points = []  # 存储所有的点
    all_vert_inds = []  #

    for face_verts, face, l, sides in zip(faces, faces3d, lengths, opposite_sides):
        """
        face_verts: 每个面片的三个顶点的索引 (3,)
        face: 面片的三个顶点的坐标 (3,3)
        l: 面片顶点之间的欧式距离 (3,)
        sides: 面片顶点之间的坐标差 (3,3)
        """

        face_points = []  # 当前面片产生的所有点

        if np.min(l) < 1e-9:  # 两点之间距离过近，忽略掉此面片
            continue

        if np.max(l) < dl:  # 若最大边长小于dl，面片的中心作为一个新的点
            face_points.append(np.mean(face, axis=0))
            continue

        a_idx = np.argmax(l)  # 找出最大的一条边
        b_idx = (a_idx + 1) % 3  # 按顺序往下排
        c_idx = (a_idx + 2) % 3  # 按顺序往下排，总共三条边
        i = -sides[b_idx] / l[b_idx]  # (3,)
        j = sides[c_idx] / l[c_idx]  # (3,)

        # Create a mesh grid of points along the two smallest sides
        s1 = (l[b_idx] % dl) / 2  # B边上生成网格的起点
        s2 = (l[c_idx] % dl) / 2  # C边上生成网格的起点
        x, y = np.meshgrid(np.arange(s1, l[b_idx], dl), np.arange(s2, l[c_idx], dl))
        # 网格是两条短边平行的
        points = face[a_idx, :] + (np.expand_dims(x.ravel(), 1) * i + np.expand_dims(y.ravel(), 1) * j)
        points = points[x.ravel() / l[b_idx] + y.ravel() / l[c_idx] <= 1, :]  # 只保留面片里面的点
        face_points.append(points)

        # 边上的点添加进来
        for edge_idx in range(3):
            i = sides[edge_idx] / l[edge_idx]
            a_idx = (edge_idx + 1) % 3
            s1 = (l[edge_idx] % dl) / 2
            x = np.arange(s1, l[edge_idx], dl)
            points = face[a_idx, :] + np.expand_dims(x.ravel(), 1) * i
            face_points.append(points)

        face_points.append(face)  # 三个顶点也添加进来

        # 对于生成的每个点，计算其与面片三个顶点的距离  N*3， 共N个点，每行代表其与三个顶点的距离
        dists = np.sum(np.square(np.expand_dims(np.vstack(face_points), 1) - face), axis=2)
        # 对于生成的每个点，其余三个顶点中的哪个顶点距离最近，就把那个顶点的在原始点云中的索引号添加进来
        all_vert_inds.append(face_verts[np.argmin(dists, axis=1)])

        all_points += face_points  # 保存该面片生成的所有点

    return np.vstack(all_points).astype(np.float32), np.hstack(all_vert_inds)  # 最后的all_vert_inds为一个一维向量


def cylinder_mesh(cylinder, precision=24):
    # Get parameters
    center = cylinder[:3]
    h = cylinder[3]
    r = cylinder[4]

    # Create vertices
    theta = 2.0 * np.pi / precision
    thetas = np.arange(precision) * theta
    circleX = r * np.cos(thetas)
    circleY = r * np.sin(thetas)
    top_vertices = np.vstack((circleX, circleY, circleY * 0 + h / 2)).T
    bottom_vertices = np.vstack((circleX, circleY, circleY * 0 - h / 2)).T
    vertices = np.array([[0, 0, h / 2],
                         [0, 0, -h / 2]])
    vertices = np.vstack((vertices, top_vertices, bottom_vertices))
    vertices += center

    # Create faces
    top_faces = [[0, 2 + i, 2 + ((i + 1) % precision)] for i in range(precision)]
    bottom_faces = [[1, 2 + precision + i, 2 + precision + ((i + 1) % precision)] for i in range(precision)]
    side_faces1 = [[2 + i, 2 + precision + i, 2 + precision + ((i + 1) % precision)] for i in range(precision)]
    side_faces2 = [[2 + precision + ((i + 1) % precision), 2 + i, 2 + ((i + 1) % precision)] for i in
                   range(precision)]
    faces = np.array(top_faces + bottom_faces + side_faces1 + side_faces2, dtype=np.int32)

    return vertices.astype(np.float32), faces
