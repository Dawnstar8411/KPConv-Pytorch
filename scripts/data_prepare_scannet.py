import json
import pickle
from os import makedirs, listdir
from os.path import join, exists, isfile

import numpy as np
from sklearn.neighbors import KDTree

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import write_ply, read_ply

subsampling_parameter = 0.04

label_to_names = {0: 'unclassified',
                  1: 'wall',
                  2: 'floor',
                  3: 'cabinet',
                  4: 'bed',
                  5: 'chair',
                  6: 'sofa',
                  7: 'table',
                  8: 'door',
                  9: 'window',
                  10: 'bookshelf',
                  11: 'picture',
                  12: 'counter',
                  14: 'desk',
                  16: 'curtain',
                  24: 'refridgerator',
                  28: 'shower curtain',
                  33: 'toilet',
                  34: 'sink',
                  36: 'bathtub',
                  39: 'otherfurniture'}

label_values = np.sort([k for k, v in label_to_names.items()])  # 排序好的label 键值

data_path = '/home/yc/chen/data/point_cloud/scannet/'  # Scannet数据根目录
paths = [join(data_path, 'scans'), join(data_path, 'scans_test')]  # 原始数据的训练集与测试集路径
new_paths = [join(data_path, 'train_points'), join(data_path, 'test_points')]  # 与处理后的ply文件的路径，经过了稠密化，下采样
mesh_paths = [join(data_path, 'train_meshes'), join(data_path, 'test_meshes')]  # 与处理后的mesh文件路径，原始的点云信息

label_files = join(data_path, 'scannetv2-labels.combined.tsv')
with open(label_files, 'r') as f:
    lines = f.readlines()
    names1 = [line.split('\t')[1] for line in lines[1:]]  # 类别名称 wall, chair, floor 等
    IDs = [int(line.split('\t')[4]) for line in lines[1:]]  # 该类别在nyu数据集中的label编号
    annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}  # 每个类别名称与nyu中label编号的字典

# 依次处理训练集与测试集，测试集没有ground truth的label
for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):
    if not exists(new_path):
        makedirs(new_path)
    if not exists(mesh_path):
        makedirs(mesh_path)

    scenes = np.sort([f for f in listdir(path)])  # 场景列表： scene0000_00, scene0000_01 等
    for i, scene in enumerate(scenes):
        print('Scene {}/{}: {}'.format(i + 1, len(scenes), scene.split('/')[-1]))
        if exists(join(new_path, scene + '.ply')):
            continue

        vertex_data, faces = read_ply(join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
        vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
        vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)

        if new_path == join(data_path, 'train_points'):
            align_mat = None
            with open(join(path, scene, scene + '.txt'), 'r') as txtfile:
                lines = txtfile.readlines()
            for line in lines:
                line = line.split()
                if line[0] == 'axisAlignment':
                    align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)

            R = align_mat[:3, :3]
            T = align_mat[:3, 3]
            vertices = vertices.dot(R.T) + T  # 对点云进行校正

            with open(join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                segmentations = json.load(f)

            segIndices = np.array(segmentations['segIndices'])

            with open(join(path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                aggregation = json.load(f)

            for segGroup in aggregation['segGroups']:
                c_name = segGroup['label']
                if c_name in names1:
                    nyuID = annot_to_nyuID[c_name]
                    if nyuID in label_values:
                        for segment in segGroup['segments']:
                            vertices_labels[segIndices == segment] = nyuID
            write_ply(join(mesh_path, scene + '_mesh.ply'),
                      [vertices, vertices_colors, vertices_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                      triangular_faces=faces)
        else:
            write_ply(join(mesh_path, scene + '_mesh.ply'),
                      [vertices, vertices_colors],
                      ['x', 'y', 'z', 'red', 'green', 'blue'],
                      triangular_faces=faces)

        # 利用三角面片稠密化点云，进行插值操作，得到新的点云以及其与原始点云中的哪个点最近
        points, associated_vert_inds = DP.rasterize_mesh(vertices, faces, 0.003)

        # 进行点云采样
        sub_points, sub_vert_inds = DP.grid_sub_sampling(points, labels=associated_vert_inds, grid_size=0.01)

        # 利用原始点云的颜色值来填充新的点云
        sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

        if new_path == join(data_path, 'train_points'):

            # 利用原始点云的label值来填充新的点云
            sub_labels = vertices_labels[sub_vert_inds.ravel()]

            # 保存点云
            write_ply(join(new_path, scene + '.ply'),
                      [sub_points, sub_colors, sub_labels, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
        else:
            # 保存点云
            write_ply(join(new_path, scene + '.ply'),
                      [sub_points, sub_colors, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

tree_path = join(data_path, 'input_{:.3f}'.format(subsampling_parameter))
if not exists(tree_path):
    makedirs(tree_path)

train_path = join(data_path, 'train_points')  # 训练集，需要对其进行下采样
test_path = join(data_path, 'test_points')  # 测试集，需要对其进行下采样

train_files = np.sort([join(train_path, f) for f in listdir(train_path) if f[-4] == '.ply'])
test_files = np.sort([join(test_path, f) for f in listdir(test_path) if f[-4] == '.ply'])
files = np.hstack((train_files, test_files))  # 将所有的点云文件路径整合在一起

for i, file_path in enumerate(files):
    print('Processing point data {}/{}'.format(i, len(files)))
    cloud_name = file_path.split('/')[-1][:-4]
    cloud_folder = file_path.split('/')[-2]
    if 'test' in cloud_folder:
        cloud_split = 'test'
    else:
        cloud_split = 'train'

    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(tree_path, '{:s}.ply.pkl'.format(cloud_name))
    proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
    if isfile(KDTree_file):
        print('{:s} KDTree_file already exists\n'.format(cloud_name))
        continue
    else:
        data = read_ply(file_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        if cloud_split == 'test':
            int_features = data['vert_ind']
        else:
            int_features = np.vstack((data['vert_ind'], data['class'])).T
        sub_points, sub_colors, sub_int_features = DP.grid_sub_sampling(points,
                                                                        features=colors,
                                                                        labels=int_features,
                                                                        grid_size=subsampling_parameter)
        sub_colors = sub_colors / 255
        if cloud_split == 'test':
            sub_vert_inds = np.squeeze(sub_int_features)
            sub_labels = None
        else:
            sub_vert_inds = sub_int_features[:, 0]
            sub_labels = sub_int_features[:, 1]

        search_tree = KDTree(sub_points, leaf_size=50)

        with open(KDTree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        if cloud_split == 'test':
            write_ply(sub_ply_file,
                      [sub_points, sub_colors, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
        else:
            write_ply(sub_ply_file,
                      [sub_points, sub_colors, sub_labels, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
    if isfile(proj_file):
        print('{:s} proj_file already exists\n'.format(cloud_name))
    else:
        mesh_path = file_path.split('/')
        mesh_path[-2] = cloud_split + '_meshes'
        mesh_path = '/'.join(mesh_path)
        vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
        vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        if cloud_split == 'train':
            labels = vertex_data['class']
        else:
            labels = np.zeros(vertices.shape[0], dtype=np.int32)
        proj_inds = np.squeeze(search_tree.query(vertices, return_distance=False))
        proj_inds = proj_inds.astype(np.int32)
        with open(proj_file, 'wb') as f:
            pickle.dump([proj_inds, labels], f)
print('Scannet dataset done!')
