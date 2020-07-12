import pickle
from os import makedirs, listdir
from os.path import join, exists, isfile

import numpy as np
from sklearn.neighbors import KDTree

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import write_ply, read_ply

subsampling_parameter = 0.06

label_to_names = {0: 'unlabeled',
                  1: 'man-made terrain',
                  2: 'natural terrain',
                  3: 'high vegetation',
                  4: 'low vegetation',
                  5: 'buildings',
                  6: 'hard scape',
                  7: 'scanning artefacts',
                  8: 'cars'}

data_path = '/home/yc/chen/data/point_cloud/semantic3d/'  # Semantic3d 数据根目录
data_folder = 'original_data'                             # 保存原始数据的目录
train_path = join(data_path, 'ply_subsampled/train')      # 训练集ply文件保存目录
test_path = join(data_path, 'ply_full/reduced-8')         # 测试集ply文件保存目录

if not exists(train_path):
    makedirs(train_path)
if not exists(test_path):
    makedirs(test_path)

old_folder = join(data_path, data_folder)
cloud_names = [file_name[:-4] for file_name in listdir(old_folder) if file_name[-4:] == '.txt']
# 对每个点云文件进行处理
for cloud_name in cloud_names:
    txt_file = join(old_folder, cloud_name + '.txt')
    label_file = join(old_folder, cloud_name + '.labels')

    if exists(label_file):
        ply_file_full = join(train_path, cloud_name + '.ply')
    else:
        ply_file_full = join(test_path, cloud_name + '.ply')

    if exists(ply_file_full):
        print('{:s} already exists\n'.format(cloud_name))
        continue
    print('Preparation of {:s}'.format(cloud_name))
    data = np.loadtxt(txt_file)
    points = data[:, :3].astype(np.float32)
    colors = data[:, 4:7].astype(np.uint8)

    if exists(label_file):
        labels = np.loadtxt(label_file, dtype=np.int32)
        sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(points,
                                                                  features=colors,
                                                                  labels=labels,
                                                                  grid_size=0.01)
        write_ply(ply_file_full,
                  (sub_points, sub_colors, sub_labels),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    else:
        write_ply(ply_file_full,
                  (points, colors),
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
# 训练集与测试集的ply文件
train_files = np.sort([join(train_path, f) for f in listdir(train_path) if f[-4:] == '.ply'])
test_files = np.sort([join(test_path, f) for f in listdir(test_path) if f[-4:] == '.ply'])

tree_path = join(data_path, 'input_{:.3f}'.format(subsampling_parameter))
if not exists(tree_path):
    makedirs(tree_path)

files = np.hstack((train_files, test_files))
print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

valid_proj = []
valid_labels = []
test_proj = []
test_labels = []

for i, file_path in enumerate(files):
    print('Processing semantic3d data {}/{}'.format(i, len(train_files)))
    cloud_name = file_path.split('/')[-1][:-4]
    cloud_folder = file_path.split('/')[-2]
    if 'train' in cloud_folder:
        cloud_split = 'train'
    else:
        cloud_split = 'test'

    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

    if isfile(KDTree_file):
        print('{:s} already exists\n'.format(cloud_name))
        continue
    else:
        data = read_ply(file_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        if cloud_split == 'test':
            int_features = None
        else:
            int_features = data['class']

        sub_data = DP.grid_sub_sampling(points,
                                        features=colors,
                                        labels=int_features,
                                        grid_size=subsampling_parameter)
        sub_colors = sub_data[1] / 255

        search_tree = KDTree(sub_data[0], leaf_size=50)

        with open(KDTree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        if cloud_split == 'test':
            sub_labels = None
            write_ply(sub_ply_file,
                      [sub_data[0], sub_colors],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])
        else:
            sub_labels = np.squeeze(sub_data[2])
            write_ply(sub_ply_file,
                      [sub_data[0], sub_colors, sub_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    if 'train' in cloud_folder:
        proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
        if isfile(proj_file):
            print('{:s} already exists\n'.format(cloud_name))
        else:
            data = read_ply(file_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            labels = data['class']

            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)

            with open(proj_file, 'wb') as f:
                pickle.dump([proj_inds, labels], f)
    if '-8' in cloud_folder:
        proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
        if isfile(proj_file):
            print('{:s} already exists\n'.format(cloud_name))
        else:
            full_ply_path = file_path.split('/')
            print(full_ply_path)
            full_ply_path[-3] = 'ply_full'
            full_ply_path = '/'.join(full_ply_path)
            data = read_ply(full_ply_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            labels = np.zeros(points.shape[0], dtype=np.int32)

            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            with open(proj_file, 'wb') as f:
                pickle.dump([proj_inds, labels], f)
