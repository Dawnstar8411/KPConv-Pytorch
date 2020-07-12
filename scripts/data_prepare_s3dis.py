import pickle
from os import makedirs, listdir
from os.path import join, exists, isdir, isfile

import numpy as np
from sklearn.neighbors import KDTree

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import write_ply, read_ply

subsampling_parameter = 0.04

label_to_names = {0: 'ceiling',
                  1: 'floor',
                  2: 'wall',
                  3: 'beam',
                  4: 'column',
                  5: 'window',
                  6: 'door',
                  7: 'table',
                  8: 'chair',
                  9: 'sofa',
                  10: 'bookcase',
                  11: 'board',
                  12: 'clutter'}

name_to_labels = {v: k for k, v in label_to_names.items()}  # 名字对应的label 键值

data_path = '/home/yc/chen/data/point_cloud/s3dis/'  # 点云数据根目录
data_folder = 'Stanford3dDataset_v1.2_Aligned_Version'  # 原始点云数据目录
cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']  # 点云名称

ply_path = join(data_path, 'original_ply')  # 将原始点云处理成ply文件，保存在original_ply文件夹中
if not exists(ply_path):
    makedirs(ply_path)

# 点云数据的预处理
for cloud_name in cloud_names:
    # 如果当前场景点云已存在，进行下一轮
    cloud_file = join(ply_path, cloud_name + '.ply')
    if exists(cloud_file):
        print('{:s}.ply already exists\n'.format(cloud_name))
        continue

    cloud_folder = join(data_path, data_folder, cloud_name)
    # 该area中所有房间的点云文件路径
    room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if
                    isdir(join(cloud_folder, room))]

    points = []
    colors = []
    classes = []

    # 处理每一个room的点云
    for i, room_folder in enumerate(room_folders):
        print('Cloud {} - Room {}/{}: {}'.format(cloud_name, i + 1, len(room_folders), room_folder.split('/')[-1]))
        for object_name in listdir(join(room_folder, 'Annotations')):
            if object_name[-4:] == '.txt':
                # 该room内的某个物体的路径
                object_file = join(room_folder, 'Annotations', object_name)
                tmp = object_name[:-4].split('_')[0]  # 物体的名称
                if tmp in name_to_labels:
                    object_class = name_to_labels[tmp]
                elif tmp in ['stairs']:
                    object_class = name_to_labels['clutter']  # 将stairs归为clutter类别
                else:
                    raise ValueError('Unknown object name: ' + str(tmp))
                if object_name == 'ceiling_1.txt':
                    with open(object_file, 'r') as f:
                        lines = f.readlines()
                    for l_i, line in enumerate(lines):
                        if '103.0\x100000' in line:
                            lines[l_i] = line.replace('103.0\x100000', '103.000000')
                    with open(object_file, 'w') as f:
                        f.writelines(lines)

                # 读取每个物体的点云数据
                with open(object_file, 'r') as f:
                    object_data = np.array([[float(x) for x in line.split()] for line in f])
                points.append(object_data[:, 0:3].astype(np.float32))
                colors.append(object_data[:, 3:6].astype(np.uint8))
                object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                classes.append(object_classes)
    points = np.vstack(points)
    colors = np.vstack(colors)
    classes = np.vstack(classes)

    write_ply(cloud_file,
              (points, colors, classes),
              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

# 对点云进行采样处理

tree_path = join(data_path, 'input_{:.3f}'.format(subsampling_parameter))
if not exists(tree_path):
    makedirs(tree_path)

train_files = [join(ply_path, f + '.ply') for f in cloud_names]

for i, file_path in enumerate(train_files):
    print('Processing train data {}/{}'.format(i, len(train_files)))
    cloud_name = file_path.split('/')[-1][:-4]
    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
    proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
    if isfile(KDTree_file):
        print('{:s} already exists\n'.format(cloud_name))
    else:
        data = read_ply(file_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = data['class']

        sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(points,
                                                                  features=colors,
                                                                  labels=labels,
                                                                  grid_size=subsampling_parameter)

        sub_colors = sub_colors / 255
        sub_labels = np.squeeze(sub_labels)
        search_tree = KDTree(sub_points, leaf_size=50)

        proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
        proj_inds = proj_inds.astype(np.int32)

        with open(KDTree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        write_ply(sub_ply_file,
                  [sub_points, sub_colors, sub_labels],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        with open(proj_file, 'wb') as f:
            pickle.dump([proj_inds, labels], f)
print('S3DIS dataset done!')
