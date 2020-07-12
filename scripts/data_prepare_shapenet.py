import json
import pickle
from os import listdir, makedirs
from os.path import join, exists

import numpy as np

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import read_ply, write_ply

subsampling_parameter = 0.02

label_to_names = {0: 'Airplane',
                  1: 'Bag',
                  2: 'Cap',
                  3: 'Car',
                  4: 'Chair',
                  5: 'Earphone',
                  6: 'Guitar',
                  7: 'Knife',
                  8: 'Lamp',
                  9: 'Laptop',
                  10: 'Motorbike',
                  11: 'Mug',
                  12: 'Pistol',
                  13: 'Rocket',
                  14: 'Skateboard',
                  15: 'Table'}

name_to_labels = {v: k for k, v in label_to_names.items()}  # 名字对应的label 键值
label_values = np.sort([k for k, v in label_to_names.items()])  # 排序好的label 键值
label_names = [label_to_names[k] for k in label_values]  # 对应的label name

data_path = '/home/yc/chen/data/point_cloud/shapenet/'  # 点云数据跟目录
data_folder = 'shapenetcore_partanno_segmentation_benchmark_v0'  # 原始点云目录

# 对Shapenet数据库的ply文件做预处理
category_and_synsetoffset = [['Airplane', '02691156'],
                             ['Bag', '02773838'],
                             ['Cap', '02954340'],
                             ['Car', '02958343'],
                             ['Chair', '03001627'],
                             ['Earphone', '03261776'],
                             ['Guitar', '03467517'],
                             ['Knife', '03624134'],
                             ['Lamp', '03636649'],
                             ['Laptop', '03642806'],
                             ['Motorbike', '03790512'],
                             ['Mug', '03797390'],
                             ['Pistol', '03948459'],
                             ['Rocket', '04099429'],
                             ['Skateboard', '04225987'],
                             ['Table', '04379243']]

synsetoffset_to_category = {s: n for n, s in category_and_synsetoffset}

# 训练集文件列表
split_file = join(data_path, data_folder, 'train_test_split', 'shuffled_train_file_list.json')
with open(split_file, 'r') as f:
    train_files = json.load(f)
train_files = [name[11:] for name in train_files]

# 交叉验证集文件列表
split_file = join(data_path, data_folder, 'train_test_split', 'shuffled_val_file_list.json')
with open(split_file, 'r') as f:
    val_files = json.load(f)
val_files = [name[11:] for name in val_files]

# 测试集文件列表
split_file = join(data_path, data_folder, 'train_test_split', 'shuffled_test_file_list.json')
with open(split_file, 'r') as f:
    test_files = json.load(f)
test_files = [name[11:] for name in test_files]

# 将数据转化成ply数据
splits = ['train', 'valid', 'test']
split_files = [train_files, val_files, test_files]

# 因此处理训练集，交叉验证集，测试集
for split, files in zip(splits, split_files):
    ply_path = join(data_path, data_folder, '{:s}_ply'.format(split))  # 训练集，交叉验证集，测试集各保存在一个文件夹中
    if not exists(ply_path):
        makedirs(ply_path)

    class_nums = {n: 0 for n, s in category_and_synsetoffset}  # 每一类点云的数量
    for i, file in enumerate(files):
        synset = file.split('/')[0]  # 类别文件夹的数字标识
        class_name = synsetoffset_to_category[synset]  # 由数字标识得到label名称

        # 将每个物体的点云保存成单独的一个文件
        ply_name = join(ply_path, '{:s}_{:04d}.ply'.format(class_name, class_nums[class_name]))
        if exists(ply_name):
            class_nums[class_name] += 1
            continue

        file_name = file.split('/')[1]  # 得到文件名称

        # 在文本文件中读入points和labels
        points = np.loadtxt(join(data_path, data_folder, synset, 'points', file_name + '.pts')).astype(np.float32)
        labels = np.loadtxt(join(data_path, data_folder, synset, 'points_label', file_name + '.seg')).astype(np.int32)

        # 中心化与归一化点云
        pmin = np.min(points, axis=0)
        pmax = np.max(points, axis=0)
        points -= (pmin + pmax) / 2
        scale = np.max(np.linalg.norm(points, axis=1))
        points *= 1.0 / scale

        # 调整点云坐标为 x,y,z顺序
        points = points[:, [0, 2, 1]]

        # 保存成ply文件
        write_ply(ply_name,
                  (points, labels),
                  ['x', 'y', 'z', 'label'])

        # 该类点云数量加1
        class_nums[class_name] += 1

        print('Preparing {:s} ply : {:.1f}%'.format(split, 100 * i / len(files)))

input_points = {'train': [], 'valid': [], 'test': []}
input_labels = {'train': [], 'valid': [], 'test': []}
input_point_labels = {'train': [], 'valid': [], 'test': []}

print('Prepare train points')
if subsampling_parameter > 0:
    filename = join(data_path, 'train_{:.3f}.pkl'.format(subsampling_parameter))  # 所有训练集数据放在一个文件中
else:
    filename = join(data_path, 'train_original.pkl'.format(subsampling_parameter))  # 所有训练集数据放在一个文件中
# 训练集
if exists(filename):
    print('{:s} already exists\n'.format(filename))
else:
    split_path = join(data_path, data_folder, 'train_ply')
    names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']  # 获取点云的文件名
    names = np.sort(names)  # 对点云文件按名称进行排序

    for i, cloud_name in enumerate(names):
        print('Processing train data {}/{}'.format(i, len(names)))
        data = read_ply(join(split_path, cloud_name + '.ply'))
        points = np.vstack((data['x'], data['y'], data['z'])).T
        point_labels = data['label']

        if subsampling_parameter > 0:
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=point_labels, grid_size=subsampling_parameter)
            input_points['train'] += [sub_points]
            input_point_labels['train'] += [sub_labels]
        else:
            input_points['train'] += [points]
            input_point_labels['train'] += [point_labels]

    # 获得点云级别的label
    label_names = ['_'.join(n.split('_')[:-1]) for n in names]  # 获得点云的类别列表
    input_labels['train'] = np.array([name_to_labels[name] for name in label_names])  # 由类别名称得到类别的编号

    split_path = join(data_path, data_folder, 'valid_ply')
    names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']  # 获取点云的文件名
    names = np.sort(names)

    for i, cloud_name in enumerate(names):
        print('Processing valid data {}/{}'.format(i, len(names)))
        data = read_ply(join(split_path, cloud_name + '.ply'))
        points = np.vstack((data['x'], data['y'], data['z'])).T
        point_labels = data['label']
        if subsampling_parameter > 0:
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=point_labels, grid_size=subsampling_parameter)
            input_points['train'] += [sub_points]
            input_point_labels['train'] += [sub_labels]
        else:
            input_points['train'] += [points]
            input_point_labels['train'] += [point_labels]

    # 获得点云级别的label
    label_names = ['_'.join(n.split('_')[:-1]) for n in names]
    input_labels['train'] = np.hstack(
        (input_labels['train'], np.array([name_to_labels[name] for name in label_names])))  # 由类别名称得到类别的编号

    with open(filename, 'wb') as file:
        pickle.dump((input_points['train'],
                     input_labels['train'],
                     input_point_labels['train']), file)

print('Train dataset done!\n')
print('Prepare test points\n')
if subsampling_parameter > 0:
    filename = join(data_path, 'test_{:.3f}.pkl'.format(subsampling_parameter))
else:
    filename = join(data_path, 'test_original.pkl')

# 训练集
if exists(filename):
    print('{:s} already exists\n'.format(filename))
else:
    split_path = join(data_path, data_folder, 'test_ply')
    names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']  # 获取点云的文件名
    names = np.sort(names)

    for i, cloud_name in enumerate(names):
        print('Processing valid data {}/{}'.format(i, len(names)))
        data = read_ply(join(split_path, cloud_name + '.ply'))
        points = np.vstack((data['x'], data['y'], data['z'])).T
        point_labels = data['label']
        if subsampling_parameter > 0:
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=point_labels, grid_size=subsampling_parameter)
            input_points['test'] += [sub_points]
            input_point_labels['test'] += [sub_labels]
        else:
            input_points['test'] += [points]
            input_point_labels['test'] += [point_labels]

    # 获得点云级别的label
    label_names = ['_'.join(n.split('_')[:-1]) for n in names]
    input_labels['test'] = np.array([name_to_labels[name] for name in label_names])  # 由类别名称得到类别的编号

    with open(filename, 'wb') as file:
        pickle.dump((input_points['test'],
                     input_labels['test'],
                     input_point_labels['test']), file)
print('Test dataset done!')