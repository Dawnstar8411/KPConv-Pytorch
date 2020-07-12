import pickle
from os.path import exists

from utils.helper_data_processing import *

subsampling_parameter = 0.02

label_to_names = {0: 'airplane',
                  1: 'bathtub',
                  2: 'bed',
                  3: 'bench',
                  4: 'bookshelf',
                  5: 'bottle',
                  6: 'bowl',
                  7: 'car',
                  8: 'chair',
                  9: 'cone',
                  10: 'cup',
                  11: 'curtain',
                  12: 'desk',
                  13: 'door',
                  14: 'dresser',
                  15: 'flower_pot',
                  16: 'glass_box',
                  17: 'guitar',
                  18: 'keyboard',
                  19: 'lamp',
                  20: 'laptop',
                  21: 'mantel',
                  22: 'monitor',
                  23: 'night_stand',
                  24: 'person',
                  25: 'piano',
                  26: 'plant',
                  27: 'radio',
                  28: 'range_hood',
                  29: 'sink',
                  30: 'sofa',
                  31: 'stairs',
                  32: 'stool',
                  33: 'table',
                  34: 'tent',
                  35: 'toilet',
                  36: 'tv_stand',
                  37: 'vase',
                  38: 'wardrobe',
                  39: 'xbox'}

name_to_labels = {v: k for k, v in label_to_names.items()}  # 名字对应的label 键值
label_values = np.sort([k for k, v in label_to_names.items()])  # 排序好的label 键值
label_names = [label_to_names[k] for k in label_values]  # 对应的label name

data_path = '/home/yc/chen/data/point_cloud/modelnet40/'  # 数据根目录
data_folder = 'modelnet40_normal_resampled'  # 原始形式数据目录

input_points = {'train': [], 'valid': []}
input_normals = {'train': [], 'valid': []}
input_labels = {'train': [], 'valid': []}

# 处理训练集数据
if subsampling_parameter > 0:
    filename = join(data_path, 'train_{:.3f}.pkl'.format(subsampling_parameter))  # 将所有的训练集数据保存到一个文件里
else:
    filename = join(data_path, 'train_original.pkl')

if exists(filename):
    print('{:s} already exists\n'.format(filename))
else:
    # 训练集点云的名字列表  airplane_0001
    names = np.loadtxt(join(data_path, data_folder, 'modelnet40_train.txt'), dtype=np.str)

    for i, cloud_name in enumerate(names):
        print('Processing train data {}/{}'.format(i, len(names)))
        class_folder = '_'.join(cloud_name.split('_')[:-1])  # 由 airplane_0001.txt 变为 airplane
        txt_file = join(data_path, data_folder, class_folder, cloud_name) + '.txt'  # 点云路径
        data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)  # 读取保存在txt文件中的点云数据

        # 对原始点云进行下采样
        if subsampling_parameter > 0:
            points, normals = grid_sub_sampling(data[:, :3],
                                                features=data[:, 3:],
                                                grid_size=subsampling_parameter)
        else:
            points = data[:, :3]
            normals = data[:, 3:]

        input_points['train'] += [points]
        input_normals['train'] += [normals]

    # 获取label
    label_names = ['_'.join(name.split('_')[:-1]) for name in names]
    input_labels['train'] = np.array([name_to_labels[name] for name in label_names])

    with open(filename, 'wb') as file:
        pickle.dump((input_points['train'],
                     input_normals['train'],
                     input_labels['train']), file)

    print('Training dataset done!')

# 处理测试集数据
if subsampling_parameter > 0:
    filename = join(data_path, 'test_{:.3f}.pkl'.format(subsampling_parameter))
else:
    filename = join(data_path, 'test_original.pkl')

if exists(filename):
    print('{:s} already exists\n'.format(filename))
else:
    names = np.loadtxt(join(data_path, data_folder, 'modelnet40_test.txt'), dtype=np.str)

    for i, cloud_name in enumerate(names):
        print('Processing test data {}/{}'.format(i, len(names)))
        class_folder = '_'.join(cloud_name.split('_')[:-1])
        txt_file = join(data_path, data_folder, class_folder, cloud_name) + '.txt'
        data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

        if subsampling_parameter > 0:
            points, normals = grid_sub_sampling(data[:, :3],
                                                features=data[:, 3:],
                                                grid_size=subsampling_parameter)
        else:
            points = data[:, :3]
            normals = data[:, 3:]

        input_points['valid'] += [points]
        input_normals['valid'] += [normals]

    label_names = ['_'.join(name.split('_')[:-1]) for name in names]
    input_labels['valid'] = np.array([name_to_labels[name] for name in label_names])

    with open(filename, 'wb') as file:
        pickle.dump((input_points['valid'],
                     input_normals['valid'],
                     input_labels['valid']), file)
    print('Training dataset done!')
