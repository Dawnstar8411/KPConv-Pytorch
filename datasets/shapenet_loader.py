import torch
import glob
import pickle

import numpy as np
import time
import pickle
from os.path import join, exists
from os import listdir
from torch.utils.data import Dataset
from utils.helper_ply_io import read_ply
from utils.utils import bcolors
from utils.helper_data_processing import DataProcessing as DP


class ShapeNetDataset(Dataset):
    def __init__(self, args, train=True):
        self.label_to_names = {0: 'Airplane',
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
        # Init labels
        self.name_to_labels = {v: k for k, v in self.label_to_names.items()}  # 名字对应的label 键值
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # 排序好的label 键值
        self.label_names = [self.label_to_names[k] for k in self.label_values]  # 对应的label name
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # label键值不一定连续，给出连续的index
        self.ignored_labels = np.array([])
        self.num_cls = len(self.label_to_names) - len(self.ignored_labels)
        self.num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]  # 每个物体有多少个part

        self.data_path = args.data_path
        self.sub_grid_size = args.sub_grid_size
        self.train = train
        self.part_type = args.part_type
        if self.part_type == 'multi':
            self.num_train = 14007
            self.num_test = 2870
        elif self.part_type in self.label_names:
            # 后面根据具体的物体类别进行计算
            self.num_train = None
            self.num_test = None

        self.points = {'train': [], 'valid': [], 'test': []}
        self.labels = {'train': [], 'valid': [], 'test': []}
        self.point_labels = {'train': [], 'valid': [], 'test': []}

        train_filename = join(self.data_path, 'train_{:.3f}.pkl'.format(self.sub_grid_size))
        test_filename = join(self.data_path, 'test_{:.3f}.pkl'.format(self.sub_grid_size))

        if exists(train_filename):
            with open(train_filename, 'rb') as file:
                self.points['train'], self.normals['train'], self.labels['train'] = pickle.load(file)
        else:
            print('Required data dose not exist')

        if exists(test_filename):
            with open(test_filename, 'rb') as file:
                self.points['valid'], self.normals['valid'], self.labels['valid'] = pickle.load(file)
        else:
            print('Required data dose not exist')

        if self.part_type in self.label_names:
            # 关注的这一类的索引
            wanted_label = self.name_to_labels[self.part_type]

            boolean_mask = self.labels['train'] == wanted_label
            self.labels['train'] = self.labels['train'][boolean_mask]
            self.points['train'] = np.array(self.points['train'])[boolean_mask]
            self.point_labels['train'] = np.array(self.point_labels['train'])[boolean_mask]
            self.num_train = len(self.labels['train'])

            boolean_mask = self.labels['valid'] == wanted_label
            self.labels['valid'] = self.labels['valid'][boolean_mask]
            self.points['valid'] = np.array(self.points['test'])[boolean_mask]
            self.point_labels['valid'] = np.array(self.point_labels['valid'])[boolean_mask]
            self.num_train = len(self.labels['valid'])

        self.point_labels['train'] = [p_l - 1 for p_l in self.point_labels['train']]
        self.point_labels['valid'] = [p_l - 1 for p_l in self.point_labels['valid']]
