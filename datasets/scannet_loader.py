import glob
from os.path import exists,join,isfile
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import bcolors
from utils.helper_ply_io import read_ply
from utils.helper_data_processing import DataProcessing as DP


class ScanNetDataset(Dataset):
    def __init__(self, args, train=True):
        self.data_path = args.data_path
        self.sub_grid_size = args.sub_grid_size
        self.train = train
        self.split = 'train' if train else 'valid'

        self.label_to_names = {0: 'unclassified',
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

        self.name_to_labels = {v: k for k, v in self.label_to_names.items()}  # 名字对应的label 键值
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # 排序好的label 键值
        self.label_names = [self.label_to_names[k] for k in self.label_values]  # 对应的label name
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # label键值不一定连续，给出连续的index
        self.ignored_labels = np.array([0])
        self.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

        self.tree_path = join(self.data_path,'input_{.3f}'.format(self.sub_grid_size))

        self.trees = {'train': [], 'valid': [],'test':[]}
        self.colors = {'train': [], 'valid': [],'test':[]}
        self.labels = {'train': [], 'valid': []}
        self.vert_inds = {'train': [], 'valid': []}
        self.val_proj = []
        self.val_proj_labels = []
        self.test_proj = []
        self.val_proj_labels = []

        self.train_path = 'utils/meta/scannetv2_train.txt'
        self.val_path = 'utils/meta/scannetv2_val.txt'

        self.all_files = glob.glob(join(self.data_path, 'train_points', '*.ply'))
        # .../scanet/input_0.040_train

        self.train_files = [line.rstrip() for line in open(self.train_path)]
        self.val_files = [line.rstrip() for line in open(self.val_path)]

        for i, file_path in enumerate(self.all_files):
            cloud_name = file_path.split('/')[-1][:-4]  # ignore '.ply'
            if cloud_name in self.val_files:
                cloud_split = 'valid'
            elif cloud_name in self.train_files:
                cloud_split = 'train'
            else:
                cloud_split = 'test'

            KDTree_file = join(self.tree_path,'{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(self.tree_path,'{:s}.ply'.format(cloud_name))
            if isfile(KDTree_file):
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T  # N * 3
                sub_vert_inds = data['vert_ind']
                if cloud_split =='test':
                    sub_labels = None
                else:
                    sub_labels = data['class']
                    #sub_labels = np.array([self.label_to_idx[label] for label in sub_labels])
                with open(KDTree_file,'rb') as f:
                    search_tree = pickle.load(f)
            else:
                print('No File')

            self.trees[cloud_split]+=[search_tree]
            self.colors[cloud_split]+=[sub_colors]
            self.vert_inds[cloud_split]+=[sub_vert_inds]
            if cloud_split in ['train','valid']:
                self.input_labels[cloud_split] += [sub_labels]


            if cloud_split == 'valid':
                proj_file = join(self.tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    print('No File')
                self.val_proj+=[proj_inds]
                self.val_proj_labels+=[labels]
            if cloud_split == 'test':
                proj_file = join(self.tree_path,'{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    proj_inds,labels = pickle.load(f)
                else:
                    print('No file')
                self.test_proj += [proj_inds]
                self.test_labels +=[labels]

    def __getitem__(self, index):

        # Choose the cloud with the lowest probability
        # 根据min_possibility的值选一个点云出来
        cloud_idx = int(np.argmin(self.min_possibility[self.split]))

        # Choose the point with the minimum of possibility in the cloud as query point
        # 对于找出来的点云的所有点，根据possibility的值找出一个点来
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])

        # Get all points within the cloud from tree structure
        # KDTree中保存了点云的 x,y,z值， N*3
        points = np.array(self.trees[self.split][cloud_idx].data, copy=False)

        # Center point of input region
        # 提取出找到的种子点的坐标
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=args.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < args.n_pts:
            # Query all points within the cloud   shape:(2,1,k) distance,index
            queried_idx = self.trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.trees[self.split][cloud_idx].query(pick_point, k=args.n_pts)[1][0]

        # shuffle index
        idx = np.arange(len(queried_idx))
        np.random.shuffle(idx)
        queried_idx = queried_idx[idx]

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.colors[self.split][cloud_idx][queried_idx]
        queried_pc_labels = self.labels[self.split][cloud_idx][queried_idx]

        # 到此为止，先选了一个点云，然后在里面选了一个点，以这个点为中心，找到了与其最近的k个点，打乱顺序并进行中心化，得到 x y z red green blue label

        # Update the possibility of the selected points
        # 按照与选取的种子点的距离远近，来增加其possibility值，距离越近，加的越多
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.split][cloud_idx][queried_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        # up_sampled with replacement
        if len(points) < args.n_pts:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = DP.data_aug(queried_pc_xyz,
                                                                                            queried_pc_colors,
                                                                                            queried_pc_labels,
                                                                                            queried_idx, args.n_pts)
        features = np.concatenate([queried_pc_xyz, queried_pc_colors], axis=-1)
        # print(np.shape(features))
        input_points = None
        input_neighbors = None
        input_pools = None
        input_up_samples = None

        for i in range(args.num_layers):
            # (1,N1,3)
            batch_queried_pc_xyz = queried_pc_xyz[np.newaxis, :]
            # (1,N1,k)
            batch_neighbor_idx = DP.knn_search(batch_queried_pc_xyz, batch_queried_pc_xyz, args.num_knn)
            # 对输入的点云，求每个点云的k近邻点的索引 (N1,K)
            neighbour_idx = np.squeeze(batch_neighbor_idx, axis=0)

            # 对点云及其k近邻点索引进行下采样，因为前面将点云数据打乱了，所以直接去减 1/n的点等价于随机下采样
            sub_points = queried_pc_xyz[:np.shape(queried_pc_xyz)[0] // args.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:np.shape(queried_pc_xyz)[0] // args.sub_sampling_ratio[i], :]
            # 寻找初始点云在下采样点云中的最近邻点的索引
            batch_sub_points = sub_points[np.newaxis, :]
            batch_up_i = DP.knn_search(batch_sub_points, batch_queried_pc_xyz, 1)
            up_i = np.squeeze(batch_up_i, axis=0)

            if input_points is None:
                input_points = queried_pc_xyz
            else:
                input_points = np.concatenate((input_points, queried_pc_xyz), axis=0)

            if input_neighbors is None:
                input_neighbors = neighbour_idx
            else:
                input_neighbors = np.concatenate((input_neighbors, neighbour_idx), axis=0)

            if input_pools is None:
                input_pools = pool_i
            else:
                input_pools = np.concatenate((input_pools, pool_i), axis=0)

            if input_up_samples is None:
                input_up_samples = up_i
            else:
                input_up_samples = np.concatenate((input_up_samples, up_i), axis=0)

            queried_pc_xyz = sub_points

        input_points = input_points.transpose((1, 0))  # 3, N

        return torch.from_numpy(input_points), torch.from_numpy(input_neighbors), torch.from_numpy(
            input_pools), torch.from_numpy(input_up_samples), torch.from_numpy(features), torch.from_numpy(
            queried_pc_labels)

    def __len__(self):
        if self.train:
            return len(self.trees['train']) * 10
        else:
            return len(self.trees['valid']) * 10
