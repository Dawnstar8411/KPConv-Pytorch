import numpy as np
from torch.utils.data import Dataset

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import read_ply


class CommonDataset(Dataset):
    def __init__(self):
        self.label_to_names = {}
        self.name_to_labels = {v: k for k, v in self.label_to_names.items()}  # 名字对应的label 键值
        self.num_cls = 0
        self.label_values = np.zeros((0,), dtype=np.int32)
        self.label_names = []
        self.label_to_idx = {}
        self.name_to_label = {}
        self.neighborhood_limits = []
        self.ignored_labels = np.array([])
        return

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return 0

    def init_labels(self):
        self.num_cls = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # 排序好的label 键值
        self.label_names = [self.label_to_names[k] for k in self.label_values]  # 对应的label name
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # label键值不一定连续，给出连续的index
        self.name_to_labels = {v: k for k, v in self.label_to_names.item()}

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def load_evaluation_points(self, file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z']))

    def classification_inputs(self,
                              stacked_points,  # 数据集中所有的点，顺序排列
                              stacked_features,  # 数据集中所有点的feature
                              labels,  # 数据集中所有点云的label
                              stack_lengths,
                              args):  # 每个点云的点的个数
        r_normal = self.args.sub_grid_size * self.args.conv_radius  # 初始卷积的半径范围，米为单位，与点云坐标一致

        layer_blocks = []
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        arch = self.args.architecture
        #self.num_layers = len([block for block in arch if 'pool' in block or 'strided' in block]) + 1
        for block_i, block in enumerate(arch):
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue
            deform_layer = False

            if layer_blocks:
                # 此分支处理卷积操作的数据
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.args.deform_redius / self.args.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                # 每个点在半径r内的neigbors点的索引
                conv_i = DP.batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)
            else:
                # 这一层负责下采样
                conv_i = np.zeros((0, 1), dtype=np.int32)

            if 'pool' in block or 'strided' in block:
                dl = 2 * r_normal / self.args.conv_radius  # 新的采样grid_size 是上一次的两倍
                # 下采样之后的点
                pool_p, pool_b = DP.batch_grid_subsampling(stacked_points, stack_lengths, grid_size=dl)

                if 'deformable' in block:
                    r = r_normal * self.args.deform_redius / self.args.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                # 下采样之后的点，在原始点云中以半径r找neighbors点
                pool_i = DP.batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)
            else:
                # 没有下采样
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
            # 因为返回的neighbors点是按照距离由近到元排列的，可根据点的数量要求删除距离较远的点
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # 更新下一层新的点
            stacked_points = pool_p
            stack_lengths = pool_b

            # 更新半径和网络模块
            r_normal *= 2
            layer_blocks = []

            # 当遇到global pooling或者 upsampling时停止
            if 'global' in block or 'upsample' in block:
                break

        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li

    def segmentation_inputs(self,
                            stacked_points,
                            stacked_features,
                            labels,
                            stack_lengths):
        # 最开始的卷积半径
        r_normal = self.args.first_subsampling_dl * self.args.conv_radius

        layer_blocks = []

        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        arch = self.args.architecture

        for block_i, block in enumerate(arch):

            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            deform_layer = False
            if layer_blocks:
                if layer_blocks:
                    if np.any(['deformable' in blck for blck in layer_blocks]):
                        r = r_normal * self.args.deform_radius / self.args.conv_radius
                        deform_layer = True
                    else:
                        r = r_normal
                    conv_i = DP.batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

                else:
                    conv_i = np.zeros((0, 1), dtype=np.int32)

                if 'pool' in block or 'strided' in block:
                    dl = 2 * r_normal / self.args.conv_radius

                    pool_p, pool_b = DP.batch_grid_subsampling(stacked_points, stack_lengths, grid_size=dl)

                    if 'deformable' in block:
                        r = r_normal * self.args.deform_radius / self.args.conv_radius
                        deform_layer = True
                    else:
                        r = r_normal

                    pool_i = DP.batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)
                else:
                    pool_i = np.zeros((0, 1), dtype=np.int32)
                    pool_p = np.zeros((0, 3), dtype=np.float32)
                    pool_b = np.zeros((0,), dtype=np.int32)
                    up_i = np.zeros((0, 1), dtype=np.int32)

                conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
                pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
                if up_i.shape[0] > 0:
                    up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

                input_points += [stacked_points]
                input_neighbors += [conv_i.astype(np.int64)]
                input_pools += [pool_i.astype(np.int64)]
                input_upsamples += [up_i.astype(np.int64)]
                input_stack_lengths += [stack_lengths]
                deform_layers += [deform_layer]

                stacked_points = pool_p
                stack_lengths = pool_b

                r_normal *= 2
                layer_blocks = []

                if 'global' in block or 'upsample' in block:
                    break

            li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
            li += [stacked_features, labels]

            return li
