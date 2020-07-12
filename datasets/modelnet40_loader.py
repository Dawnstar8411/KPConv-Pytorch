# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import time

from torch.utils.data import Sampler

from config.args_train import *
from datasets.common import CommonDataset
from utils.helper_custom_transform import *
from utils.helper_visualization import *
from utils.utils import BColors


class ModelNet40Dataset(CommonDataset):
    def __init__(self, config, train=True, orient_correction=True):
        """
        ModelNet40数据集体量较小，选择一次性将全部数据加载到内存
        :param config: 所有的参数
        :param train:  train模式还是valid模式
        :param orient_correction: 更改点云x,y,z顺序到正确的朝向
        """
        CommonDataset.__init__(self)
        self.args = config
        self.architecture = args.architecture
        self.label_to_names = {0: 'airplane',
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
        # 初始化label各项数据
        self.init_labels()
        # 训练过程中需要忽视的类
        self.ignored_labels = np.array([])

        self.data_path = args.data_path
        self.sub_grid_size = args.sub_grid_size
        self.train = train
        self.split = 'train' if self.train else 'test'
        if self.train:
            self.num_models = 9843
            if args.train_steps and args.train_steps * args.batch_size < self.num_models:
                self.epoch_n = args.train_steps * args.batch_size
            else:
                self.epoch_n = self.num_models
        else:
            self.num_models = 2468
            self.epoch_n = min(self.num_models, args.valid_steps * args.batch_size)

        self.points = []
        self.normals = []
        self.labels = []

        filename = join(self.data_path, '{:s}_{:.3f}.pkl'.format(self.split, self.sub_grid_size))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.points, self.normals, self.labels = pickle.load(file)
        else:
            raise ValueError('{:s} data dose not exist'.format(self.split))

        if orient_correction:
            self.points = [pp[:, [0, 2, 1]] for pp in self.points]
            self.normals = [nn[:, [0, 2, 1]] for nn in self.normals]

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        tp_list = []  # 保存points坐标
        tn_list = []  # 保存normals
        tl_list = []  # 保存label
        ti_list = []  # 保存点云编号
        s_list = []
        R_list = []

        for p_i in idx_list:
            # 读取一个点云的坐标，normal和label
            points = self.points[p_i].astype(np.float32)
            normals = self.normals[p_i].astype(np.float32)
            label = self.label_to_idx[self.labels[p_i]]

            rotation_matrix, rotated_points, rotated_normals = rotate_point_cloud(points, normals=normals, args=args)
            scales, scaled_points, scaled_normals = scale_point_cloud(rotated_points, normals=rotated_normals,
                                                                      args=args)

            tp_list += [scaled_points]
            tn_list += [scaled_normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scales]
            R_list += [rotation_matrix]

        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)  # 每个点云的编号
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)  # 每个点云的点的个数
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)  # 输入特征，最简单的形式为所有点都是1

        if args.in_features_dim == 1:
            pass
        elif args.in_features_dim == 4:
            stacked_features = np.concatenate((stacked_features, stacked_normals), axis=1)
        elif args.in_features_dim == 7:
            stacked_features = np.concatenate((stacked_features, stacked_points, stacked_normals), axis=1)
        else:
            raise ValueError('Only accepted input dimensions are 1,4 and 7(without and with XYZ)')

        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        input_list += [scales, rots, model_inds]

    def __len__(self):
        return self.num_models


class ModelNet40Sampler(Sampler):
    def __init__(self, dataset: ModelNet40Dataset, use_potential=True, balance_labels=False):
        Sampler.__init__(self, dataset)
        self.use_potential = use_potential
        self.balance_labels = balance_labels  # 采样时是否平衡各类别

        self.dataset = dataset  # 需要采样处理的dataset,内存中没有复制

        if self.use_potential:
            # 大小是数据集点云的数目
            self.potentials = np.random.rand(len(dataset.labels)) * 0.1 + 0.1
        else:
            self.potentials = None
        # 每个batch所能包含的最多的点数
        self.batch_limit = 10000

        return

    def __iter__(self):
        if self.use_potential:
            if self.balance_labels:
                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_cls + 1  # 平均每个类要出现多少次
                # 一次处理每个class label
                for i, l in enumerate(self.dataset.label_values):
                    # 找到当前处理label l 的点云索引
                    label_inds = np.where(np.equal(self.dataset.labels, l))[0]
                    class_potentials = self.potentials([label_inds])  # 取出属于类l的所有点云的potentials值
                    if pick_n < class_potentials.shape[0]:
                        # 某一类的点云个数多于平均数，进行舍弃
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)
                gen_indices = np.random.permutation(np.hstack(gen_indices))  # 将 gen_indices打乱顺序

            else:
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    # argpartition对potentials进行从小到大排序，但是其只负责找到前epoch_n小的数据，其他的不保证，这样可以提高效率
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    # 生成随机序列
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)  # 随机打乱gen_indices的顺序

            # 将选中点的potentials值全部五入到大于等于该值的最小整数
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            # 更新选中点的potentials值
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1
        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_cls + 1
                gen_indices = []
                for l in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.labels, l))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)  # 随机选取pick_n个点，允许重复
                    gen_indices = [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ti_list = []  # 已经添加到当前batch的点云的索引
        batch_n = 0  # 已经添加到当前batch的点的数目

        # 依次处理每个点
        for p_i in gen_indices:

            # 选取的点云的点的个数
            n = self.dataset.points[p_i].shape[0]
            # 如果当前点的数目超过了batch最大限制，打包成一个batch返回
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            ti_list += [p_i]
            batch_n += n
        yield np.array(ti_list, dtype=np.int32)  # 最后一个batch
        return 0

    def __len__(self):
        # yieled samples的数量是不确定的
        return None

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False):
        """
        完成batch数据和neighbors数据的校准。
        Batch calibration: 设置batch_limit
        Neighbors calibration:设置neighborhood_limit
        :param dataloader:
        :param untouched_ratio:
        :param verbost:
        :return:
        """
        print('\nStarting Calibration(use verbose=True for more details)')
        redo = False  # 是否需要重新计算

        ## 计算batch limit
        batch_lim_file = join(self.dataset.data_path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        key = '{:.3f}_{:d}'.format(self.dataset.args.first_subsampling_dl,
                                   self.dataset.args.batch_size)

        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = BColors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = BColors.FAIL
                v = '?'
            print('{:}\"{:s}\":{:s}{:}'.format(color, key, v, BColors.ENDC))

        ## 计算neighbors limit

        neighb_lim_file = join(self.dataset.data_path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        neighb_limits = []
        ## TODO:num_layers怎么算的？
        for layer_ind in range(self.dataset.args.num_layers):
            dl = self.dataset.args.first_subsampling_dl * (2 ** layer_ind)
            if self.dataset.args.deform_layers[layer_ind]:
                r = dl * self.dataset.args.deform_radius
            else:
                r = dl * self.dataset.args.conv_radius

            key = '{:,3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.args.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.args.num_layers):
                dl = self.dataset.args.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.args.deform_layers[layer_ind]:
                    r = dl * self.dataset.args.deform_radius
                else:
                    r = dl * self.dataset.args.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = BColors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = BColors.FAIL
                    v = '?'
                print('{:}\"{:s}\":{:s}{:}'.format(color, key, v, BColors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.conv_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in
                              batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = BColors.FAIL
                        else:
                            color = BColors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                          neighb_hists[layer, neighb_size],
                                                          BColors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                       self.dataset.config.batch_num)
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


def ModelNet40Collate(batch_data):
    return ModelNet40CustomBatch(batch_data)


class ModelNet40CustomBatch:
    """Custom batch definition with memory pinning for ModelNet40"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 5) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_sampling(dataset, sampler, loader):
    """Shows which labels are sampled according to strategy chosen"""
    label_sum = np.zeros((dataset.num_classes), dtype=np.int32)
    for epoch in range(10):

        for batch_i, (points, normals, labels, indices, in_sizes) in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            label_sum += np.bincount(labels.numpy(), minlength=dataset.num_classes)
            print(label_sum)
            # print(sampler.potentials[:6])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, sampler, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.labels) - estim_b) / 100

            # Pause simulating computations
            time.sleep(0.050)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, sampler, loader):
    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, sampler, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


class ModelNet40WorkerInitDebug:
    """Callable class that Initializes workers."""

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __call__(self, worker_id):
        # Print workers info
        worker_info = torch.utils.data.get_worker_info()
        print(worker_info)

        # Get associated dataset
        dataset = worker_info.dataset  # the dataset copy in this worker process

        # In windows, each worker has its own copy of the dataset. In Linux, this is shared in memory
        print(dataset.input_labels.__array_interface__['data'])
        print(worker_info.dataset.input_labels.__array_interface__['data'])
        print(self.dataset.input_labels.__array_interface__['data'])

        # configure the dataset to only process the split workload

        return
