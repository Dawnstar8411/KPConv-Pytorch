# ！/usr/bin/python3
# -*- coding: utf-8 -*-

from models.loss import *

from .blocks import *


class KPCNN(nn.Module):
    """
    Kernel point conv， 点云分类网络
    """

    def __init__(self, args):
        super(KPCNN, self).__init__()
        layer = 0  # 当前是第几层
        r = args.sub_grid_size * args.conv_radius  # 初始卷积范围，米为单位
        in_dim = args.in_features_dim  # 输入点云特征维度
        out_dim = args.first_features_dim  # 输出的第一层特征的维度
        self.num_kpts = args.num_kpts  # kernel point 个数

        self.block_ops = nn.ModuleList()  # 将所有的网络模块保存在一个列表中

        block_in_layer = 0
        # 根据architecture 列表中的顺序构建网络
        for block_i, block in enumerate(args.architecture):
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # 分类网络，遇到上采样层即停止构建
            if 'upsample' in block:
                break

            self.block_ops.append(get_block_ops(block,  # 模块名字
                                                r,  # 卷积半径
                                                in_dim,  # 输入特征维度
                                                out_dim,  # 输出特征维度
                                                layer,  # 当前层的编号，根据此从batch中提取相应的点云数据
                                                args))
            # 当前模块在当前层中的编号，网络的层编号是与输入的不同尺度的点云相对应的
            block_in_layer += 1

            # simple层的输出是out_dim//2, 更改in_dim的值，保持下一层的输入维度与simple 层的输出维度一致
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # 遇到下采样层时，更新相关参数
            if 'pool' in block or 'strided' in block:
                layer += 1  # 网络进行了下采样操作，因此下一层的数据要更新了
                r *= 2  # 网络进行了下采样，卷积半径需要扩大为之前的2倍
                out_dim *= 2  # 输出维度扩大一倍
                block_in_layer = 0  # 层内模块编号清零

        # 最后的分类操作，全连接层
        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, args.num_cls, False, 0)

        # 损失函数相关参数
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = args.deform_fitting_mode
        self.deform_fitting_mode = args.deform_fitting_power
        self.deform_lr_factor = args.deform_lr_factor
        self.repulse_extent = args.repulse_extent
        self.cls_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

    def forward(self, batch, args):
        x = batch.features.clone().detach()

        # 按网络模块顺序处理数据
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # 最后的分类操作
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):

        # 分类结果的损失
        self.cls_loss = self.cls_criterion(outputs, labels)

        # 对deformable offsets的约束
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        return self.cls_loss, self.reg_loss, self.clc_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        predicted = torch.armax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total
