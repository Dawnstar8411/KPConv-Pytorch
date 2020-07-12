import torch.nn as nn
import torch
from torch.nn.init import kaiming_uniform_
from config.args_train import *
import numpy as np
from .blocks import *
from .loss import *


class KPFCNN(nn.Module):
    """
    点云分割网络
    """

    def __init__(self, args, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        layer = 0  # 当前是第几层
        r = args.sub_grid_size * args.conv_radius  # 初始卷积半径，米为单位
        in_dim = args.in_features_dim
        out_dim = args.first_features_dim
        self.num_kpts = args.num_kpts
        self.num_cls = len(lbl_values) - len(ign_lbls)  # 去除掉忽略的分类

        self.encoder_blocks = nn.ModuleList()  # 将网络各层存储在列表中
        self.encoder_skip_dims = []
        self.encoder_skips = []

        for block_i, block in enumerate(args.architecture):
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimensions is not a factor of 3')

            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                # 找出对点云进行上采样或下采样层
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # 先处理encoder部分
            if 'upsample' in block:
                break

            self.encoder_blocks.append(get_block_ops(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     args))

            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            if 'pool' in block or 'strided' in block:
                layer += 1
                r *= 2
                out_dim *= 2
        # 处理decoder部分

        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # 找到第一个上采样层,从而与encoder的module list连接起来
        start_i = 0
        for block_i, block in enumerate(args.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        for block_i, block in enumerate(args.architecture[start_i:]):
            if block_i > 0 and 'upsample' in args.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            self.decoder_blocks.append(get_block_ops(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     args))
            in_dim = out_dim

            if 'upsample' in block:
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, args.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(args.first_features_dim, self.num_cls, False, 0)

        # 损失函数
        # 参与训练的类别
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # 这里平衡各类的权重在args中提供，可以将不同数据集的参数保存在某个文件里，然后直接读取
        if len(args.class_w) > 0:
            class_w = torch.from_numpy(np.array(args.class_w, dtype=np.float32))
            self.cls_criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.cls_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = args.deform_fitting_mode
        self.deform_fitting_power = args.deform_fitting_power
        self.deform_lr_factor = args.deform_lr_factor
        self.repulse_extent = args.repulse_extent
        self.cls_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, args):

        # 输入的特征
        x = batch.features.clone().detach()

        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return

    def loss(self, outputs, labels):
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        self.cls_loss = self.cls_criterion(outputs, target)

        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        return self.cls_loss, self.reg_loss, self.cls_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / totall
