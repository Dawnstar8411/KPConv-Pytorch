# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import datetime
from os.path import join


class BColors:
    """
    在终端打印时字体的颜色
    例：print('{:}{:s}{:}'.format(bcolors.OKGREEN, key, bcolors.ENDC))
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def save_path_formatter(args):
    """根据args中的 数据集名称--模型名称--时间 生成保存路径"""
    args_dict = vars(args)
    save_path = []
    save_path.append(str(args_dict['dataset_name']))  # 数据集名称
    save_path.append(str(args_dict['model_name']))  # 模型名称
    save_path = '-'.join(save_path)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")  # 训练时间: 月-天-小时-分钟
    if args.is_debug:
        return join(save_path, 'debug')
    else:
        return join(save_path, timestamp)


class AverageMeter(object):
    """计算和存储当前值和累积的平均值"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):  # 这里的n是指val传过来的数据是几个数的平均值
        if not isinstance(val, list):
            val = [val]
        assert (len(val) == self.meters)
        self.count += n
        for i, value in enumerate(val):
            self.val[i] = value
            self.sum[i] += value * n
            self.avg[i] = self.sum[i] / self.count
