# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def fast_confusion(true, pred, label_values=None):
    """
    快速confusion matrix 计算（比Scikit learn的快100倍）
    :param true: Ground Truth label值
    :param pred: 预测的label 值
    :param label_values: 所有的label值列表，元素不能有重复
    :return: confusion matrix
    """
    true = np.squeeze(true)  # 去掉为1的维度
    pred = np.squeeze(pred)

    if len(true.shape) != 1:
        raise ValueError('True values are stored in a {:d}D array instead of 1d array'.format(len(true.shape)))
    if len(pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1d array'.format(len(pred.shape)))
    if true.dtype not in [np.int32, np.int64]:
        raise ValueError('True values are {:s} instead of int32 or int64'.format(true.dtype))
    if pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))

    true = true.astype(np.int32)
    pred = pred.astype(np.int32)

    if label_values is None:
        # 从true 和 pred中统计有那些label值
        label_values = np.unique(np.hstack((true, pred)))
    else:
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values)) < len(label_values):
            raise ValueError('Given labels are not unique')

    label_values = np.sort(label_values)  # 按从小到大的顺序排序

    num_classes = len(label_values)  # 类别总数，用来构造confusion matrix

    # 进行confusion  matrix 计算
    if label_values[0] == 0 and label_values[-1] == num_classes - 1:
        # label值以0结束，以类别数目减1结束
        # np.bincount(x) 的结果为一个array,其中第i个位置的数字表示的是在x中，i这个值出现了几次
        vec_conf = np.bincount(true * num_classes + pred)

        # np.bincount(x)的长度有x中的最大值决定。假设true中没有出现最后一个label值，vec_conf最后就会少num_classes个值
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes * 2 - vec_conf.shape[0]), 'constant')

        return vec_conf.reshape((num_classes, num_classes))  # 将向量变换成矩阵的形式
    else:
        if label_values[0] < 0:  # 确保label values都是大于等于0的
            raise ValueError('Unsupported negative classes')
        label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
        for k, v in enumerate(label_values):
            label_map[v] = k  # 构建label value值与索引的映射,confusion matrix使用索引值来计算

        pred = label_map[pred]
        true = label_map[true]

        vec_conf = np.bincount(true * num_classes + pred)

        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes * 2 - vec_conf.shape[0]), 'constant')

        return vec_conf.reshape((num_classes, num_classes))


def metrics(confusions, ignore_unclassified=False):
    """
    由confusion matrix计算各种指标
    :param confusions: ([..., n_c, n_c] np.int32). 前几维维度任意，最后两个维度为confusion matrix
    :param ignore_unclassified: (bool). 是否忽略掉'unclassified'类
    :return: ([..., n_c] np.float32) 准确率，召回率，F1值，IoU交并比
    """
    # 当第一类的label为'unclassified'时，该类不参与计算，confusion matrix 第一行和第一列都设为0
    if ignore_unclassified:
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    # 使用sklearn中的confusion_matrix函数求得的confusion matrix，行代表ground truth, 列代表predictions
    # 二维的confusion matrix中，第一个维度代表ground truth，第二个维度代表predictions,例如第3行第4列代表第3类物体被分成第4类的个数
    tp = np.diagonal(confusions, axis1=-2, axis2=-1)  # 提取对角线元素
    tp_plus_fp = np.sum(confusions, axis=-2)  # 每一列求和，表示被预测为各个类别的数目
    tp_plus_fn = np.sum(confusions, axis=-1)  # 每一行求和，表示某个类别的总数目

    precision = tp / (tp_plus_fp + 1e-6)  # 准确率
    recall = tp / (tp_plus_fn + 1e-6)  # 召回率

    accuracy = np.sum(tp, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)  # 计算准确率，所有预测正确的个数，除以所有的样本数
    f1 = 2 * tp / (tp_plus_fp + tp_plus_fn + 1e-6)  # 计算每一类的F1值
    iou = f1 / (2 - f1)  # 计算每一类的交并比

    return precision, recall, f1, iou, accuracy


def smooth_metrics(confusions, smooth_n=0, ignore_unclassified=False):
    """
    由confusion matrix计算各项指标，在一定数量的epochs上做平滑
    :param confusions: ([..., n_c, n_c] np.int32). 最后两维是confusion matrix, 倒数第三位是epoch维
    :param smooth_n: (int). 平滑范围
    :param ignore_unclassified: (bool). 第一个class是否要被忽略掉
    :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
    """

    # 当第一类的label为'unclassified'时，该类不参与计算，confusion matrix 第一行和第一列都设为0
    if ignore_unclassified:
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    smoothed_confusions = confusions.copy()
    if confusions.ndim > 2 and smooth_n > 0:  # 具有epoch维且需要进行平滑
        for epoch in range(confusions.shape[-3]):
            i0 = max(epoch - smooth_n, 0)  # 平滑起点
            i1 = min(epoch + smooth_n + 1, confusions.shape[-3])  # 平滑终点
            # 将 i0 到 i1之间的confusion matrix加起来，替代该epoch原始的confusion matrix
            smoothed_confusions[..., epoch, :, :] = np.sum(confusions[..., i0:i1, :, :], axis=-3)

    tp = np.diagonal(smoothed_confusions, axis1=-2, axis2=-1)
    tp_plus_fp = np.sum(smoothed_confusions, axis=-2)
    tp_plus_fn = np.sum(smoothed_confusions, axis=-1)

    precision = tp / (tp_plus_fp + 1e-6)
    recall = tp / (tp_plus_fn + 1e-6)

    accuracy = np.sum(tp, axis=-1) / (np.sum(smoothed_confusions, axis=(-2, -1)) + 1e-6)  # 计算准确率，所有预测正确的个数，除以所有的样本数
    f1 = 2 * tp / (tp_plus_fp + tp_plus_fn + 1e-6)  # 计算每一类的F1值
    iou = f1 / (2 - f1)  # 计算每一类的交并比

    return precision, recall, f1, iou, accuracy


def IoU_from_confusions(confusions, ignore_unclassified=False):
    """
    由混淆矩阵计算IoU交并比
    :param confusions: ([..., n_c, n_c] np.int32
    :param ignore_unclassified: (bool). 是否忽略掉第一个类
    :return: ([..., n_c] np.float32) IoU score
    """

    # 当第一类的label为'unclassified'时，该类不参与计算，confusion matrix 第一行和第一列都设为0
    if ignore_unclassified:
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    tp = np.diagonal(confusions, axis1=-2, axis2=-1)
    tp_plus_fp = np.sum(confusions, axis=-2)
    tp_plus_fn = np.sum(confusions, axis=-1)

    # 计算交并比
    iou = tp / (tp_plus_fp + tp_plus_fn - tp + 1e-6)

    mask = tp_plus_fn < 1e-3  # 某些类别没有出现
    counts = np.sum(1 - mask, axis=-1, keepdims=True)  # 计算有多少类出现了
    miou = np.sum(iou, axis=-1, keepdims=True) / (counts + 1e-6)  # 只计算已经出现了的类别的mIoU

    # 如果某些类别没有出现，就用出现的类的mIoU来代替，方便后面统一计算平均值
    iou += mask * miou

    return iou
