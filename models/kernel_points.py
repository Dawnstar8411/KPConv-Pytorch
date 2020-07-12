# ！/usr/bin/python3
# -*- coding: utf-8 -*-

from os import makedirs
from os.path import join, exists

import matplotlib.pyplot as plt
import numpy as np
from utils.utils import BColors
from utils.helper_ply_io import read_ply, write_ply
from utils.helper_data_processing import create_3D_rotations


def spherical_Lloyd(radius, num_cells, dimension=3, fixed='center', approximation='monte-carlo',
                    approx_n=5000, max_iter=500, momentum=0.9, verbose=0):
    """
    使用Lloyd算法生成kernel point. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: 生成kernel points的坐标半径，这里统一设为1，后面再进行缩放
    :param num_cells: kernel points的点的个数，Voronoi diagram中cell的数量
    :param dimension: 三维点还是二维点
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Lloyd's 算法的近似方法，('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :param verbose: 是否进行显示
    :return: points [num_kernels, num_points, dimension]
    """

    radius0 = 1.0  # 优化时点的半径，最后再根据radius的值进行缩放

    kernel_points = np.zeros((0, dimension))  # 用来保存kernel_points的数组
    while kernel_points.shape[0] < num_cells:  # 不停地生成-筛选，知道留下的点的数目大于等于需要的点为止
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0  # 随机生成点，[-radius0,radius0]范围
        kernel_points = np.vstack((kernel_points, new_points))  # 将原来的点与新生成的点拼接在一起
        d2 = np.sum(np.power(kernel_points, 2), axis=1)  # 计算点的坐标的平方和
        kernel_points = kernel_points[np.logical_and(d2 < radius0 ** 2, (0.9 * radius0) ** 2 < d2), :]  # 筛选附和条件的点
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))  # 选取出num_cell个点，作为优化的初始值

    # Optional fixing
    if fixed == 'center':  # 如果固定模式为 "center"，固定点为(0,0,0)
        kernel_points[0, :] *= 0
    if fixed == 'verticals':  # 如果固定模式为 "verticals"，固定点为(0,0,0),(0,0,2/3),(0,0,-2/3)，三个点都在z轴上
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    ##############################
    # 近似算法初始化
    ##############################

    # 初始化显示界面
    if verbose > 1:
        fig = plt.figure()

    # Initialize discretization in this method is chosen
    if approximation == 'discretization':
        side_n = int(np.floor(approx_n ** (1. / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError('Unsupported dimension (max is 4)')
    elif approximation == 'monte-carlo':
        X = np.zeros((0, dimension))
    else:
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))

    # Only points inside the sphere are used
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]

    #####################
    # 优化过程
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    max_moves = np.zeros((0,))

    for iter in range(max_iter):

        # 蒙特卡洛法，重新生成点
        if approximation == 'monte-carlo':
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0  # 随机生成approx_n个点
            d2 = np.sum(np.power(X, 2), axis=1)  # 每个点坐标的平方和
            X = X[d2 < radius0 * radius0, :]  # 取半径为radius0的球形内的点

        differences = np.expand_dims(X, 1) - kernel_points  # X中每个点与所有kernel points的坐标差 [approx_n, num_kpts, dimension]
        sq_distances = np.sum(np.square(differences), axis=2)  # 坐标差的平方和 [approx_n, num_kpts]

        # Compute cell centers
        cell_inds = np.argmin(sq_distances, axis=1)  # approx_n个点分别于哪个kernel points最近
        centers = []
        for c in range(num_cells):
            bool_c = (cell_inds == c)  # 得到与第c个kernel point最近的有哪些点
            num_c = np.sum(bool_c.astype(np.int32))  # 计算随机生成的approx_n个点中，有多少个点的最近邻是第c个kernel point
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)  # 这些点的坐标的平均值
            else:
                warning = True
                centers.append(kernel_points[c])

        # 更新kernel points的坐标
        centers = np.vstack(centers)  # [num_kpts, dimension]
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # 找到最大的move值
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        # 固定某个或某几个点
        if fixed == 'center':
            kernel_points[0, :] *= 0
        if fixed == 'verticals':
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

        if verbose:
            print('iter {:5d} / max move = {:f}'.format(iter, np.max(np.linalg.norm(moves, axis=1))))
            if warning:
                print('{:}WARNING: at least one point has no cell{:}'.format(BColors.WARNING, BColors.ENDC))
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0,
                        marker='.', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)

    ###################
    # 用户最终确认
    ###################

    if verbose:
        if dimension == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0,
                        marker='.', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect('equal')
            plt.title('Check if kernel is correct.')
            plt.draw()
            plt.show()

        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title('Check if kernel is correct.')
            plt.show()

    # 返回kernel points
    return kernel_points * radius


def kernel_point_optimization_debug(radius=1.0, num_kpoints=15, num_kernels=1, dimension=3, fixed='center', ratio=1.0,
                                    verbose=0):
    """
    :param radius: kernel points的尺度，默认为1.0
    :param num_kpoints: 要生成的kernel points的点数
    :param num_kernels: 要生成多少组候选kernel points, 最后会从中选择一组最优的
    :param dimension: 二维点还是三维点
    :param fixed: kernel points 固定点方式，'none','vertical','center'
    :param ratio: 比例值
    :param verbose: 是否进行可视化
    :return: kernel_points [num_kernels, num_kpoints, dimension], saved_gradient_norms [10000, num_kernels]
    """
    radius0 = 1  # 以半径为1生成kernel_points,最后再根据真实的尺度进行缩放
    diameter0 = 2  # 直径

    moving_factor = 1e-2  # 类似于学习率
    continuous_moving_decay = 0.9995  # 学习率的衰减系数

    thresh = 1e-5  # 提前停止优化的梯度阈值

    clip = 0.05 * radius0  # 梯度的切片值

    # 随机初始化kernel_points,这里减1是为了满足while循环的条件，防止一开始就进不去循环
    kernel_points = np.random.rand(num_kernels * num_kpoints - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_kpoints):
        # 1. 不断生成新的随机点
        new_points = np.random.rand(num_kernels * num_kpoints - 1, dimension) * diameter0 - radius0
        # 2. 将新的点与旧的点合并在一起
        kernel_points = np.vstack((kernel_points, new_points))
        # 3. 求每个点的x,y,z坐标的平方和
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        # 4. 选择出平方和小于某个值的点，直到选择出足够数量的点为止
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    # 选择 num_kernels * num_kpoints个点，然后reshape
    kernel_points = kernel_points[:num_kernels * num_kpoints, :].reshape((num_kernels, num_kpoints, -1))

    # 选择中心点的确定方式
    if fixed == 'center':  # 如果固定模式为 "center"，固定点为(0,0,0)
        kernel_points[:, 0, :] *= 0
    elif fixed == 'verticals':  # 如果固定模式为 "verticals"，固定点为(0,0,0),(0,0,2/3),(0,0,-2/3)，三个点都在z轴上`
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] += 2 * radius0 / 3

    # Kernel points优化
    if verbose > 1:
        fig = plt.figure()
    # 保存所有轮优化的梯度
    saved_gradient_norms = np.zeros((10000, num_kernels))
    # 每一轮的优化过程中，每个点的梯度
    old_gradient_norms = np.zeros((num_kernels, num_kpoints))  # e.g (100,15)
    for iter in range(10000):
        A = np.expand_dims(kernel_points, axis=2)  # 第二个轴加一维
        B = np.expand_dims(kernel_points, axis=1)  # 第一个轴加一维
        interd2 = np.sum(np.power(A - B, 2), axis=-1)  #
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        circle_grads = 10 * kernel_points

        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # 每个点的x,y,z的梯度的平方和开根号
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        # 在每组kernel points中，找出gradients_norm的最大值 => [10000,num_kernels]
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # 两轮优化的所有点的梯度的L2范数的变化量的最大值小于某个阈值时，停止迭代, 不同的中心点有少许细节的不同
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms  # 将当前所有点的梯度范数保存，用作下一轮比较

        moving_dists = np.nimimum(moving_factor * gradients_norms, clip)  # 超过clip的都按clip算

        # 第一个点不更新
        if fixed == 'center' or fixed == 'verticals':
            moving_dists[:, 0] = 0

        # 更新点的位置
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad={:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')  # 只画出点的x,y坐标
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))  # 设置横坐标轴数值范围
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))  # 设置纵坐标轴的数值范围
            fig.axes[0].set_aspect('equal')  # x,y轴设置为相同的尺度
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)
        moving_factor *= continuous_moving_decay

    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))  # 求所有点向量的模 [num_kernels,num_points]
    kernel_points *= ratio / np.mean(r[:, 1:])  # 除以所有点向量的模长的平均值

    # 将kernel_points 缩放到真实的尺度
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpts, dimension, fixed, lloyd=False):
    """
    :param radius: 半径，对生成的kernel points进行缩放的倍数
    :param num_kpts: kernal point的个数
    :param dimension: 点云空间的维度，2维点云或3维点云
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals').
    :param lloyd: 选择不同的生成kernel points的算法
    :return: 生成的kernel points
    """

    # 保存kernel points的路径
    kernel_dir = 'kernels'
    if not exists(kernel_dir):
        makedirs(kernel_dir)  # 创建保存kernel points 文件的文件夹

    if num_kpts > 30:
        lloyd = True

    # 保存kernel point的文件
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.ply'.format(num_kpts, fixed, dimension))

    # 判断是否存在kernel points
    if not exists(kernel_file):
        if lloyd:
            kernel_points = spherical_Lloyd(1.0,
                                            num_kpts,
                                            dimension=dimension,
                                            fixed=fixed,
                                            verbose=0)
        else:
            kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                        num_kpoints=num_kpts,
                                                                        num_kernels=100,
                                                                        dimension=dimension,
                                                                        fixed=fixed,
                                                                        verbose=0)
            # 寻找最优值
            best_k = np.argmin(grad_norms[-1, :])
            # 保存kernel points
            kernel_points = kernel_points[best_k, :, :]
        write_ply(kernel_file, kernel_points, ['x', 'y', 'z'])
    else:
        data = read_ply(kernel_file)  # 读取kernel points 坐标 （num_pts,dimension）
        kernel_points = np.vstack((data['x'], data['y'], data['z'])).T

    theta = np.random.rand() * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    # 暂不支持2D点的处理，直接返回原始值
    if dimension == 2:
        if fixed != 'vertical':
            rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)
    elif dimension == 3:
        if fixed == 'verticals':  # 三个固定点在z轴上，绕z轴旋转
            rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        else:  # 可绕原点随意旋转
            phi = (np.random.rand() - 0.5) * np.pi
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            alpha = np.random.rand() * 2 * np.pi
            rotation_matrix = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
            rotation_matrix = rotation_matrix.astype(np.float32)
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)  # 添加随机噪声
    kernel_points = radius * kernel_points  # 缩放kernel points坐标为原来的radius倍
    kernel_points = np.matmul(kernel_points, rotation_matrix)
    return kernel_points.astype(np.float32)


if __name__ == '__main__':
    kernel_point = load_kernels(radius=1, num_kpts=40, dimension=3, fixed='vertical', lloyd=False)
