# ！/usr/bin/python3
# -*- coding: utf-8 -*-
import colorsys
import random
from os.path import join, exists
import numpy as np
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.visualization import draw_geometries
from mayavi import mlab
import torch
from .helper_points_rotation import euler2mat
import matplotlib.pyplot as plt
import csv


########################################################
# random_colors(n, bright=True, seed=0)
# draw_pc(pc_xyzrgb)
# draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None)
########################################################


def random_colors(n, bright=True, seed=0):
    """
    生成n个RGB颜色
    :param n: 生成多少种随机的颜色
    :param bright: 明亮度控制参数
    :param seed: 随机种子
    :return: n个打乱顺序的RGB颜色,例：[(1.0, 0.0, 0.901), (1.0, 0.0, 0.299), (0.5, 1.0, 0.0)]
    """
    brightness = 1.0 if bright else 0.7  # hsv空间的饱和度
    hsv = [(0.15 + i / float(n), 1, brightness) for i in range(n)]  # n个hsv颜色
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))  # 将hsv空间转换到rgb空间
    random.seed(seed)
    random.shuffle(colors)  # 随机打乱n中颜色
    return colors


def draw_pc(pc_xyzrgb):
    """
    显示点云
    :param pc_xyzrgb: [n,3 or 6]
    :return:
    """
    pc = PointCloud()  # 创建一个点云实例
    pc.points = Vector3dVector(pc_xyzrgb[:, 0:3])  # 点云的x,y,z坐标
    if pc_xyzrgb.shape[1] == 3:  # 如果只提供了坐标值，灰度显示
        draw_geometries([pc])
        return 0
    if np.max(pc_xyzrgb[:, 3:6]) > 20:  # 如果提供了点云的颜色值，彩色显示
        pc.colors = Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)  # 若rgb值在[0，255],将颜色值归一化到[0,1]区间
        draw_geometries([pc])
        return 0
    else:
        pc.colors = Vector3dVector(pc_xyzrgb[:, 3:6])  # 使用[0,1]空间的rgb数值
        draw_geometries([pc])
        return 0


def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
    """
    显示点云，不同类别的点显示不同的颜色
    pc_xyz: [n,3] 点云的坐标
    pc_sem_ins: [n,] 每个点的label
    plot_colors: 用来显示点云的颜色列表
    """
    sem_ins_labels = np.unique(pc_sem_ins)  # 所有的类别列表

    if plot_colors is not None:
        ins_colors = plot_colors
    else:
        ins_colors = random_colors(len(sem_ins_labels) + 1, seed=2)  # 生成颜色列表

    sem_ins_bbox = []  # 每一类的点的bounding_box
    y_colors = np.zeros((pc_sem_ins.shape[0], 3))  # (n,3),初始化为全0，根据label赋予颜色值

    # 依次处理每个类别的点着色
    for _, sem_ins in enumerate(sem_ins_labels):
        valid_ind = np.argwhere(pc_sem_ins == sem_ins)[:, 0]  # 找到属于当前类的所有点的索引
        if sem_ins <= -1:
            tp = [0, 0, 0]
        else:
            tp = ins_colors[sem_ins]

        y_colors[valid_ind] = tp  # 为所有属于此类的点赋予tp颜色值
        valid_xyz = pc_xyz[valid_ind]  # 所有属于此类的点的坐标值

        # 得到bounding box的对角端点的坐标
        x_min = np.min(valid_xyz[:, 0])
        x_max = np.max(valid_xyz[:, 0])
        y_min = np.min(valid_xyz[:, 1])
        y_max = np.max(valid_xyz[:, 1])
        z_min = np.min(valid_xyz[:, 2])
        z_max = np.max(valid_xyz[:, 2])
        sem_ins_bbox.append(
            [[x_min, y_min, z_min], [x_max, y_max, z_max], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

    y_sem_ins = np.concatenate([pc_xyz[:, 0:3], y_colors], axis=-1)  # 点云坐标和颜色值[x,y,z,r,g,b]
    draw_pc(y_sem_ins)  # 显示点云
    return y_sem_ins


# TODO: 显示被错误分割的点
def draw_pc_error(pc_xyz, pred, gt, plot_colors=True):
    pass


########################################################
#
#
#
#######################################################


def point_cloud_three_views(points):
    """
    从三个角度显示点云
    :param points: （n,3), 要显示的点云坐标，y轴是向上方向
    :return: numpy数组，500*1500的灰度值图片
    Examples
    >>> data = read_ply('example.ply')
    >>> points = np.vstack((data['x'],data['y'],data['z'])).T
    >>> im_array = point_cloud_three_views(points)
    >>> img = Image.fromarray(np.uint8(im_array * 255))
    """
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def draw_point_cloud(input_points, canvaszSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """
    显示点云
    :param input_points: （n,3）点云坐标
    :param canvaszSize: 返回图像的尺寸
    :param space:
    :param diameter:
    :param xrot:
    :param yrot:
    :param zrot:
    :param switch_xyz: 变换输入点云的 x,y,z坐标顺序
    :param normalize: 是否对点云进行归一化，缩放到一个半径为1的球内
    :return: numpy 数组，灰度图
    """

    image = np.zeros((canvaszSize, canvaszSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]  # 变换点云三个轴的顺序
    rotation_matrix = euler2mat(zrot, yrot, xrot)  # 由欧拉旋转角变换成旋转矩阵
    points = (np.dot(rotation_matrix, points.transpose())).transpose()  # 对点云进行旋转变换

    if normalize:
        centroid = np.mean(points, axis=0)  # 所有点云x,y,z的平均值
        points -= centroid  # 减去均值
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))  # 求出距离原点最远的点
        points /= furthest_distance  # 进行归一化，将所有的点缩放到半径为1的球形内

    # 计算高斯模板
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))

    mask = np.argwhere(disk > 0)  # disk中大于0的值的坐标
    dx = mask[:, 0]  # x坐标
    dy = mask[:, 1]  # y坐标
    dv = disk[disk > 0]  # x,y坐标对应的disk值

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])  # 按z轴从小到大排列的索引
    points = points[zorder, :]  # 将points按z轴从小到大排列
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))  # 对z轴坐标进行归一化
    max_depth = np.max(points[:, 2])  # z轴的最大值

    for i in range(points.shape[0]):
        j = points.shape[0] - 1 - i  # 从后向前取点，即按z轴的值从大往小取
        x = points[j, 0]
        y = points[j, 1]
        xc = canvaszSize / 2 + (x * space)  # 当前点在图像上x轴的投影坐标
        yc = canvaszSize / 2 + (y * space)  # 当前点在图像上y轴的投影坐标
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc  # 每个点用一个球来表示，以xc为中心，通过高斯mask在某个范围内进行坐标扩展
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    image = image / np.max(image)  # 对图像之进行归一化，缩放到[0,1]之间
    return image


def pyplot_draw_point_cloud(points, output_filename):
    """
    在三维坐标系中显示点云并保存
    :param points: (n,3) 点云坐标
    :param output_filename: 保存图像的路径
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """

    :param vol: （vsize,vsize,vsize）的立方体表示
    :param output_filename: 显示点云并保存成图像
    """
    points = volume_to_point_cloud(vol)  # 将立方体表示转换成点云表示
    pyplot_draw_point_cloud(points, output_filename)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """
    将点云表示转换成立方体表示
    :param points: (n,3)点云坐标
    :param vsize:  立方体边长
    :param radius: 输入点云的坐标范围[-radius,radius]
    :return: (vsize,vsize,vsize)大小的数组
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)  # 立方体中每个单位对应的点云坐标尺寸
    locations = (points + radius) / voxel  # 把点云平移到第一象限，计算每个点云属于立方体中的哪个方格
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0  # 若有点落到方格内，将方格体素设置为1
    return vol


def volume_to_point_cloud(vol):
    """

    :param vol: （vsize,vsize,vsize）大小的立方体数组，值为0或1
    :return: （n,3）点云坐标
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))  # 如果一个体素值为1，则将其坐标作为点的三维坐标
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)  # （n,3）点云
    return points


def point_cloud_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """
    Batch级别数据的转换
    :param point_clouds: （B,n,3）点云坐标
    :param vsize: 立方体边长
    :param radius: 点云半径
    :param flatten: 是否将立方体数组展开成1位数组
    :return: (B,vsize,vsize,vsize)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())  # 将vol转换成一维数组
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


########################################################
#
# 通过键盘互动的方式，显示点云数据，包括：点云图，邻居点云图，各层级采样图
#
#######################################################


def show_ModelNet_models(points):
    """
    显示点云
    :param points: （num,N,3）点云数组
    """
    # 创建显示窗口
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(100, 800))
    fig1.scene.parallel_projection = False

    global file_i
    file_i = 0

    def update_scene():
        mlab.clf(fig1)  # 清空figure
        vis_points = points[file_i]  # 取出要显示的点云
        vis_points = (vis_points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50  # 对点云进行缩放
        # show point clouds colorized with activations
        activations = mlab.points3d(vis_points[:, 0],
                                    vis_points[:, 1],
                                    vis_points[:, 2],
                                    vis_points[:, 2],
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- （press g for previous' + 50 * '' + '(press h for next)---->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()
        return

    def keyboard_callback(vrk_obj, event):
        global file_i
        if vrk_obj.GetKeyCode() in ['g', 'G']:  # 前一个
            file_i = (file_i - 1) % len(points)  # 更新点云索引
            update_scene()  # 更新显示内容
        elif vrk_obj.GetKeyCode() in ['h', 'H']:  # 后一个
            file_i = (file_i + 1) % len(points)  # 更新点云索引
            update_scene()  # 更新显示内容
        return

    # 显示第一张图，后续的根据键盘输入显示
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_examples(clouds, cloud_normals=None, cloud_labels=None):
    """
    显示点云
    :param clouds: （B,n,3）
    :param cloud_normals:  (B,n,3）
    :param cloud_labels: (B,n）
    """
    # 创建显示窗口
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    if cloud_labels is None:
        cloud_labels = [points[:, 2] for points in clouds]

    # 点云索引
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        # 清除当前figure
        mlab.clf(fig1)

        # 准备新的数据
        points = clouds[file_i]
        labels = cloud_labels[file_i]
        if cloud_normals is not None:
            normals = cloud_normals[file_i]
        else:
            normals = None

        # 对点云进行缩放以显示
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    labels,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)
        if normals is not None and show_normals:
            activations = mlab.quiver3d(points[:, 0],
                                        points[:, 1],
                                        points[:, 2],
                                        normals[:, 0],
                                        normals[:, 1],
                                        normals[:, 2],
                                        scale_factor=10.0,
                                        scale_mode='none',
                                        figure=fig1)

        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # 显示第一张图，后续的根据键盘输入显示
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_neighbors(query, supports, neighbors):
    """
    在query中选一个点，显示其在supports中的邻居点
    :param query: （n,3）
    :param supports: （n,3）
    :param neighbors: (k,)
    """
    # 创建显示窗口
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # 点云索引
    global file_i
    file_i = 0

    def update_scene():

        # 清空当前窗口
        mlab.clf(fig1)

        # 对点云进行缩放以显示
        p1 = (query * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        p2 = (supports * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        l1 = p1[:, 2] * 0  # query中所有点设置为0
        l1[file_i] = 1  # 第file_i个中心点，设置为1

        l2 = p2[:, 2] * 0 + 2  # supports中所有点设置为2
        l2[neighbors[file_i]] = 3  # 第file_i个邻居点组设置为3

        # 显示点云
        activations = mlab.points3d(p1[:, 0],
                                    p1[:, 1],
                                    p1[:, 2],
                                    l1,
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        activations = mlab.points3d(p2[:, 0],
                                    p2[:, 1],
                                    p2[:, 2],
                                    l2,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(query)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(query)
            update_scene()

        return

    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_batch(batch):
    """
    显示输入数据，各个层级提前采样好的数据
    :param batch: dataloader中准备好的一个batch的数据
    """
    # 创建显示窗口
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_pools = batch.unstack_pools()

    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # 对点云坐标进行缩放以显示
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2] * 0

        if show_pools:
            p2 = (all_points[l_i + 1][b_i][neighb_i:neighb_i + 1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32) * 3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """

        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


# TODO:根据训练文件中保存的log文件完成下列内容
########################################################
#
# 画训练，测试中间过程曲线的辅助函数
#
#######################################################

def running_mean(signal, n, axis=0, stride=1):
    signal = np.array(signal)
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2 * n + 1, stride=stride, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2 * n + 1)
    if signal.ndim == 1:
        torch_signal = torch.from_numpy(signal.reshape([1, 1, -1]).astype(np.float32))
        return torch_conv(torch_signal).squeeze().numpy()

    elif signal.ndim == 2:
        print('TODO implement with torch and stride here')
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2 * n + 1,)), mode='same')
                sig_num = np.convolve(sig * 0 + 1, np.ones((2 * n + 1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2 * n + 1,)), mode='same')
                sig_num = np.convolve(sig * 0 + 1, np.ones((2 * n + 1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_class_metrics(all_IoUs, smooth_n):
    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):
    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


## TODO：根据train文件中实际保存的数据修改
def load_training_results(path):
    filename = join(path, 'progress_log.csv')  # TODO: 确定一下文件名称
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)  # 读取文件中的每一行

    epochs = []
    L_out = []
    L_p = []
    acc = []
    t = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            epochs += [int(line_info[0])]
            L_out += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
        else:
            break

    return epochs, L_out, L_p, acc, t


# TODO: IoU从文件中的第几列读？
def load_single_IoU(filename, n_parts):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)

    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):
    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += fast_confusion(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def compare_trainings(list_of_paths, list_of_labels=None):
    """
    画出训练过程的 学习率，损失，时间曲线
    :param list_of_paths: log文件列表
    :param list_of_labels: log文件的名称，作为区分彼此的标识
    """
    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]  # 用数字序列标识log文件

    plot_lr = False  # 是否画学习率曲线
    smooth_epochs = 0.5
    stride = 2

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []
    all_RAMs = []

    for path in list_of_paths:
        # Load results
        epochs, L_out, L_p, acc, t = load_training_results(path)
        # epochs = np.array(epochs, dtype=np.int32)
        # epochs_d = np.array(epochs, dtype=np.float32)

        # max_e = np.max(epochs)   # epoch的最大值
        # first_e = np.min(epochs) # epoch的起始值
        # epoch_n = []
        # for i in range(first_e, max_e):
        #     bool0 = epochs == i
        #     e_n = np.sum(bool0)
        #     epoch_n.append(e_n)
        #     epochs_d[bool0] += steps[bool0] / e_n
        # smooth_n = int(np.mean(epoch_n) * smooth_epochs)
        # smooth_loss = running_mean(L_out, smooth_n, stride=stride)
        all_loss += [L_out]
        all_epochs += [epochs]
        all_times += [t]

        ## TODO: 如何有args中的初始lr和衰减计算lr数组
        if plot_lr:
            lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
            lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
            max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
            lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
            lr_decays[0] = float(config.learning_rate)
            lr_decays[lr_decay_e] = lr_decay_v
            lr = np.cumprod(lr_decays)
            all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

    # 画出学习率曲线
    # *******************

    if plot_lr:
        fig = plt.figure('lr')  # 创建figure
        # 每个文件画一条学习率曲线
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        plt.xlabel('epochs')  # x轴名称
        plt.ylabel('lr')  # y轴名称
        plt.yscale('log')  # y轴设置为log尺度

        plt.legend(loc=1)  # figure 位置 'upper right'

        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')

    # 画损失函数曲线
    # **********

    fig = plt.figure('loss')  # 创建figure
    # 每个训练文件画一条loss曲线
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

    plt.xlabel('epochs')  # x轴名称
    plt.ylabel('loss')  # y轴名称
    plt.yscale('log')  # y轴设置为log尺度

    plt.legend(loc=1)  # figure 位置 'upper right'
    plt.title('Losses compare')

    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')

    # 画出时间曲线
    # **********

    fig = plt.figure('time')  # 创建figure
    # 每个训练文件画一条时间曲线
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    plt.xlabel('epochs')  # x轴名称
    plt.ylabel('time')  # y轴名称

    plt.legend(loc=0)  # figure位置为'best'

    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')

    plt.show()


def compare_convergences_segment(dataset, num_cls, list_of_paths, list_of_names=None):
    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]  # 用数字序列标识不同的log文件

    # 类别名称列表
    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    smooth_n = 10

    all_pred_epochs = []
    all_mIoUs = []
    all_class_IoUs = []
    all_snap_epochs = []
    all_snap_IoUs = []

    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10 * '-' + '|' + 10 * num_cls * '-')

    for path in list_of_paths:

        file = join(path, 'progress_log.csv')
        val_IoUs = load_single_IoU(file, num_cls)

        # Get mean IoU
        class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_mIoUs += [mIoUs]
        all_class_IoUs += [class_IoUs]

        s = '{:^10.1f}|'.format(100 * mIoUs[-1])
        for IoU in class_IoUs[-1]:
            s += '{:^10.1f}'.format(100 * IoU)
        print(s)

        # Get optional full validation on clouds
        snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
        all_snap_epochs += [snap_epochs]
        all_snap_IoUs += [snap_IoUs]

    print(10 * '-' + '|' + 10 * num_cls * '-')
    for snap_IoUs in all_snap_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^10.1f}|'.format(100 * np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^10.1f}'.format(100 * IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(num_cls):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    # plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            # plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_classif(list_of_paths, list_of_labels=None):
    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 12

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_OA = []
    all_train_OA = []
    all_vote_OA = []
    all_vote_confs = []

    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(list_of_paths[0])

        # Get the number of classes
        n_class = config.num_classes

        # Load epochs
        epochs, _, _, _, _, _ = load_training_results(path)
        first_e = np.min(epochs)

        # Get validation confusions
        file = join(path, 'val_confs.txt')
        val_C1 = load_confusions(file, n_class)
        val_PRE, val_REC, val_F1, val_IoU, val_ACC = smooth_metrics(val_C1, smooth_n=smooth_n)

        # Get vote confusions
        file = join(path, 'vote_confs.txt')
        if exists(file):
            vote_C2 = load_confusions(file, n_class)
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = smooth_metrics(vote_C2, smooth_n=2)
        else:
            vote_C2 = val_C1
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = (val_PRE, val_REC, val_F1, val_IoU, val_ACC)

        # Aggregate results
        all_pred_epochs += [np.array([i + first_e for i in range(len(val_ACC))])]
        all_val_OA += [val_ACC]
        all_vote_OA += [vote_ACC]
        all_vote_confs += [vote_C2]

    print()

    # Best scores
    # ***********

    for i, label in enumerate(list_of_labels):
        print('\n' + label + '\n' + '*' * len(label) + '\n')
        print(list_of_paths[i])

        best_epoch = np.argmax(all_vote_OA[i])
        print('Best Accuracy : {:.1f} % (epoch {:d})'.format(100 * all_vote_OA[i][best_epoch], best_epoch))

        confs = all_vote_confs[i]

        """
        s = ''
        for cc in confs[best_epoch]:
            for c in cc:
                s += '{:.0f} '.format(c)
            s += '\n'
        print(s)
        """

        TP_plus_FN = np.sum(confs, axis=-1, keepdims=True)
        class_avg_confs = confs.astype(np.float32) / TP_plus_FN.astype(np.float32)
        diags = np.diagonal(class_avg_confs, axis1=-2, axis2=-1)
        class_avg_ACC = np.sum(diags, axis=-1) / np.sum(class_avg_confs, axis=(-1, -2))

        print('Corresponding mAcc : {:.1f} %'.format(100 * class_avg_ACC[best_epoch]))

    # Plots
    # *****

    for fig_name, OA in zip(['Validation', 'Vote'], [all_val_OA, all_vote_OA]):

        # Figure
        fig = plt.figure(fig_name)
        for i, label in enumerate(list_of_labels):
            plt.plot(all_pred_epochs[i], OA[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel(fig_name + ' Accuracy')

        # Set limits for y axis
        # plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # for i, label in enumerate(list_of_labels):
    #    print(label, np.max(all_train_OA[i]), np.max(all_val_OA[i]))

    # Show all
    plt.show()


def compare_convergences_SLAM(dataset, list_of_paths, list_of_names=None):
    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_mIoUs = []
    all_val_class_IoUs = []
    all_subpart_mIoUs = []
    all_subpart_class_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^6}|'.format('mean')
    for c in class_list:
        s += '{:^6}'.format(c[:4])
    print(s)
    print(6 * '-' + '|' + 6 * config.num_classes * '-')
    for path in list_of_paths:

        # Get validation IoUs
        nc_model = dataset.num_classes - len(dataset.ignored_labels)
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, nc_model)

        # Get Subpart IoUs
        file = join(path, 'subpart_IoUs.txt')
        subpart_IoUs = load_single_IoU(file, nc_model)

        # Get mean IoU
        val_class_IoUs, val_mIoUs = IoU_class_metrics(val_IoUs, smooth_n)
        subpart_class_IoUs, subpart_mIoUs = IoU_class_metrics(subpart_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_val_mIoUs += [val_mIoUs]
        all_val_class_IoUs += [val_class_IoUs]
        all_subpart_mIoUs += [subpart_mIoUs]
        all_subpart_class_IoUs += [subpart_class_IoUs]

        s = '{:^6.1f}|'.format(100 * subpart_mIoUs[-1])
        for IoU in subpart_class_IoUs[-1]:
            s += '{:^6.1f}'.format(100 * IoU)
        print(s)

    print(6 * '-' + '|' + 6 * config.num_classes * '-')
    for snap_IoUs in all_val_class_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^6.1f}|'.format(100 * np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^6.1f}'.format(100 * IoU)
        else:
            s = '{:^6s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^6s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_subpart_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_pred_epochs[i], all_val_mIoUs[i], linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    # plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    # displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_val_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            # plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()
