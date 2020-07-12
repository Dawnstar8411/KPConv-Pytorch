import argparse

parser = argparse.ArgumentParser(description="Kernel Point Convolution Network",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 数据，模型权重，训练日志的保存与读取
parser.add_argument('data_path', metavar='DIR', help='Path to dataset')
parser.add_argument('--dataset_name', default="modelnet40", type=str,
                    choices=['s3dis', 'scannet', 'semantic3d', 'shapenet', 'modelnet40', 'semantic_kitti'])
parser.add_argument('--model_name', default="KPConv_Net", type=str)
parser.add_argument('--task', default="segmentation", typ=str, choices=['segmentation', 'classification'])
parser.add_argument('--part_task', default='multi', type=str)  # part segmentation的类型，全部都参与分类，还是只对某一个类进行分割
parser.add_argument('--seed', default=2048, type=int, help="Seed for random function and network initialization.")
parser.add_argument('--pretrained', default=None, metavar='PATH', help="Path to pre-trained model.")
parser.add_argument('--log_file', default='progress_log.csv', metavar='PATH', help='Name of the log file')

# 网络训练
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--workers', default=8, type=int, metavar='N', help="Number of data loading workers")
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help="Number of total epochs to run")
parser.add_argument('--batch_size', default=1, type=int, help="Batch size during training")
parser.add_argument('--train_steps', default=1000, type=int, metavar='N', help="Number of batches for training")
parser.add_argument('--valid_steps', default=100, type=int, metavar='N', help="Mumber of batches for validation")
parser.add_argument('--snapshot_gap', default=50, type=int, help="number of epoch between each snapshot")

# 网络优化器
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help="Momentum for sgd, alpha for adam")
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='Beta parameters for adam')
parser.add_argument('--weight_decay', default=0.001, type=float, metavar='W', help='Weight decay')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--decay_style', default='LambdaLR', choices=['LambdaLR', 'StepLR'])
parser.add_argument('--decay_basenum', default=0.95, type=float, help="parameter for lambdaLR")
parser.add_argument('--decay_step', default=50, type=int, help="Decay step for StepLR")
parser.add_argument('--decay_rate', default=0.7, type=float, help="Decay rate for StepLR")

# 网络结构
parser.add_argument('--use_batch_norm', default=True, type=bool, help='Batch normalization parameters')
parser.add_argument('--batch_norm_momentum', default=0.99, type=float, help='Batch normalization parameters')
parser.add_argument('--architecture', default=None, nargs='+')
parser.add_argument('--first_features_dim', default=64, type=int, help='Dimension of the first feature maps')
parser.add_argument('--in_features_dim', default=1)  # 输入特征的维度
parser.add_argument('--in_points_dim', default=3, type=int)
parser.add_argument('--aggregation_mode', default='sum', choices=['closet', 'sum'],
                    help='Decide if you sum all kernel point influences, or only take the influence of the closest KP')
# influence function when d<KP_extent. when d>KP_extent, always zero
parser.add_argument('--KP_influence', default='linear', choices=['linear', 'constant', 'gaussian'])
parser.add_argument('--KP_extent', default=1.0, type=float, help='Kernel point influence radius')  # 非deformal卷积时，控制邻域密度
parser.add_argument('--conv_radius', default=2.5, type=float,
                    help='Radius of convolution in "number grid cell"')  # 卷积的时候范围为几个grid
parser.add_argument('--density_parameter', default=5.0)  # deformal 卷积时的邻域密度
parser.add_argument('--deform_radius', default=5.0, type=float,
                    help='Radius of deformable convolution in "number grid cell",larger so that deformed kernel can spread out')
parser.add_argument('--modulated', default=False, help='use modulation in deformable convolutions')

# 损失函数
parser.add_argument('--regular_weight_decay', default=1e-3, type=float, help='Regularization loss importance')
parser.add_argument('--offsets_loss', default='fitting', choices=['permissive', 'fitting'],
                    help='permissive only constrains offsets inside the big radius,'
                         'fitting helps deformed kernels to adapt to the geometry '
                         'by penalizing distance to input points')
parser.add_argument('--offsets_decay', default=0.1, type=float)
# 'point2point' fitting geometry by penalizing distance from deform point to input points
# 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
parser.add_argument('--deform_fitting_mode', default='point2point', choices=['point2point', 'point2plane'])
parser.add_argument('--deform_fitting_power', default=1.0, type=float, help='Multiplier for the fitting/repulsive loss')
parser.add_argument('--deform_lr_factor', default=0.1, type=float,
                    help='Multiplier for leanring rate applied to the deformations')
parser.add_argument('--repulse_extent', default=1.0, type=float,
                    help='Distance of repulsion for deformed kernel points')
parser.add_argument('--grad_clip_norm', default=100.0, type=float,
                    help='Gradient clipping value, negative means no clipping')
parser.add_argument('--batch_average_loss', default=False,
                    help='Type of output loss with regard to batches when segmentation')
# The way we balance segmentation loss
#   > 'none': Each point in the whole batch has the same contribution.
#   > 'class': Each class has the same contribution (points are weighted according to class balance)
#   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
parser.add_argument('--segloss_balance', default='none', choices=['none', 'class', 'batch'])
parser.add_argument('--class_w', default=[],
                    help='Choose weights for class(used in segmentation loss). Empty list fot no weights')

# 数据准备
parser.add_argument('--in_radius', default=1.0, type=float)  # 输入的球形半径，单位米
parser.add_argument('--segmentation_ratio', default=1.0, type=float,
                    help='for segmentation models: ratio between the segmented area and the input area')
# for each layer, support points are subsampled on a grid with dl=kernel_radius/density_parameter
parser.add_argument('--density_parameter', default=3.0, type=float, help='density of neighbors in kernel range')
parser.add_argument('--num_pts', default=40960, type=int, help='Point Number')
parser.add_argument('--num_cls', default=13, type=int, help='Class number')
parser.add_argument('--num_kpts', default=15, type=int, help='Number of kernel points')
parser.add_argument('--sub_grid_size', default=0.04, type=float, help='preprocess parameter.')
parser.add_argument('--fixed_kernel_points', default='center', choices=['none', 'center', 'verticals'],
                    help='fixed points in the kernel')

# 具体数据库

# S3DIS
parser.add_argument('--val_split', default=5, type=int, help='Which area to use for test, [default:5]')
# SLAM 数据库
parser.add_argument('--n_frames', default=1, type=int,
                    help='For SLAM datasets like SemantiKitti number of frames used (minimum one)')
parser.add_argument('--max_in_points', default=0, type=int)
parser.add_argument('--val_radius', default=51.0, type=float)
parser.add_argument('--max_val_points', default=50000, type=int)

# 数据增强参数
parser.add_argument('--augment_scale_anisotropic', default=True, type=bool)  # 进行尺度缩放时，三个轴采用相同的还是不同的缩放系数
parser.add_argument('--augment_symmetries', default=[False, False, False])  # 在哪个轴上使坐标对称
parser.add_argument('--augment_rotation', default='none', choices=['none', 'vertical', 'all'])  # 旋转方式
parser.add_argument('--augment_scale_min', default=0.9, type=float)  # 尺度最小值
parser.add_argument('--augment_scale_max', default=1.1, type=float)  # 尺度最大值
parser.add_argument('--augment_noise', default=0.005, type=float)  # 尺度变换时添加的高斯噪声标准差
parser.add_argument('--augment_occlusion', default='none', choices=['none', 'planar'])
parser.add_argument('--augment_occlusion_ratio', default=0.2)
parser.add_argument('--augment_occlusion_num', default=1)
parser.add_argument('--augment_color', default=0.7, type=float)

# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

parser.add_argument('--equivar_mode', default='', help='decide the mode of equivariance')
parser.add_argument('--invar_mode', default='', help='decide the mode of invariance')

args = parser.parse_args()

