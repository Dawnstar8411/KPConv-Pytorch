import os
from torch.utils.data import DataLoader
import warnings

from datasets.modelnet40_loader import *
from datasets.s3dis_loader import *
from datasets.scannet_loader import *
from datasets.semantic3d_loader import *
from datasets.shapenet_loader import *
from config.args_vis import *
import models

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

chosen_log = '../checkpoints/xxx.pth'

deform_idx = 0  # 可以选择可视化那个feature

# 这些都写在脚本里
# config.augment_noise = 0.0001
# config.batch_num = 1
# config.in_radius = 2.0
# config.input_threads = 0

if args.dataset_name == 'modelnet40':
    test_dataset = ModelNet40Dataset(args, train=False)
    test_sampler = ModelNet40Dataset(test_dataset)
    collate_fn = ModelNet40Collate
elif args.dataset_name == 's3dis':
    test_dataset = S3DISDataset(args, use_potentials=True, train=False)
    test_sampler = S3DISSampler(test_dataset)
    collate_fn = S3DISCollate
else:
    raise ValueError('Unsupported dataset:' + args.dataset_name)

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         sampler=test_sampler,
                         collate_fn=collate_fn,
                         num_workers=args.workers,
                         pin_memory=True)
test_sampler.calibration(test_loader, verbose=True)

if args.tast == 'classification':
    kpconv_net = models.KPCNN(args)
elif args.test == 'segmentation':
    kpconv_net = models.KPFCNN(args, test_dataset.label_values, test_dataset.ignored_labels)
else:
    raise ValueError('Unsupported dataset_task for deformation visu: ' + config.dataset_task)

visualizer = ModelVisualizer(kpconv_net, args, chkp_path=chosen_log, on_gpu=False)

if __name__ == 'main':
    visualizer.show_deformable_kernels(kpconv_net, test_loader, args, deform_idx)


class ModelVisualizer:
    def __init__(self, net, args, chkp_path, on_gpu=True):
        """
        :param net: 需要运行的网络
        :param args: 参数集合
        :param chkp_path: 加载的网络权重文件路径
        :param on_gpu: 是否使用GPU
        """

        self.device = torch.device("cuda") if on_gpu and torch.cuda.is_available() else torch.device("cpu")
        net.to(self.device)
        checkpoint = torch.load(chkp_path)

        new_dict = {}

        for k, v in checkpoint['state_dict'].items():
            if 'blocs' in k:
                k = k.replace('blocs', 'blocks')
                new_dict[k] = v

        net.load_state_dict(new_dict)
        self.epoch = checkpoint['epoch']
        net.eval()