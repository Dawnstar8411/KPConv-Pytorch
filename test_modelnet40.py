from torch.utils.data import DataLoader
from datasets.modelnet40_loader import *
import models
from config.args_test import *


if __name__ == '__main__':
    chosen_log = ''
    on_val = true # 测试模式

    warnings.filterwarnings('ignore')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    test_dataset = ModelNet40Dataset(args,train=False)
    test_sampler = ModelNet40Sampler(test_dataset)
    collate_fn = ModelNet40Collate

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler = test_sampler,
                             collate_fn=collate_fn,
                             num_workers=args.workers,
                             pin_memory=True)

    test_sampler.calibration(test_loader, verbose=True)

    kpconv_net = models.KPCNN(args)

    tester = ModelTester(kpconv_net, chkp_path = chose_chkp)

    # if config.dataset_task == 'classification':
    #     a = 1 / 0
    # elif config.dataset_task == 'cloud_segmentation':
    #     tester.cloud_segmentation_test(net, test_loader, config)
    # elif config.dataset_task == 'slam_segmentation':
    #     tester.slam_segmentation_test(net, test_loader, config)
    # else:
    #     raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
