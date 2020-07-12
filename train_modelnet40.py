# ！/usr/bin/python3
# -*- coding: utf-8 -*-

import datetime
import shutil
import warnings
from os import makedirs

import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from datasets.modelnet40_loader import *
from utils.utils import AverageMeter, save_path_formatter

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.enabled = False  # cuDNN使用非确定性算法进行优化，根据实际网络和数据寻找每一层的最优算法
torch.backends.cudnn.deterministic = False  # 使用确定性算法，以方便复现
torch.backends.cudnn.benchmark = False  # 自动寻找最优配置，如果输入数据在每次iteration都有变化的话，cuDNN每次都会寻找一次最优配置，很耗时

print("1. 训练过程保存路径")

save_path = save_path_formatter(args)
args.save_path = join('checkpoints', save_path)
makedirs(args.save_path)

print("=> 将训练过程保存至 {}".format(args.save_path))

train_set = ModelNet40Dataset(args, train=True)
val_set = ModelNet40Dataset(args, train=False)

train_sampler = ModelNet40Sampler(train_set, use_potential=True, balance_labels=True)
val_sampler = ModelNet40Sampler(val_set, use_potential=True, balance_labels=True)

print('{} clouds found train scenes'.format(len(train_set.trees)))
print('{} clouds found valid scenes'.format(len(val_set.trees)))

train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          sampler=train_sampler,  # 自定义选取data是的索引，可以是整数，也可以是list
                          collate_fn=ModelNet40Collate,  # 将采集到的data打包成一个batch
                          num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set,
                        batch_size=args.batch_size,
                        sampler=val_sampler,
                        collate_fn=ModelNet40Collate,
                        num_workers=args.workers,
                        pin_memory=True)

train_sampler.calibration(train_loader)
val_sampler.calibration(val_loader)

print("3.Creating Model")

kpconv_net = models.KPCNN(args).to(device)

if args.pretrained:
    print('=> using pre-trained weights for RandLa-Net')
    weights = torch.load(args.pretrained)
    kpconv_net.load_state_dict(weights['state_dict'], strict=False)
else:
    kpconv_net.init_weights()

kpconv_net = torch.nn.DataParallel(kpconv_net)

print("4. Setting Optimization Solver")
deform_params = [v for k, v in kpconv_net.names_parameters() if 'offset' in k]
other_params = [v for k, v in kpconv_net.names_parameters() if 'offset' not in k]
deform_lr = args.lr * args.deform_lr_factor

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam([{'params': other_params},
                                  {'params': deform_params, 'lr': deform_lr}],
                                 lr=args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD([{'params': other_params},
                                 {'params': deform_params, 'lr': deform_lr}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_dacay)


def poly_scheduler(epoch, base_num=args.base_num):
    return base_num ** epoch


if args.decay_style == 'StepLR':
    exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
else:
    exp_lr_scheduler_R = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_scheduler)

print("5. Start Tensorboard ")
# tensorboard --logdir=/path_to_log_dir/ --port 6006
training_writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information")

with open(args.save_path / args.log, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    # TODO: 确定保存什么内容
    csv_writer.writerow(['train_loss', 'train sem_seg Accuracy', 'Validation seg_seg Accuracy', 'Validation MIoU'])

print("7. Start Training!")


def main():
    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(kpconv_net, optimizer)
        errors, error_names = validate(kpconv_net)

        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error > best_error
        best_error = max(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': kpconv_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_path / 'kpcnn_{}.pth.tar'.format(epoch))

        if is_best:
            shutil.copyfile(args.save_path / 'kpcnn_{}.pth.tar'.format(epoch),
                            args.save_path / 'kpcnn_best.pth.tar')

        for loss, name in zip(losses, loss_names):
            training_writer.add_scalar(name, loss, epoch)
            training_writer.flush()
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            training_writer.flush()

        with open(args.save_path / args.log, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], losses[1], errors[0]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print("Train---Total loss:{}, Accuracy:{}".format(losses[0], accuracy_train))
        print("Valid---Accuracy:{}, MIoU:{}".format(accuracy_val, mean_iou))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(kpconv_net, optimizer):
    loss_names = ['total_loss', 'train_accuracy']
    losses = AverageMeter(i=len(loss_names), precision=4)
    kpconv_net.train()
    optimizer.step()
    val_total_correct = 0
    val_total_seen = 0

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = kpconv_net(batch, args)
        clc_loss, reg_loss, total_loss = kpconv_net(outputs, batch.labels)
        acc = kpconv_net.accuracy(outputs, batch.labels)
        total_loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(kpconv_net.parameters(), args.grab_clip_norm)
        optimizer.step()
        # torch.cuda.synchronize(device)

        losses.update([loss.item(), correct.item() / (args.batch_size * new_n_pts)], args.batch_size)
    accuracy = val_total_correct / val_total_seen

    return losses.avg, loss_names, accuracy


@torch.no_grad()
def validate(kpconv_net):
    error_names = ['train_accuracy']
    errors = AverageMeter(i=len(error_names), precision=4)
    kpconv_net.eval()

    gt_classes = [0 for _ in range(args.num_cls)]
    positive_classes = [0 for _ in range(args.num_cls)]
    true_positive_classes = [0 for _ in range(args.num_cls)]
    val_total_correct = 0
    val_total_seen = 0

    # points: (B,9,n_pts）; label:（B,n_pts); targets:(B,13,n_pts)
    for i, (points, neighbors, pools, up_samples, inputs, labels) in enumerate(val_loader):
        points = points.to(device).float()
        neighbors = neighbors.to(device).long()
        pools = pools.to(device).long()
        up_samples = up_samples.to(device).long()
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = kpconv_net(points, neighbors, pools, up_samples, inputs)  # B,num_cls,n_pts
        # Ignore the invalid point
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, args.num_cls)
        labels = labels.view(-1)
        # ignored_bool = torch.zeros_like(labels, dtype=torch.bool)
        # for ing_label in args.ignored_label_inds:
        #     ignored_bool = ignored_bool + torch.eq(labels, ing_label)
        # valid_idx = torch.squeeze(torch.nonzero(torch.logical_not(ignored_bool)))
        #
        # valid_logits = outputs[valid_idx]
        # valid_labels_init = labels[valid_idx]
        #
        # reducing_list = torch.arange(0, args.num_cls, dtype=torch.long).to(device)
        # inserted_value = torch.zeros((1,), dtype=torch.long).to(device)
        # for ing_label in args.ignored_label_inds:
        #     reducing_list = torch.cat([reducing_list[:ing_label], inserted_value, reducing_list[ing_label:]], 0)
        # valid_labels = reducing_list[valid_labels_init]
        valid_logits = outputs
        valid_labels = labels
        new_n_pts = valid_logits.size()[0]

        pred_val = torch.argmax(valid_logits, 1)  # （n_pts,）
        correct = torch.sum(pred_val == valid_labels)

        valid_labels_cpu = valid_labels.cpu()
        pred_val_cpu = pred_val.cpu()

        val_total_correct += correct.item()  # 有多少个点分类正确
        val_total_seen += len(valid_labels_cpu)  # 总共有多少个点

        conf_matrix = confusion_matrix(valid_labels_cpu, pred_val_cpu, np.arange(0, args.num_cls, 1))

        positive_classes += np.sum(conf_matrix, axis=0)
        gt_classes += np.sum(conf_matrix, axis=1)
        true_positive_classes += np.diagonal(conf_matrix)

        errors.update([correct.item() / (args.batch_size * new_n_pts)], args.batch_size)
    accuracy = val_total_correct / val_total_seen
    iou_list = []
    for n in range(0, args.num_cls, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou = np.nan_to_num(iou)
        print(iou)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(args.num_cls)
    return errors.avg, error_names, accuracy, mean_iou


if __name__ == '__main__':
    main()
