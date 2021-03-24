#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mobilenet_v1_ffd_lm
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from bernstein_ffd.ffd_utils import cp_num
from losses.lm_loss import LMLoss
from losses.deform_loss_ import DeformVDCLoss#, ChamferLoss

# global args (configuration)
args = None
lr = None
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


def parse_args():
    parser = argparse.ArgumentParser(description='FFD')
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int) # 128 for v2
    parser.add_argument('-vb', '--val-batch-size', default=128, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.002, type=float)
    parser.add_argument('--print-freq', '-p', default=500, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='2', type=str)
    parser.add_argument('--filelists-train', default='train.configs/train_aug_120x120.list.train', type=str)
    parser.add_argument('--filelists-val', default='train.configs/train_aug_120x120.list.val', type=str)
    parser.add_argument('--root', default='../Datasets/train_aug_120x120')
    parser.add_argument('--snapshot', default='snapshot/ffd_lm_adam', type=str)
    parser.add_argument('--log-file', default='training/logs/ffd_lm_adam_210303.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--param-classes', default=648, type=int)
    parser.add_argument('--lm-classes', default=136, type=int)
    parser.add_argument('--arch', default='mobilenet_1', type=str, choices=arch_choices)
    parser.add_argument('--milestones', default='30, 40', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--param-fp-train',default='train.configs/param_lm_train.pkl', type=str)
    parser.add_argument('--param-fp-val', default='train.configs/param_lm_val.pkl')
    parser.add_argument('--loss', default='vdc', type=str)

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lm_losses = AverageMeter()
    param_losses = AverageMeter()

    model.train()

    end = time.time()
    step = len(train_loader) // args.print_freq * (epoch - 1)

    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        output = model(input)

        param_target = target[:, :62]
        param_output = output[:, :cp_num]
        lm_target = target[:, 62:]
        lm_output = output[:, cp_num:]
        lm_criterion = LMLoss().cuda()

        data_time.update(time.time() - end)

        param_loss = criterion(param_output, param_target)
        lm_loss = lm_criterion(lm_output, lm_target)
        loss = 0.8 * param_loss + 0.2 * lm_loss

        losses.update(loss.item(), input.size(0))
        lm_losses.update(lm_loss.item(), input.size(0))
        param_losses.update(param_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log
        if i > 0 and i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                         f'Param Loss {param_losses.val:.4f} ({param_losses.avg:.4f})\t'
                         f'Landmark Loss {lm_losses.val:.4f} ({lm_losses.avg:.4f})\t'
                         )
            writer.add_scalar('training_loss', losses.avg, step)
            step += 1

    writer.add_scalar('training_loss_by_epoch', losses.avg, epoch)
    writer.add_scalar('param_loss_by_epoch', param_losses.avg, epoch)
    writer.add_scalar('landmark_loss_by_epoch', lm_losses.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    model.eval()

    end = time.time()
    with torch.no_grad():
        losses = []
        param_losses = []
        lm_losses = []
        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input)

            param_target = target[:, :62]
            param_output = output[:, :cp_num]
            lm_target = target[:, 62:]
            lm_output = output[:, cp_num:]
            lm_criterion = LMLoss().cuda()

            param_loss = criterion(param_output, param_target)
            lm_loss = lm_criterion(lm_output, lm_target)
            loss = 0.8 * param_loss + 0.2 * lm_loss

            losses.append(loss.item())
            lm_losses.append(lm_loss.item())
            param_losses.append(param_loss.item())


        elapse = time.time() - end
        loss = np.mean(losses)
        param_loss = np.mean(param_losses)
        lm_loss = np.mean(lm_losses)

        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     f'Param Loss {param_loss:.4f}\t'
                     f'Landmark Loss {lm_loss:.4f}\t'
                     f'Time {elapse:.3f}')

        writer.add_scalar('validation_loss_by_epoch', loss, epoch)
        writer.add_scalar('param_val_loss_by_epoch', param_loss, epoch)
        writer.add_scalar('landmark_val_loss_by_epoch', lm_loss, epoch)


def main():
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    
    # step1: define the model structure
    model = getattr(mobilenet_v1_ffd_lm, args.arch)(param_classes=args.param_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    criterion = DeformVDCLoss().cuda()
    # chamfer = ChamferLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.base_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)


    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(checkpoint) #, strict=False)

            # # remove fc2 trained for lm regression
            # if "module.fc2.weight" in checkpoint:
            #     checkpoint = OrderedDict([(k,v) for k, v in checkpoint.items() if k != 'module.fc2.weight'])
            #     checkpoint = OrderedDict([(k,v) for k, v in checkpoint.items() if k != 'module.fc2.bias'])

            # model.load_state_dict(checkpoint, strict=False)
            
            # # freeze all layers except last ffd fc layer
            # for name, param in model.named_parameters():
            #     if name not in ['module.fc2.weight', 'module.fc2.bias']:
            #         param.requires_grad = False

            # logging.info(f'=> freezed layers except last ffd layer')

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    print("Loading Data...")
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    print("Data Loaded...")

    # step4: run
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            },
            filename
        )

        validate(val_loader, model, criterion, epoch)


if __name__ == '__main__':
    writer = SummaryWriter('training/runs/ffd_lm_adam')
    main()
    writer.close()