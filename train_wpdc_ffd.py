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
import models.mobilenet_v1_ffd_lm
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from losses.vdc_loss import VDCLoss
from losses.wpdc_loss import WPDCLoss
from losses.lm_loss import LMLoss

# global args (configuration)
args = None
lr = None
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting (WPDC)')
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=512, type=int) # 128 for v2
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.02, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float)
    parser.add_argument('--print-freq', '-p', default=50, type=int)
    parser.add_argument('--resume', default='snapshot/phase2_wpdc_lm_vdc_all_checkpoint_epoch_19.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0', type=str)
    parser.add_argument('--filelists-train', default='train.configs/train_aug_120x120.list.train', type=str)
    parser.add_argument('--filelists-val', default='train.configs/train_aug_120x120.list.val', type=str)
    parser.add_argument('--root', default='../data/train_aug_120x120')
    parser.add_argument('--snapshot', default='snapshot/wpdc_ffd', type=str)
    parser.add_argument('--log-file', default='training/logs/wpdc_ffd_210216.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--param-classes', default=62, type=int)
    parser.add_argument('--lm-classes', default=136, type=int)
    parser.add_argument('--arch', default='mobilenet_1', type=str, choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--milestones', default='30, 40', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--param-fp-train',default='train.configs/param_lm_train.pkl', type=str)
    parser.add_argument('--param-fp-val', default='train.configs/param_lm_val.pkl')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='wpdc', type=str)

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


def train(train_loader, model, criterion, ffd_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    param_losses = AverageMeter()
    ffd_losses = AverageMeter()
    # lm_losses = AverageMeter()


    model.train()

    end = time.time()
    step = len(train_loader) // args.print_freq * (epoch - 1)

    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        output = model(input)

        param_target = target[:, :62]
        param_output = output[:, :62]
        # lm_target = target[:, 62:]
        # lm_output = output[:, 62:]
        # lm_criterion = LMLoss().cuda()

        data_time.update(time.time() - end)

        if args.loss.lower() == 'vdc':
            loss = criterion(param_output, param_target)
        elif args.loss.lower() == 'wpdc':
            param_loss = criterion(param_output, param_target) # wpdc_loss
            lm_loss = lm_criterion(lm_output, lm_target)
            loss = 0.8 * param_loss + 0.2 * lm_loss
        else:
            raise Exception(f'Unknown loss {args.loss}')


        losses.update(loss.item(), input.size(0))
        param_losses.update(param_loss.item(), input.size(0))
        lm_losses.update(lm_loss.item(), input.size(0))

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
            writer.add_scalar('training loss', losses.avg, step)
            step += 1

    writer.add_scalar('training loss by epoch', losses.avg, epoch)
    writer.add_scalar('param loss by epoch', param_losses.avg, epoch)
    writer.add_scalar('landmark loss by epoch', lm_losses.avg, epoch)

def validate(val_loader, model, criterion, ffd_criterion, epoch):
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
            lm_target = target[:, 62:]
            param_output = output[:, :62]
            lm_output = output[:, 62:]
            lm_criterion = LMLoss().cuda()

            if args.loss.lower() == 'vdc':
                param_loss = criterion(param_output, param_target) # vdc_loss
                lm_loss = lm_criterion(lm_output, lm_target)
                loss = 0.8 * param_loss + 0.2 * lm_loss
            elif args.loss.lower() == 'wpdc':
                param_loss = criterion(param_output, param_target) # wpdc_loss
                lm_loss = lm_criterion(lm_output, lm_target)
                loss = 0.8 * param_loss + 0.2 * lm_loss

            losses.append(loss.item())
            param_losses.append(param_loss.item())
            lm_losses.append(lm_loss.item())

        elapse = time.time() - end
        loss = np.mean(losses)
        param_loss = np.mean(param_losses)
        lm_loss = np.mean(lm_losses)
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     f'Param Loss {param_loss:.4f}\t'
                     f'Landmark Loss {lm_loss:.4f}\t'
                     f'Time {elapse:.3f}')

        writer.add_scalar('validation loss by epoch', loss, epoch)
        writer.add_scalar('param val loss by epoch', param_loss, epoch)
        writer.add_scalar('landmark val loss by epoch', lm_loss, epoch)

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
    model = getattr(mobilenet_v1_ffd_lm, args.arch)(param_classes=args.param_classes, lm_classes=args.lm_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU


    # step2: optimization: loss and optimization method
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use VDC Loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)


    # map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    # checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    # if "module.fc1.weight" in checkpoint:
    #     checkpoint = OrderedDict([('module.fc.weight', v) if k == 'module.fc1.weight' else (k,v) for k, v in checkpoint.items()])
    #     checkpoint = OrderedDict([('module.fc.bias', v) if k == 'module.fc1.bias' else (k,v) for k, v in checkpoint.items()])

    # model.load_state_dict(checkpoint, strict=False)


    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)
            print(checkpoint.keys())

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
    writer = SummaryWriter('training/runs/wpdc_ffd')
    main()
    writer.close()