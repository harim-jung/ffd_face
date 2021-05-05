#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
import os
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models.mobilenet_v1_ffd as mobilenet_v1_ffd
import torch.backends.cudnn as cudnn
import torchvision

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from losses.deform_loss_flex import DeformVDCLoss, RegionVDCLoss, VertexOutput, MouthLoss, RegionLMLoss
from losses.lm_loss import LMFittedLoss, LML1Loss
from losses.wpdc_deform_loss import WPDCPoseLoss


# global args (configuration)
args = None
lr = None

def parse_args():
    parser = argparse.ArgumentParser(description='FFD')
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int) # 128 for v2
    parser.add_argument('--val-batch-size', default=64, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.001, type=float)
    parser.add_argument('--print-freq', '-p', default=1000, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    # parser.add_argument('--resume', default='snapshot/ffd_resnet_region/ffd_resnet_region_checkpoint_epoch_33.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='2', type=str)
    parser.add_argument('--filelists-train', default='train.configs/train_aug_120x120.list.train', type=str)
    parser.add_argument('--filelists-val', default='train.configs/train_aug_120x120.list.val', type=str)
    parser.add_argument('--root', default='../Datasets/train_aug_120x120')
    parser.add_argument('--snapshot', default='snapshot/ffd_resnet_pred_pose_vertex_lm', type=str)
    parser.add_argument('--log-file', default='training/logs/ffd_resnet_pred_pose_vertex_lm_210411.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--dimensions', default='3, 6, 3', type=str)
    parser.add_argument('--param-classes', default=348, type=int) # 336 + 12
    parser.add_argument('--arch', default='resnet', type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--milestones', default='30, 40', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--param-fp-train',default='train.configs/param_all_full_norm.pkl', type=str) # todo - changed to normalized version
    parser.add_argument('--param-fp-val', default='train.configs/param_all_val_full_norm.pkl', type=str)
    parser.add_argument('--source_mesh', default='300w-lp mean shape', type=str)
    parser.add_argument('--loss', default='pose_vdc_lm_mse', type=str)
    parser.add_argument('--weights', default='0.15, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05', type=str)
    # parser.add_argument('--weights', default='0.15, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001', type=str)


    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]
    args.weights = [float(w) for w in args.weights.split(",")]

    if not osp.isdir(args.snapshot): 
        os.mkdir(args.snapshot)
    
    # snapshot_dir = osp.split(args.snapshot)[0]
    # mkdir(snapshot_dir)


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


def train(train_loader, model, criterion, vertex_criterion, lm_criterion, param_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    param_losses = AverageMeter()
    vertex_losses = AverageMeter()
    lm_losses = AverageMeter()
    up_mouth_losses = AverageMeter()
    low_mouth_losses = AverageMeter()
    up_nose_losses = AverageMeter()
    low_nose_losses = AverageMeter()
    l_brow_losses = AverageMeter()
    r_brow_losses = AverageMeter()
    l_eye_losses = AverageMeter()
    r_eye_losses = AverageMeter()
    contour_losses = AverageMeter()

    model.train()

    end = time.time()
    step = len(train_loader) // args.print_freq * (epoch - 1)

    loss_weights = args.weights.copy()
    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)
        output = model(input)

        # param_target = target# [:, :12]
        param_output = output[:, :12]
        deform_output = output[:, 12:]

        param_loss = param_criterion(param_output, target)

        target_vert, deformed_vert = vertex_criterion(output, target)

        vertex_loss = criterion(deformed_vert, target_vert, loss_type="mse")
        up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour = lm_criterion(deformed_vert, target_vert, loss_type="mse")

        loss_group = torch.stack((param_loss, vertex_loss, up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour))
        # delta_p_l2 = torch.mean(torch.sqrt(deform_output ** 2))
        # loss_group = torch.stack((param_loss, vertex_loss, up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour, delta_p_l2))

        # delta_p_norm = (delta_p_l2 / (3 * epoch)).item()
        # loss_weights[-1] = delta_p_norm
        loss = torch.tensor(loss_weights).double().cuda() @ loss_group
        # update loss
        # loss = 0.46 * vertex_loss + 0.06 * up_mouth + 0.06 * low_mouth + 0.06 * up_nose + 0.06 * low_nose + 0.06 * l_brow + \
        # 0.06 * r_brow + 0.06 * l_eye + 0.06 * r_eye + 0.06 * contour
        
        losses.update(loss.item(), input.size(0))
        param_losses.update(param_loss.item(), input.size(0))
        vertex_losses.update(vertex_loss.item(), input.size(0))
        lm_losses.update(loss_group[2:].mean(), input.size(0)) # without vertex loss
        up_mouth_losses.update(up_mouth.item(), input.size(0))
        low_mouth_losses.update(low_mouth.item(), input.size(0))
        up_nose_losses.update(up_nose.item(), input.size(0))
        low_nose_losses.update(low_nose.item(), input.size(0))
        l_brow_losses.update(l_brow.item(), input.size(0))
        r_brow_losses.update(r_brow.item(), input.size(0))
        l_eye_losses.update(l_eye.item(), input.size(0))
        r_eye_losses.update(r_eye.item(), input.size(0))
        contour_losses.update(contour.item(), input.size(0))


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
                         f'Up Mouth Loss {up_mouth_losses.val:.4f} ({up_mouth_losses.avg:.4f})\t'
                         f'Low Mouth Loss {low_mouth_losses.val:.4f} ({low_mouth_losses.avg:.4f})\t'
                         f'Up Nose Loss {up_nose_losses.val:.4f} ({up_nose_losses.avg:.4f})\t'
                         f'Low Nose Loss {low_nose_losses.val:.4f} ({low_nose_losses.avg:.4f})\t'
                         f'Left Brow Loss {l_brow_losses.val:.4f} ({l_brow_losses.avg:.4f})\t'
                         f'Right Brow Loss {r_brow_losses.val:.4f} ({r_brow_losses.avg:.4f})\t'
                         f'Left Eye Loss {l_eye_losses.val:.4f} ({l_eye_losses.avg:.4f})\t'
                         f'Right Eye Loss {r_eye_losses.val:.4f} ({r_eye_losses.avg:.4f})\t'
                         f'Contour Loss {contour_losses.val:.4f} ({contour_losses.avg:.4f})\t'
                         f'Landmark Loss {lm_losses.val:.4f} ({lm_losses.avg:.4f})\t'
                         f'Vertex Loss {vertex_losses.val:.4f} ({vertex_losses.avg:.4f})\t'
                         f'Param Loss {param_losses.val:.4f} ({param_losses.avg:.4f})\t'
                        #  f'Delta P Norm {delta_p_norm} \t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            )

                         
            writer.add_scalar('training_loss', losses.avg, step)
            step += 1

    writer.add_scalar('training_loss_by_epoch', losses.avg, epoch)
    writer.add_scalar('param_loss_by_epoch', param_losses.avg, epoch)
    writer.add_scalar('vertex_loss_by_epoch', vertex_losses.avg, epoch)
    writer.add_scalar('landmark_loss_by_epoch', lm_losses.avg, epoch)
    writer.add_scalar('up_mouth_loss_by_epoch', up_mouth_losses.avg, epoch)
    writer.add_scalar('low_mouth_loss_by_epoch', low_mouth_losses.avg, epoch)
    writer.add_scalar('up_nose_loss_by_epoch', up_nose_losses.avg, epoch)
    writer.add_scalar('low_nose_loss_by_epoch', low_nose_losses.avg, epoch)
    writer.add_scalar('l_brow_loss_by_epoch', l_brow_losses.avg, epoch)
    writer.add_scalar('r_brow_loss_by_epoch', r_brow_losses.avg, epoch)
    writer.add_scalar('l_eye_loss_by_epoch', l_eye_losses.avg, epoch)
    writer.add_scalar('r_eye_loss_by_epoch', r_eye_losses.avg, epoch)
    writer.add_scalar('contour_loss_by_epoch', contour_losses.avg, epoch)

def validate(val_loader, model, criterion, vertex_criterion, lm_criterion, param_criterion,  epoch, log=True):
    model.eval()

    end = time.time()

    with torch.no_grad():
        losses = []
        lm_losses = []
        vertex_losses = []
        param_losses = []
        up_mouth_losses = []
        low_mouth_losses = []
        up_nose_losses = []
        low_nose_losses = []
        l_brow_losses = []
        r_brow_losses = []
        l_eye_losses = []
        r_eye_losses = []
        contour_losses = []

        loss_weights = args.weights.copy()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input)

            # param_target = target #[:, :12]
            param_output = output[:, :12]
            deform_output = output[:, 12:]

            param_loss = param_criterion(param_output, target)

            target_vert, deformed_vert = vertex_criterion(output, target)

            vertex_loss = criterion(deformed_vert, target_vert, loss_type="mse")
            up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour = lm_criterion(deformed_vert, target_vert, loss_type="mse")
            
            # delta_p_l2 = torch.mean(torch.sqrt(deform_output ** 2))
            # loss_group = torch.stack((param_loss, vertex_loss, up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour, delta_p_l2))
            loss_group = torch.stack((param_loss, vertex_loss, up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour))

            # delta_p_norm = (delta_p_l2 / (3 * epoch)).item()
            # loss_weights[-1] = delta_p_norm
            loss = torch.tensor(loss_weights).double().cuda() @ loss_group

            losses.append(loss.item())
            lm_losses.append(loss_group[2:].mean().item())
            param_losses.append(param_loss.item())
            vertex_losses.append(vertex_loss.item())
            up_mouth_losses.append(up_mouth.item())
            low_mouth_losses.append(low_mouth.item())
            up_nose_losses.append(up_nose.item())
            low_nose_losses.append(low_nose.item())
            l_brow_losses.append(l_brow.item())
            r_brow_losses.append(r_brow.item())
            l_eye_losses.append(l_eye.item())
            r_eye_losses.append(r_eye.item())
            contour_losses.append(contour.item())

        elapse = time.time() - end
    
        loss = np.mean(losses)
        lm_loss = np.mean(lm_losses)
        param_loss = np.mean(param_losses)
        vertex_loss = np.mean(vertex_losses)
        up_mouth_loss = np.mean(up_mouth_losses)
        low_mouth_loss = np.mean(low_mouth_losses)
        up_nose_loss = np.mean(up_nose_losses)
        low_nose_loss= np.mean(low_nose_losses)
        l_brow_loss = np.mean(l_brow_losses)
        r_brow_loss = np.mean(r_brow_losses)
        l_eye_loss = np.mean(l_eye_losses)
        r_eye_loss = np.mean(r_eye_losses)
        contour_loss = np.mean(contour_losses)
        
        
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                    f'Up Mouth Loss {up_mouth_loss:.4f}\t'
                    f'Low Mouth Loss {low_mouth_loss:.4f}\t'
                    f'Up Nose Loss {up_nose_loss:.4f}\t'
                    f'Low Nose Loss {low_nose_loss:.4f}\t'
                    f'Left Brow Loss {l_brow_loss:.4f}\t'
                    f'Right Brow Loss {r_brow_loss:.4f}\t'
                    f'Left Eye Loss {l_eye_loss:.4f}\t'
                    f'Right Eye Loss {r_eye_loss:.4f}\t'
                    f'Contour Loss {contour_loss:.4f}\t'
                     f'Landmark Loss {lm_loss:.4f}\t'
                     f'Vertex Loss {vertex_loss:.4f}\t'
                     f'Param Loss {param_loss:.4f}\t'
                     f'Loss {loss:.4f}\t'
                     f'Time {elapse:.3f}')
        if log:
            writer.add_scalar('validation_loss_by_epoch', loss, epoch)
            writer.add_scalar('param_val_loss_by_epoch', param_loss, epoch)
            writer.add_scalar('vertex_val_loss_by_epoch', vertex_loss, epoch)
            writer.add_scalar('landmark_val_loss_by_epoch', lm_loss, epoch)
            writer.add_scalar('up_mouth_val_loss_by_epoch', up_mouth_loss, epoch)
            writer.add_scalar('low_mouth_val_loss_by_epoch', low_mouth_loss, epoch)
            writer.add_scalar('up_nose_val_loss_by_epoch', up_nose_loss, epoch)
            writer.add_scalar('low_nose_val_loss_by_epoch', low_nose_loss, epoch)
            writer.add_scalar('l_brow_val_loss_by_epoch', l_brow_loss, epoch)
            writer.add_scalar('r_brow_val_loss_by_epoch', r_brow_loss, epoch)
            writer.add_scalar('l_eye_val_loss_by_epoch', l_eye_loss, epoch)
            writer.add_scalar('r_eye_val_loss_by_epoch', r_eye_loss, epoch)
            writer.add_scalar('contour_val_loss_by_epoch', contour_loss, epoch)


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
    if args.arch.startswith("mobilenet"):
        model = getattr(mobilenet_v1_ffd, args.arch)(param_classes=args.param_classes)
    elif args.arch.startswith("resnet"):
        model = torchvision.models.resnet50(pretrained=False, num_classes=args.param_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    vertex_criterion = VertexOutput().cuda()
    criterion = DeformVDCLoss().cuda()
    lm_criterion = RegionLMLoss().cuda()
    param_criterion = WPDCPoseLoss().cuda()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)


    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            
            model.load_state_dict(checkpoint)
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
    snapshot_ind = osp.split(args.snapshot)[1]
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, vertex_criterion, lm_criterion, param_criterion, args.start_epoch, log=False)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    # for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, criterion, vertex_criterion, lm_criterion, param_criterion, optimizer, epoch)
        filename = f'{args.snapshot}/{snapshot_ind}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            },
            filename
        )

        validate(val_loader, model, criterion, vertex_criterion, lm_criterion, param_criterion, epoch)


if __name__ == '__main__':
    writer = SummaryWriter('training/runs/ffd_resnet_pred_pose_vertex_lm')
    main()
    writer.close()