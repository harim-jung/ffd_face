#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch.nn as nn
import torch, cv2
import torchvision.transforms as transforms
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz, reconstruct_vertex, LpDataset, _parse_param, _parse_ffd_param, _parse_param_batch
from utils.params import *


# class DepthWiseBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, prelu=False):
#         super(DepthWiseBlock, self).__init__()
#         inplanes, planes = int(inplanes), int(planes)
#         self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
#                                  bias=False)
#         self.bn_dw = nn.BatchNorm2d(inplanes)
#         self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn_sep = nn.BatchNorm2d(planes)
#         if prelu:
#             self.relu = nn.PReLU()
#         else:
#             self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.conv_dw(x)
#         out = self.bn_dw(out)
#         out = self.relu(out)

#         out = self.conv_sep(out)
#         out = self.bn_sep(out)
#         out = self.relu(out)

#         return out


class SimpleRegressionNet(nn.Module):
    def __init__(self, param_classes=343, input_dim=107127):
    # def __init__(self, widen_factor=1.0, param_classes=192, lm_classes=136, prelu=False, input_channel=3):
        super(SimpleRegressionNet, self).__init__()

        # block = DepthWiseBlock
        # self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
        #                        bias=False)

        # self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        # if prelu:
        #     self.relu = nn.PReLU()
        # else:
        #     self.relu = nn.ReLU(inplace=True)

        # self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        # self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        # self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        # self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        # self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        # self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        # self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        # self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        # self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        # self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        # self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        # self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        # self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.relu = nn.ReLU(inplace=True)   
        self.fc1 = nn.Linear(107127, param_classes*2)
        # self.fc2 = nn.Linear(param_classes*3, param_classes*2)
        self.fc2 = nn.Linear(param_classes*2, param_classes)
        # self.fc1 = nn.Linear(107127, param_classes*3)
        # self.fc2 = nn.Linear(param_classes*3, param_classes*2)
        # self.fc3 = nn.Linear(param_classes*2, param_classes)
        

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.dw2_1(x)
        # x = self.dw2_2(x)
        # x = self.dw3_1(x)
        # x = self.dw3_2(x)
        # x = self.dw4_1(x)
        # x = self.dw4_2(x)
        # x = self.dw5_1(x)
        # x = self.dw5_2(x)
        # x = self.dw5_3(x)
        # x = self.dw5_4(x)
        # x = self.dw5_5(x)
        # x = self.dw5_6(x)
        # x = self.dw6(x)

        # x = self.avgpool(x)
        
        # x = x.contiguous().view(x.size(0), -1) # N X 107127
        # x = self.fc1(x) # N X 1029
        # x = self.relu(x) # N X 1029
        # x = self.fc2(x) # N X 686
        # x = self.relu(x)
        # param = self.fc3(x)

        x = x.contiguous().view(x.size(0), -1) # N X 107127
        x = self.fc1(x) # N X 1029
        x = self.relu(x) # N X 1029
        param = self.fc2(x) # N X 686

        return param

if __name__ == '__main__':
    model = SimpleRegressionNet(param_classes=343, input_dim=107127)

    filelists_train ='train.configs/train_aug_120x120.list.train'
    filelists_val = 'train.configs/train_aug_120x120.list.val'
    root ='../Datasets/train_aug_120x120/'
    param_fp_train = 'train.configs/param_all_full.pkl'
    param_fp_val ='train.configs/param_all_val_full.pkl'

    train_dataset = DDFADataset(
        root=root,
        filelists=filelists_train,
        param_fp=param_fp_train,
        transform=transforms.Compose([ToTensorGjz()])
    )
    val_dataset = DDFADataset(
        root=root,
        filelists=filelists_val,
        param_fp=param_fp_val,
        transform=transforms.Compose([ToTensorGjz()])
    )

    param = train_dataset[0][1]
    params = param.repeat(2,1).float()
    batch = params.shape[0]
    p, offset, alpha_shp, alpha_exp = _parse_param_batch(params)
    u_ = torch.from_numpy(u_)#.double()
    w_shp_ = torch.from_numpy(w_shp_)#.double()
    w_exp_ = torch.from_numpy(w_exp_)#.double()
    gt_vert = (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) # n x 3 x 35709

    # transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    # x = cv2.imread('samples/outputs/300w-lp/LFPWFlip_LFPW_image_train_0812_1_1_gt.jpg')
    # x = transform(x).unsqueeze(0)
    # y = model(x)

    y = model(gt_vert)
