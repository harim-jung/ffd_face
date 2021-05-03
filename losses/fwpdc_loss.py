#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from math import sqrt
from utils.io import _numpy_to_cuda
from utils.params import *
from utils.ddfa import _parse_param_batch, get_rot_mat_from_axis_angle_batch, get_axis_angle_s_t_from_rot_mat_batch

_to_tensor = _numpy_to_cuda  # gpu

class FWPDCAxisAngleLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(FWPDCAxisAngleLoss, self).__init__()
        self.opt_style = opt_style
        self.param_mean = _to_tensor(param_full_mean)
        self.param_std = _to_tensor(param_full_std)

        self.u = _to_tensor(u_).double()
        self.w_shp = _to_tensor(w_shp_).double()
        self.w_exp = _to_tensor(w_exp_).double()

    def calc_weights(self, input_, target_):
        # freeze only for calcualting the weights
        input = torch.tensor(input_.data.clone(), requires_grad=False)
        target = torch.tensor(target_.data.clone(), requires_grad=False)

        # rewhiten
        target = self.param_std * target + self.param_mean

        (pg, offsetg, alpha_shpg, alpha_expg) = _parse_param_batch(target)

        N = input.shape[0] 

        weights = torch.zeros((N, 7), dtype=torch.float).cuda()

        target_axis_angle_s_t = get_axis_angle_s_t_from_rot_mat_batch(target[:, :12]) # s, axis_angle, t
        tmpv = (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg).view(N, -1, 3).permute(0, 2, 1)

        tmpv_norm = torch.norm(tmpv, dim=2) # 64 x 3
        tmpv_norm_all = torch.norm(tmpv, dim=(1,2)) # 64 x 1
        offset_norm = sqrt(self.w_shp.shape[0] // 3)

        # for pose 
        param_diff_pose = torch.abs(self.calc_diff(input, target[:, :12])) # N x 7

        # w_s = ( s - s^g) * |S| 
        weights[:, 0] = param_diff_pose[:, 0] * tmpv_norm_all  # scale
        weights[:, 1] = param_diff_pose[:, 1] * tmpv_norm[:, 0] # axis angle x
        weights[:, 2] = param_diff_pose[:, 2] * tmpv_norm[:, 1] # axis angle y
        weights[:, 3] = param_diff_pose[:, 3] * tmpv_norm[:, 2] # axis angle z
        weights[:, 4] = param_diff_pose[:, 4] * offset_norm
        weights[:, 5] = param_diff_pose[:, 5] * offset_norm

        eps = 1e-6
        weights[:, :] += eps

        maxes, _ = weights.max(dim=1)
        maxes = maxes.view(-1, 1)
        weights /= maxes

        # zero the z
        weights[:, 6] = 0

        return weights
        

    def calc_diff(self, input_pose, target_pose):
        N = input_pose.shape[0]
        pg = self.param_std[:12] * target_pose + self.param_mean[:12] # N x 12 x 1
        pg_axis_angle = get_axis_angle_s_t_from_rot_mat_batch(pg)
        # scale
        p_axis_angle = torch.cat((torch.abs(input_pose[:, 0].view(N, 1)), input_pose[:, 1:]), 1)
        # p_axis_angle = input_pose
        # p_axis_angle[:, 0] = torch.abs(p_axis_angle[:, 0])

        return p_axis_angle - pg_axis_angle
    
    def forward(self, input, target):
        # input, target --> pose parameter
        weights = self.calc_weights(input, target) # N x 7
        loss = weights * self.calc_diff(input, target[:, :12]) ** 2 # MSE

        return loss.mean()


if __name__ == '__main__':
    pass
