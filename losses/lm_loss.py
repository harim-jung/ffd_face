#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

from bernstein_ffd.ffd_utils import *
from utils.params import *
from utils.ddfa import _parse_param_batch, _parse_full_param_batch
from utils.io import _numpy_to_cuda, _tensor_to_cuda

_to_tensor = _numpy_to_cuda


class LMLoss(nn.Module):
    """Input and target are all 136-d param"""

    def __init__(self):
        super(LMLoss, self).__init__()

    def forward(self, input, target):
        loss = nn.MSELoss()
        lms_loss = loss(input, target)
        # lms_loss = torch.sqrt(loss(input, target))

        return lms_loss


class LMFittedLoss(nn.Module):
    """Input and target are all 136-d param"""

    def __init__(self):
        super(LMFittedLoss, self).__init__()

        self.keypoints = keypoints_

    def get_lms(self, vert):
        lms = vert[:, self.keypoints, :] # Nx68x3

        return lms

    def forward(self, input, target, z_shift=True):
        loss = nn.MSELoss()

        # N = target.shape[0]

        # target_lm = self.reconstruct_lm(target, N, z_shift=z_shift)
        # deformed_lm = self.deformed_lm(input, N)

        target_lm = self.get_lms(target)
        deformed_lm = self.get_lms(input)

        lms_loss = loss(deformed_lm, target_lm)

        return lms_loss


class LML1Loss(nn.Module):
    """Input and target are all 136-d param"""

    def __init__(self):
        super(LML1Loss, self).__init__()

        self.keypoints = keypoints_

    def get_lms(self, vert):
        lms = vert[:, self.keypoints, :] # Nx68x3

        return lms

    def forward(self, input, target, z_shift=True):
        loss = nn.L1Loss()

        target_lm = self.get_lms(target)
        deformed_lm = self.get_lms(input)

        lms_loss = loss(deformed_lm, target_lm)

        return lms_loss


class LMFittedLoss_(nn.Module):
    """Input and target are all 136-d param"""

    def __init__(self):
        super(LMFittedLoss_, self).__init__()

        self.u_c = _to_tensor(u_base_c)# .double()
        self.u_lp = _to_tensor(u_base_)

        self.w_shp_c = _to_tensor(w_shp_base_c)# .double()
        self.w_exp_c = _to_tensor(w_exp_base_c)# .double()
        self.w_shp_lp = _to_tensor(w_shp_base_lp)
        self.w_exp_lp = _to_tensor(w_exp_base_lp)

        self.deform_matrix = _to_tensor(deform_matrix_)
        self.control_points = _to_tensor(control_points_)


    def reconstruct_lm(self, param, batch, z_shift=True):
        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)

        # parse param
        if param.shape[1] == 62:
            target_lm = p @ (self.u_lp + self.w_shp_lp @ alpha_shp + self.w_exp_lp @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        else:
            target_lm = p @ (self.u_c + self.w_shp_c @ alpha_shp + self.w_exp_c @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        
        target_lm = target_lm.permute(0, 2, 1).type(torch.float32)

        if z_shift:
            for i in range(target_lm.shape[0]):
                target_lm[i,:,2] -= target_lm[i,:,2].min()

        return target_lm


    def deformed_lm(self, param, batch):
        deform = param.view(batch, cp_num_//3, -1) # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).type(torch.float32)

        lms = deformed_vert[:, keypoints_, :] # Nx68x3

        return lms


    def forward(self, input, target, z_shift=True):
        loss = nn.MSELoss()

        N = target.shape[0]

        target_lm = self.reconstruct_lm(target, N, z_shift=z_shift)
        deformed_lm = self.deformed_lm(input, N)

        lms_loss = loss(deformed_lm, target_lm)

        return lms_loss


if __name__ == '__main__':
    pass
