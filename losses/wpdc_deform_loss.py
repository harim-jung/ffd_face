#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from math import sqrt
from utils.io import _numpy_to_cuda
from utils.params import *
from utils.ddfa import _parse_param_batch

_to_tensor = _numpy_to_cuda  # gpu


class WPDCPoseLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(WPDCPoseLoss, self).__init__()
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
        input = self.param_std[:12] * input + self.param_mean[:12]
        target = self.param_std * target + self.param_mean

        (pg, offsetg, alpha_shpg, alpha_expg) = _parse_param_batch(target)

        # input has shape N x 62, the predicted parameters for N samples in the current batch
        # p refers to the rotation matrix part from the pose matrix
        # offset refers to the last column of the pose matrix, which is the translation part of the pose matrix
        # the pose transform T = f[R; t3d], 3 x 4 matrix

        N = input.shape[0] # input is the predicted output of the network, which has the form of  N x 62 matrix;
                           # N = the number of samples (the size of the mini-batch);
                           # Note that this N is NOT the same as N in Alogorithm 2, where N refers to the number of vertices to be used
                           # to compute the weights for the 62 parameters.
                           # This number of vertices  is represented by w_shp_base.shape[0] // 3 in this code. See below.

        weights = torch.zeros((N, 12), dtype=torch.float).cuda()
        #input = the predicted parameters; weights has the same shape as input, N x 62 matrix, where N is the number of samples.

        # V3d(p^g) of equation 7
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        # calculate weights
        for i in range(weights.shape[1]):
            # V3d(p^de,i) of equation 7
            p_degraded = target[:, :12]
            p_degraded[:, i] = input[:, i]

            p_ = p_degraded[:, :12].view(N, 3, -1)
            p = p_[:, :, :3]
            offset = p_[:, :, -1].view(N, 3, 1)

            pred_vertex = p @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

            weights[:, i] = torch.norm(pred_vertex - gt_vertex, dim = (1,2))
            # weights[:, i] = torch.norm(pred_vertex - gt_vertex)#, dim=2)
        
        eps = 1e-6
        weights[:, :11] += eps
        weights[:, 12:] += eps

        maxes, _ = weights.max(dim=1)
        maxes = maxes.view(-1, 1)
        weights /= maxes

        return weights

    def forward(self, input, target):
        # input, target --> pose parameter
        weights = self.calc_weights(input, target)
        loss = weights * (input - target[:, :12]) ** 2 # MSE
        return loss.mean()


if __name__ == '__main__':
    pass
