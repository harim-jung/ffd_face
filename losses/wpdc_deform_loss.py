#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from math import sqrt
from utils.io import _numpy_to_cuda
from utils.params import *
from utils.ddfa import _parse_param_batch, get_rot_mat_from_axis_angle_batch, get_axis_angle_s_t_from_rot_mat_batch

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
        input = self.param_std[:12] * input + self.param_mean[:12] # input: N x 12
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


class PDCAxisAngleLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(PDCAxisAngleLoss, self).__init__()
        self.opt_style = opt_style
        self.param_mean = _to_tensor(param_full_mean)
        self.param_std = _to_tensor(param_full_std)

        self.u = _to_tensor(u_).double()
        self.w_shp = _to_tensor(w_shp_).double()
        self.w_exp = _to_tensor(w_exp_).double()

    def calc_diff(self, input_pose, target_pose):
        # freeze only for calcualting the weights
        # input_pose = torch.tensor(input_.data.clone(), requires_grad=False)
        # target_pose = torch.tensor(target_.data.clone(), requires_grad=False)

        N = input_pose.shape[0]

        pg = (self.param_std[:12] * target_pose + self.param_mean[:12]).view(N, -1, 1) # N x 12 x1
        pg_ = target_pose[:, :12].view(N, 3, -1) # N x 3 x 4
        rotg = pg_[:, :, :3] # N x 3 x 3
        offsetg = pg_[:, :, -1].view(N, 3, 1) # the 4th column

        s = torch.abs(input_pose[:, 0]).view(N, 1) # N x 1 from input_pose: N x 7:  s, axis_angle, offset
        axis_angle = input_pose[:, 1:4]
        offset = input_pose[:, 4:].view(N, 3, 1)
        rot_mat = get_rot_mat_from_axis_angle_batch(axis_angle) # N x 3 x 3
        rot = (torch.einsum('ab,acd->acd', s, rot_mat))

        p = torch.cat((rot, offset), 2).view(N, -1, 1)

        return p - pg
    
    def forward(self, input, target):
        # input, target --> pose parameter
        loss = self.calc_diff(input, target) ** 2 # MSE
        return loss.mean()


class WPDCAxisAngleLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(WPDCAxisAngleLoss, self).__init__()
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
        #input = the predicted parameters; weights has the same shape as input, N x 62 matrix, where N is the number of samples.

        # V3d(p^g) of equation 7
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        # calculate weights
        for i in range(weights.shape[1]):
            # V3d(p^de,i) of equation 7
            p_degraded = target[:, :12]

            # assume that p_degraded_axis_angle_s_t has 7 elements in the order of s, axis_angle, t
            p_degraded_axis_angle_s_t = get_axis_angle_s_t_from_rot_mat_batch(p_degraded)  
            
            # degrade the target pose by input
            p_degraded_axis_angle_s_t[:,i] = input[:,i]

            s = torch.abs(p_degraded_axis_angle_s_t[:, 0]).view(N, 1) # N x 1 from input_pose: N x 7:  s, axis_angle, offset
            axis_angle = p_degraded_axis_angle_s_t[:, 1:4]
            offset = p_degraded_axis_angle_s_t[:, 4:].view(N, 3, 1)
            rot_mat = get_rot_mat_from_axis_angle_batch(axis_angle) # N x 3 x 3
            p = (torch.einsum('ab,acd->acd', s, rot_mat))
           
            pred_vertex = p @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

            weights[:, i] = torch.norm(pred_vertex - gt_vertex, dim = (1,2))

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
        # loss = self.calc_diff(input, target[:, :12]) ** 2 # MSE

        return loss.mean()


if __name__ == '__main__':
    pass
