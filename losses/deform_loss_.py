#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

from bernstein_ffd.ffd_utils import *
from utils.params import *
from utils.ddfa import _parse_param_batch, _parse_full_param_batch
from utils.io import _numpy_to_cuda, _tensor_to_cuda


_to_tensor = _numpy_to_cuda

class DeformVDCLoss(nn.Module):
    def __init__(self):
        super(DeformVDCLoss, self).__init__()

        self.u = _to_tensor(u_)# .double()
        # self.param_mean = _to_tensor(param_mean)
        # self.param_std = _to_tensor(param_std)
        self.w_shp = _to_tensor(w_shp_c)# .double()
        self.w_exp = _to_tensor(w_exp_c)# .double()

        self.deform_matrix = _to_tensor(deform_matrix_)
        self.control_points = _to_tensor(control_points_)
    
    def reconstruct_mesh(self, param, batch):
        # param = param * self.param_std + self.param_mean
        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)

        # parse param
        # p, offset, alpha_shp, alpha_exp = _parse_full_param_batch(param)

        target_vert = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

        return target_vert


    def deform_mesh(self, param, batch):
        deform = param.view(batch, cp_num_//3, -1) # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).type(torch.float32)

        return deformed_vert


    def forward(self, input, target, z_shift=False):
        # input: deformation of control points
        # target: GT vertex
        loss = nn.MSELoss()

        N = target.shape[0]

        target_vert = self.reconstruct_mesh(target, N)
        deformed_vert = self.deform_mesh(input, N)
        
        # todo - shift z to start from 0
        if z_shift:
            for i in range(target.shape[0]):
                target_vert[i,:, 2] -= target_vert[i,:, 2].min()
                deformed_vert[i,:, 2] -= deformed_vert[i,:, 2].min()

        deform_loss = loss(deformed_vert,target_vert)
        # deform_loss = torch.sqrt(deform_loss) # add sqrt v (RMSE)

        return deform_loss


# class ChamferLoss(nn.Module):
#     def __init__(self):
#         super(ChamferLoss, self).__init__()

#         self.u = _to_tensor(u_)
#         self.param_mean = _to_tensor(param_mean)
#         self.param_std = _to_tensor(param_std)
#         self.w_shp = _to_tensor(w_shp_)
#         self.w_exp = _to_tensor(w_exp_)

#         self.deform_matrix = _to_tensor(deform_matrix)
#         self.control_points = _to_tensor(control_points)


#     def forward(self, input, target):
#         # input: deformation of control points
#         # target: GT vertex
#         # loss = nn.MSELoss()

#         N = target.shape[0]
#         # rewhiten param
#         param_gt = target * self.param_std + self.param_mean
#         # parse param
#         p, offset, alpha_shp, alpha_exp = _parse_param_batch(param_gt)
#         target_vert = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).view(N, -1, 3).permute(0, 2, 1) + offset
#         target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

#         # todo - shift z to start from 0
        
#         deform = input.view(N, cp_num//3, -1) # reshape to 3d
#         deformed_vert = (self.deform_matrix @ (self.control_points + deform)).type(torch.float32)
        
#         chamfer_loss = chamfer_distance_with_batch(_tensor_to_cuda(deformed_vert), _tensor_to_cuda(target_vert))
#         # chamfer_loss = torch.sqrt(chamfer_loss) # add sqrt v

#         return chamfer_loss



if __name__ == '__main__':
    pass
