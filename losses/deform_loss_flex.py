   #!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
 
from bernstein_ffd.ffd_utils import *
from utils.params import *
from utils.ddfa import _parse_param_batch, _parse_full_param_batch
from utils.io import _numpy_to_cuda, _tensor_to_cuda


_to_tensor = _numpy_to_cuda


class VertexOutput(nn.Module):
    def __init__(self):
        super(VertexOutput, self).__init__()

        # self.param_mean = _to_tensor(param_mean)
        # self.param_std = _to_tensor(param_std)
        self.u_c = _to_tensor(u_c)# .double()
        self.u_lp = _to_tensor(u_)

        self.w_shp_c = _to_tensor(w_shp_c)# .double()
        self.w_exp_c = _to_tensor(w_exp_c)# .double()
        self.w_shp_lp = _to_tensor(w_shp_lp)
        self.w_exp_lp = _to_tensor(w_exp_lp)

        self.deform_matrix = _to_tensor(deform_matrix_)
        self.control_points = _to_tensor(control_points_)
    
    def reconstruct_mesh(self, param, batch, z_shift=False):
        # param = param * self.param_std + self.param_mean
        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)

        # parse param
        # p, offset, alpha_shp, alpha_exp = _parse_full_param_batch(param)
        if param.shape[1] == 62:
            target_vert = p @ (self.u_lp + self.w_shp_lp @ alpha_shp + self.w_exp_lp @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        else:
            target_vert = p @ (self.u_c + self.w_shp_c @ alpha_shp + self.w_exp_c @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        
        target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

        if z_shift:
            for i in range(target_vert.shape[0]):
                target_vert[i,:,2] -= target_vert[i,:,2].min()

        return target_vert


    def deform_mesh(self, param, batch):
        deform = param.view(batch, cp_num_//3, -1) # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).type(torch.float32)

        return deformed_vert


    def forward(self, input, target, z_shift=True):
        # input: deformation of control points
        # target: GT vertex
        # loss = nn.MSELoss()

        N = target.shape[0]

        target_vert = self.reconstruct_mesh(target, N, z_shift=z_shift)
        deformed_vert = self.deform_mesh(input, N)

        # deform_loss = loss(deformed_vert,target_vert)
        # deform_loss = torch.sqrt(deform_loss) # add sqrt v (RMSE)

        return target_vert, deformed_vert


class DeformVDCLoss(nn.Module):
    def __init__(self):
        super(DeformVDCLoss, self).__init__()

    def forward(self, input, target, z_shift=True):
        loss = nn.MSELoss()

        deform_loss = loss(input, target)

        return deform_loss


class RegionVDCLoss(nn.Module):
    def __init__(self):
        super(RegionVDCLoss, self).__init__()

        self.mouth_index = mouth_index
        self.eye_index = eye_index
        self.all_index = np.arange(0, 35709)
        self.region_index = np.append(mouth_index, eye_index)
        self.rest_index = np.delete(self.all_index, self.region_index)

    def mouth_region(self, vert):
        lms = vert[:, mouth_index, :]
        return lms
    
    def eye_region(self, vert):
        lms = vert[:, eye_index, :]
        return lms

    def rest_region(self, vert):
        lms = vert[:, self.rest_index, :]
        return lms

    def forward(self, input, target, z_shift=True):
        loss = nn.L1Loss()

        mouth_input = self.mouth_region(input)
        mouth_target = self.mouth_region(target)
        eye_input = self.eye_region(input)
        eye_target = self.eye_region(target)
        rest_input = self.rest_region(input)
        rest_target = self.rest_region(target)

        mouth_loss = loss(mouth_input, mouth_target)
        eye_loss = loss(eye_input, eye_target)
        rest_loss = loss(rest_input, rest_target)

        return mouth_loss, eye_loss, rest_loss



class MouthLoss(nn.Module):
    def __init__(self):
        super(MouthLoss, self).__init__()

        # self.param_mean = _to_tensor(param_mean)
        # self.param_std = _to_tensor(param_std)
        self.u_c = _to_tensor(u_c)# .double()
        self.u_lp = _to_tensor(u_)

        self.w_shp_c = _to_tensor(w_shp_c)# .double()
        self.w_exp_c = _to_tensor(w_exp_c)# .double()
        self.w_shp_lp = _to_tensor(w_shp_lp)
        self.w_exp_lp = _to_tensor(w_exp_lp)

        self.deform_matrix = _to_tensor(deform_matrix_)
        self.control_points = _to_tensor(control_points_)
    
    def reconstruct_mesh(self, param, batch, z_shift=True):
        # param = param * self.param_std + self.param_mean
        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)

        # parse param
        # p, offset, alpha_shp, alpha_exp = _parse_full_param_batch(param)
        if param.shape[1] == 62:
            target_vert = p @ (self.u_lp + self.w_shp_lp @ alpha_shp + self.w_exp_lp @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        else:
            target_vert = p @ (self.u_c + self.w_shp_c @ alpha_shp + self.w_exp_c @ alpha_exp).view(batch, -1, 3).permute(0, 2, 1) + offset
        
        target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

        if z_shift:
            for i in range(target_vert.shape[0]):
                target_vert[i,:,2] -= target_vert[i,:,2].min()

        # mouth index
        lms = target_vert[:, mouth_index, :]

        return lms


    def deform_mesh(self, param, batch):
        deform = param.view(batch, cp_num_//3, -1) # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).type(torch.float32)

        lms = deformed_vert[:, mouth_index, :]

        return lms


    def forward(self, input, target, z_shift=True):
        # input: deformation of control points
        # target: GT vertex
        loss = nn.MSELoss()

        N = target.shape[0]

        target_vert = self.reconstruct_mesh(target, N, z_shift=z_shift)
        deformed_vert = self.deform_mesh(input, N)

        lms_loss = loss(deformed_vert,target_vert)
        # deform_loss = torch.sqrt(deform_loss) # add sqrt v (RMSE)

        return lms_loss

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
