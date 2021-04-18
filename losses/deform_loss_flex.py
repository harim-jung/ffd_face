   #!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
 
from bernstein_ffd.ffd_utils import *
from utils.params import *
from utils.ddfa import _parse_param_batch, _parse_full_param_batch, get_rot_mat_from_axis_angle_batch
from utils.io import _numpy_to_cuda, _tensor_to_cuda


_to_tensor = _numpy_to_cuda


class SimpleVertex(nn.Module):
    def __init__(self):
        super(SimpleVertex, self).__init__()

        self.deform_matrix = _to_tensor(deform_matrix).double()
        self.control_points = _to_tensor(control_points).double()
    
    def deform_mesh_no_pose(self, param, batch):
        deform = param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1)

        return deformed_vert.permute(0, 2, 1).type(torch.float32)

    def forward(self, input, target, z_shift=False):
        N = target.shape[0]

        target_vert = target
        deformed_vert = self.deform_mesh_no_pose(input, N) # delta p only without pose param

        return target_vert, deformed_vert


class VertexOutput(nn.Module):
    def __init__(self):
        super(VertexOutput, self).__init__()

        # self.param_mean = _to_tensor(param_mean)
        # self.param_std = _to_tensor(param_std)
        # self.u_c = _to_tensor(u_c)# .double()
        # self.u_lp = _to_tensor(u_)

        # self.w_shp_c = _to_tensor(w_shp_c)# .double()
        # self.w_exp_c = _to_tensor(w_exp_c)# .double()
        # self.w_shp_lp = _to_tensor(w_shp_lp)
        # self.w_exp_lp = _to_tensor(w_exp_lp)

        # 300w-lp full params
        self.param_mean = _to_tensor(param_full_mean)
        self.param_std = _to_tensor(param_full_std)

        self.u = _to_tensor(u_).double()
        self.w_shp = _to_tensor(w_shp_).double()
        self.w_exp = _to_tensor(w_exp_).double()

        self.deform_matrix = _to_tensor(deform_matrix).double()
        self.control_points = _to_tensor(control_points).double()
    
    
    def reconstruct_mesh(self, gt_param, batch, z_shift=False):
        # parse param
        gt_param = gt_param * self.param_std + self.param_mean
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(gt_param)

        if gt_param.shape[1] == 62:
            target_vert = pg @ (self.u_lp + self.w_shp_lp @ alpha_shpg + self.w_exp_lp @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        elif gt_param.shape[1] == 240:
            # rewhiten needed
            target_vert = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        else:
            target_vert = pg @ (self.u_c + self.w_shp_c @ alpha_shpg + self.w_exp_c @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        
        target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

        if z_shift:
            for i in range(target_vert.shape[0]):
                target_vert[i,:,2] -= target_vert[i,:,2].min()

        return target_vert

    def deform_mesh_no_pose(self, param, batch):
        deform = param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1)

        return deformed_vert.permute(0, 2, 1).type(torch.float32)


    def deform_pg_mesh(self, param, gt_param, batch):
        pose_param = gt_param[:, :12]
        pose_param = self.param_std[:12] * pose_param + self.param_mean[:12]

        p_ = pose_param.view(batch, 3, -1)
        pg = p_[:, :, :3]
        offsetg = p_[:, :, -1].view(batch, 3, 1)

        deform = param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = pg @ (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1) + offsetg

        return deformed_vert.permute(0, 2, 1).type(torch.float32)


    def deform_p_mesh(self, param, batch):
        pose_param = param[:, :12].double()
        pose_param = self.param_std[:12] * pose_param + self.param_mean[:12]

        p_ = pose_param.view(batch, 3, -1)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].view(batch, 3, 1)

        deform_param = param[:, 12:]
        deform = deform_param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = p @ (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1) + offset

        return deformed_vert.permute(0, 2, 1).type(torch.float32)


    def deform_aap_mesh(self, param, batch):
        pose_param = param[:, :7].double()

        s = torch.abs(pose_param[:, 0]).view(batch, 1)
        # s = pose_param[:, 0].view(batch, 1)
        # s = torch.exp(x).view(batch, 1) # N x 1
        axis_angle = pose_param[:, 1:4]
        offset = pose_param[:, 4:].view(batch, 3, 1)
        p = get_rot_mat_from_axis_angle_batch(axis_angle) # N x 3 x 3

        deform_param = param[:, 7:]
        deform = deform_param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = (torch.einsum('ab,acd->acd', s, p)) @ (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1) + offset

        return deformed_vert.permute(0, 2, 1).type(torch.float32)


    def forward(self, input, target, z_shift=False):
        N = target.shape[0]

        target_vert = self.reconstruct_mesh(target, N, z_shift=z_shift)
        # deformed_vert = self.deform_mesh_no_pose(input, N) # delta p only without pose param
        # deformed_vert = self.deform_p_mesh(input, N) # use predicted pose
        deformed_vert = self.deform_aap_mesh(input, N) # use predicted pose with s, axis_angle, offset

        return target_vert, deformed_vert


class VertexOutputwoPose(nn.Module):
    def __init__(self):
        super(VertexOutputwoPose, self).__init__()

        # self.param_mean = _to_tensor(param_mean)
        # self.param_std = _to_tensor(param_std)
        # self.u_c = _to_tensor(u_c)# .double()
        # self.u_lp = _to_tensor(u_)

        # self.w_shp_c = _to_tensor(w_shp_c)# .double()
        # self.w_exp_c = _to_tensor(w_exp_c)# .double()
        # self.w_shp_lp = _to_tensor(w_shp_lp)
        # self.w_exp_lp = _to_tensor(w_exp_lp)

        # 300w-lp full params
        self.param_mean = _to_tensor(param_full_mean)
        self.param_std = _to_tensor(param_full_std)
        # self.param_mean = _to_tensor(param_full_mean)
        # self.param_std = _to_tensor(param_full_std)

        self.u = _to_tensor(u_).double()
        self.w_shp = _to_tensor(w_shp_).double()
        self.w_exp = _to_tensor(w_exp_).double()

        self.deform_matrix = _to_tensor(deform_matrix).double()
        self.control_points = _to_tensor(control_points).double()
    
    
    def reconstruct_mesh(self, gt_param, batch, z_shift=False):
        # parse param
        gt_param = gt_param * self.param_std + self.param_mean
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(gt_param)

        if gt_param.shape[1] == 62:
            target_vert = pg @ (self.u_lp + self.w_shp_lp @ alpha_shpg + self.w_exp_lp @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        elif gt_param.shape[1] == 240:
            # rewhiten needed
            target_vert = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        else:
            target_vert = pg @ (self.u_c + self.w_shp_c @ alpha_shpg + self.w_exp_c @ alpha_expg).view(batch, -1, 3).permute(0, 2, 1) + offsetg
        
        target_vert = target_vert.permute(0, 2, 1).type(torch.float32)

        if z_shift:
            for i in range(target_vert.shape[0]):
                target_vert[i,:,2] -= target_vert[i,:,2].min()

        return target_vert

    def deform_mesh_no_pose(self, param, batch):
        deform = param.view(batch, cp_num//3, -1).double() # reshape to 3d
        deformed_vert = (self.deform_matrix @ (self.control_points + deform)).permute(0, 2, 1)

        return deformed_vert.permute(0, 2, 1).type(torch.float32)

    def forward(self, input, target, z_shift=False):
        N = target.shape[0]

        target_vert = self.reconstruct_mesh(target, N, z_shift=z_shift)
        deformed_vert = self.deform_mesh_no_pose(input, N) # delta p only without pose param

        return target_vert, deformed_vert


class DeformVDCLoss(nn.Module):
    def __init__(self):
        super(DeformVDCLoss, self).__init__()

    def forward(self, input, target, loss_type="l1"):
        if loss_type == "l1":
            loss = nn.L1Loss()
        elif loss_type == "mse":
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


class RegionLMLoss(nn.Module):
    def __init__(self):
        super(RegionLMLoss, self).__init__()

    def forward(self, input, target, loss_type="l1"):
        if loss_type == "l1":
            loss = nn.L1Loss()
        elif loss_type == "mse":
            loss = nn.MSELoss()
        
        up_mouth = loss(input[:, upper_mouth, :], target[:, upper_mouth, :]) 
        low_mouth = loss(input[:, lower_mouth, :], target[:, lower_mouth, :]) 
        up_nose = loss(input[:, upper_nose, :], target[:, upper_nose, :]) 
        low_nose = loss(input[:, lower_nose, :], target[:, lower_nose, :]) 
        l_brow = loss(input[:, left_brow, :], target[:, left_brow, :]) 
        r_brow = loss(input[:, right_brow, :], target[:, right_brow, :]) 
        l_eye = loss(input[:, left_eye, :], target[:, left_eye, :]) 
        r_eye = loss(input[:, right_eye, :], target[:, right_eye, :]) 
        contour = loss(input[:, contour_boundary, :], target[:, contour_boundary, :]) 

        return up_mouth, low_mouth, up_nose, low_nose, l_brow, r_brow, l_eye, r_eye, contour


class MouthLoss(nn.Module):
    def __init__(self):
        super(MouthLoss, self).__init__()

    def mouth_region(self, vert):
        lms = vert[:, mouth_whole_index, :]
        return lms

    def mouth_inner(self, vert):
        lms = vert[:, mouth_index, :]
        return lms
    
    def forward(self, input, target, whole=True):
        loss = nn.L1Loss()

        if whole:
            mouth_input = self.mouth_region(input)
            mouth_target = self.mouth_region(target)
        else:
            mouth_input = self.mouth_inner(input)
            mouth_target = self.mouth_inner(target)

        mouth_loss = loss(mouth_input, mouth_target)

        return mouth_loss


class PDCLoss(nn.Module):
    def __init__(self):
        super(PDCLoss, self).__init__()

    def forward(self, input, target):
        loss = nn.L1Loss()

        pdc_loss = loss(input, target)

        return pdc_loss 

class PDCMSELoss(nn.Module):
    def __init__(self):
        super(PDCMSELoss, self).__init__()

    def forward(self, input, target):
        loss = nn.MSELoss()

        pdc_loss = loss(input, target)#.float()

        return pdc_loss 

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
