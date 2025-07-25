#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
from collections import OrderedDict

import torch
import torch.utils.data as data
import cv2
import argparse
from .io import _numpy_to_tensor, _load_cpu, _load_gpu
from .params import *
from math import cos, sin, sqrt, hypot
# from bernstein_ffd.ffd_utils import uv_map, face_contour, reference_mesh

# from scipy.spatial.transform import Rotation as R

def _parse_ffd_param(params):
    """Work for both numpy and tensor"""
    # 204 params
    p_ = params[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    delta_p = params[12:].reshape(-1, 1)

    return p, offset, delta_p

def _parse_param(params):
    """Work for both numpy and tensor"""
    if params.shape[0]==62:
        # 300w-lp-aug
        p_ = params[:12].reshape(3, -1)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = params[12:52].reshape(-1, 1)
        alpha_exp = params[52:].reshape(-1, 1)
    elif params.shape[0] == 235:
        # 300w-lp & aflw
        pose = params[:7] # pitch, yaw, roll, t3dx, t3dy, t3dz, f
        rot = pose[:3]
        R = get_rotation_matrix(rot)
        offset = pose[3:6].reshape((-1, 1))
        scale = pose[6]
        alpha_shp = params[7:206].reshape((-1, 1))
        alpha_exp = params[206:].reshape((-1, 1))
        p = R * scale
    elif params.shape[0] == 240:
        p_ = params[:12].reshape(3, -1)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = params[12:211].reshape(-1, 1)
        alpha_exp = params[211:].reshape(-1, 1)
    else:
        # coarse data
        alpha_shp = params[:100].reshape((-1, 1))
        alpha_exp = params[100:179].reshape((-1, 1))
        pose = params[179:] #  pitch, yaw, roll, t2dx, t2dy, f
        rot = pose[:3]
        R = get_rotation_matrix(rot)
        offset = np.append(pose[3:5], [0]).reshape(-1, 1)
        scale = pose[5]
        p = R * scale

    return p, offset, alpha_shp, alpha_exp


def _parse_param_batch(param):
    N = param.shape[0]
    if param.shape[1]==62:
        p_ = param[:, :12].view(N, 3, -1)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].view(N, 3, 1)
        alpha_shp = param[:, 12:52].view(N, -1, 1)
        alpha_exp = param[:, 52:].view(N, -1, 1)
    elif param.shape[1] == 240:
        p_ = param[:, :12].view(N, 3, -1)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].view(N, 3, 1)
        alpha_shp = param[:, 12:211].view(N, -1, 1)
        alpha_exp = param[:, 211:].view(N, -1, 1)
    else:
        alpha_shp = param[:, :100].view(N, -1, 1)
        alpha_exp = param[:, 100:179].view(N, -1, 1)
        pose = param[:, 179:] #  pitch, yaw, roll, t2dx, t2dy, f
        rot = pose[:, :3]
        R = get_rotation_matrix_batch(rot)
        offset = torch.cat((pose[:, 3:5], torch.zeros(N, 1).cuda()), 1).view(N, -1, 1)
        scale = pose[:, 5].view(N, -1, 1)
        p = R * scale 

    return p, offset, alpha_shp, alpha_exp



def _parse_full_param_batch(param):
    N = param.shape[0]
    pose = param[:, :7]
    rot = pose[:, :3].float()
    R = get_rotation_matrix_batch(rot).double()
    offset = pose[:, 3:6].view(N, 3, 1)
    scale = pose[:, 6].view(N, -1, 1)
    shp = param[:, 7:206].view(N, -1, 1)
    exp = param[:, 206:].view(N, -1, 1)
    return R * scale, offset, shp, exp


def _parse_full_param(params):
    """one inference with numpy"""
    pose = params[:7] # pitch, yaw, roll, t3dx, t3dy, t3dz, f
    rot = pose[:3]
    R = _numpy_to_tensor(get_rotation_matrix(rot))
    offset = pose[3:6].reshape((-1, 1)) # move reshape?
    scale = pose[6]
    shp = params[7:206].reshape((-1, 1))
    exp = params[206:].reshape((-1, 1))

    return R * scale, offset, shp, exp


def get_rotation_matrix(rot_param):
    pitch, yaw, roll = rot_param[0], rot_param[1], rot_param[2]
    # from matlab code
    Rx = np.array([[1,0,0],[0,cos(pitch),sin(pitch)],[0,-1*sin(pitch),cos(pitch)]], dtype=np.float32)
    Ry = np.array([[cos(yaw),0,-1*sin(yaw)],[0,1,0],[sin(yaw),0,cos(yaw)]], dtype=np.float32)
    Rz = np.array([[cos(roll),sin(roll),0],[-1*sin(roll),cos(roll),0],[0,0,1]], dtype=np.float32)
    R = Rx @ Ry @ Rz

    return R


def get_rotation_matrix_batch(rot_param):
    N = rot_param.shape[0]
    pitch, yaw, roll = rot_param[:, 0], rot_param[:, 1], rot_param[:,2]

    tens = torch.ones(())
    Rx = tens.new_empty((N, 3, 3)).cuda()
    Ry = tens.new_empty((N, 3, 3)).cuda()
    Rz = tens.new_empty((N, 3, 3)).cuda()

    Rx[:, 0, :] = torch.tensor((1,0,0))
    Rx[:, 1, :] = torch.cat((torch.zeros((N)).cuda(), torch.cos(pitch), torch.sin(pitch))).reshape(-1, N).T
    Rx[:, 2, :] = torch.cat((torch.zeros((N)).cuda(), -1*torch.sin(pitch), torch.cos(pitch))).reshape(-1, N).T

    Ry[:, 0, :] = torch.cat((torch.cos(yaw), torch.zeros((N)).cuda(), -1*torch.sin(yaw))).reshape(-1, N).T
    Ry[:, 1, :] = torch.tensor((0,1,0))
    Ry[:, 2, :] = torch.cat((torch.sin(yaw), torch.zeros((N)).cuda(), torch.cos(yaw))).reshape(-1, N).T

    Rz[:, 0, :] = torch.cat((torch.cos(roll), torch.sin(roll), torch.zeros((N)).cuda())).reshape(-1, N).T
    Rz[:, 1, :] = torch.cat((-1*torch.sin(roll), torch.cos(roll), torch.zeros((N)).cuda())).reshape(-1, N).T
    Rz[:, 2, :] = torch.tensor((0,0,1))

    R = Rx @ Ry @ Rz

    return R

# def get_rot_mat_from_axis_angle_(axis_angle):
#     # rotation = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
#     rotation = R.from_rotvec(axis_angle)

#     return rotation.as_matrix()

def get_rot_mat_from_axis_angle(r):
    r = torch.from_numpy(r)
    theta = torch.norm(r)
    r_hat = (r / theta).view(-1, 1) # 3x1

    r_hat_x = torch.tensor([[0, -r_hat[2], r_hat[1]],
                            [r_hat[2], 0, -r_hat[0]],
                            [-r_hat[1], r_hat[0], 0]])
    R = torch.cos(theta) * torch.eye(3) + torch.sin(theta) * r_hat_x + (1-torch.cos(theta)) * (r_hat @ r_hat.T)

    return R

def get_rot_mat_from_axis_angle_np(r):
    theta = np.linalg.norm(r)
    r_hat = (r / theta).reshape(-1, 1) # 3x1

    r_hat_x = np.array([[0, -r_hat[2][0], r_hat[1][0]],
                        [r_hat[2][0], 0, -r_hat[0][0]],
                        [-r_hat[1][0], r_hat[0][0], 0]])
    R = np.cos(theta) * np.eye(3) + np.sin(theta) * r_hat_x + (1-np.cos(theta)) * (r_hat @ r_hat.T)

    return R

def get_rot_mat_from_axis_angle_batch(r):
    # r -> N x 3
    N = r.shape[0]
    theta = torch.norm(r, dim=1).view(-1, 1) # N x 1
    
    # change 0 to 1 for division
    temp_theta = torch.zeros_like(theta)
    for i, t in enumerate(theta):
        if t == 0:
            temp_theta[i] = 1
        else:
            temp_theta[i] = theta[i]
    
    # r_hat = r / theta # N x 3
    r_hat = r / temp_theta # N x 3

    r_hat_x = torch.zeros((N, 3, 3)).cuda() # N x 3 x 3
    r_hat_x[:, 0, 1] = -r_hat[:, 2] 
    r_hat_x[:, 0, 2] = r_hat[:, 1]
    r_hat_x[:, 1, 0] = r_hat[:, 2]
    r_hat_x[:, 1, 2] = -r_hat[:, 0]
    r_hat_x[:, 2, 0] = -r_hat[:, 1]
    r_hat_x[:, 2, 1] = r_hat[:, 0]

    # N x 3 x 3
    R = torch.einsum('ab,acd->acd', torch.cos(theta), torch.eye(3).repeat(N, 1, 1).cuda()) + \
        torch.einsum('ab,acd->acd', torch.sin(theta), r_hat_x) + \
        torch.einsum('ab,acd->acd', (1-torch.cos(theta)), (r_hat.view(N, -1, 1) @ r_hat.view(N, -1, 1).permute(0,2,1)))

    return R


def get_axis_angle_s_t_from_rot_mat_batch(pose_param):
    # assume that p_degraded_axis_angle_s_t has 7 elements in the order of axis_angle, s, t
    # when you convert p_degraded (= target[:,12] ) to p_degraded_axis_angle_s_t, do:
    N = pose_param.shape[0]
    p_ = pose_param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3] # p is s * 3x3 rotation matrix
    offset = p_[:, :, -1]
    # Gt fR= [f * r1 f * r2 f * r3]이고 |f * r1| = f
    # s = p[:,:,0] / torch.norm(p[:,:,0]) 
    # s = s.view(N, -1, 1)
    s = torch.norm(p[:,:,0], dim=1)
    rot_mat = p / s.view(N, -1, 1) # pure rotation without scale multiplied

    # assert
    # r_i norm == 1
    # print(torch.norm(rot_mat[:,:,0], dim=1))
    # print(torch.norm(rot_mat[:,:,1], dim=1))
    # print(torch.norm(rot_mat[:,:,2], dim=1))
    # # r_1 dot r_2, r_2 dot r_3, r_3 dot r_1 == 0
    # print(torch.dot(rot_mat[1, 0], rot_mat[1, 1]))
    # print(torch.dot(rot_mat[1, 2], rot_mat[1, 0]))
    # print(torch.dot(rot_mat[1, 1], rot_mat[1, 2]))

    axis_angle = get_axis_angle_from_rot_mat_batch(rot_mat)

    return torch.cat((s.view(N, 1), axis_angle, offset), 1)


def get_axis_angle_from_rot_mat(rot_mat):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """
    # Axes.
    axis = np.zeros(3, dtype=np.float64)
    axis[0] = rot_mat[2,1] - rot_mat[1,2]
    axis[1] = rot_mat[0,2] - rot_mat[2,0]
    axis[2] = rot_mat[1,0] - rot_mat[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = rot_mat[0,0] + rot_mat[1,1] + rot_mat[2,2] # the sum of the diagonal elements of the rotation matrix
    theta = np.arctan2(r, t-1)


    # Normalise the axis.
    if r != 0:
        axis = axis / r

    # multiply axis and angle
    axis_angle = axis * theta

    return axis_angle

def get_axis_angle_from_rot_mat_batch(rot_mat):
    # rot_mat --> (N, 3, 3) torch tensor
    N = rot_mat.shape[0]
    # Axes.
    axis = torch.zeros((N, 3)).cuda() #, dtype=np.float64)
    axis[:, 0] = rot_mat[:, 2,1] - rot_mat[:, 1,2]
    axis[:, 1] = rot_mat[:, 0,2] - rot_mat[:, 2,0]
    axis[:, 2] = rot_mat[:, 1,0] - rot_mat[:, 0,1]

    # Angle.
    r = torch.hypot(axis[:, 0], torch.hypot(axis[:, 1], axis[:, 2]))
    t = (rot_mat[:, 0,0] + rot_mat[:, 1,1] + rot_mat[:, 2,2]) # the sum of the diagonal elements of the rotation matrix
    theta = torch.atan2(r, t-1).view(-1, 1)

    # Normalise the axis.
    axis = axis / r.view(-1, 1)

    # multiply axis and angle
    axis_angle = axis * theta

    return axis_angle

# def get_rot_mat_from_axis_angle_batch_(axis_angle, s):
#     # rotation = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
#     N = axis_angle.shape[0]
#     rotation = torch.zeros((N, 3, 3))
#     for n in range(N):
#         rotation[n] = s[n].cpu() * torch.from_numpy(R.from_rotvec(axis_angle[n].detach().cpu()).as_matrix())
    
#     return rotation.double().cuda()


# def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
#     """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
#     dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
#     image coordinate space, but without alignment caused by face cropping.
#     transform: whether transform to image space
#     """
#     if len(param) == 12:
#         param = np.concatenate((param, [0] * 50))
#     if whitening:
#         if len(param) == 62:
#             param = param * param_std + param_mean
#         else:
#             param = np.concatenate((param[:11], [0], param[11:]))
#             param = param * param_std + param_mean

#     p, offset, alpha_shp, alpha_exp = _parse_param(param)

#     if dense:
#         vertex = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset

#         if transform:
#             # transform to image coordinate space
#             vertex[1, :] = std_size + 1 - vertex[1, :]
#     else:
#         """For 68 pts"""
#         vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

#         if transform:
#             # transform to image coordinate space
#             vertex[1, :] = std_size + 1 - vertex[1, :]

#     return vertex


def reconstruct_vertex(param, dense=True, face=True, transform=True, std_size=std_size):
    """
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    face: if True, only face region without ears and neck
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if param.shape[0] == 62:
        # 300w-lp
        param = param * param_std + param_mean
    
    # p, offset, alpha_shp, alpha_exp = _parse_full_param(param)
    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        if param.shape[0] == 62:
            vertex = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
        elif param.shape[0] == 235:
            vertex = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
        else:
            vertex = p @ (u_c + w_shp_c @ alpha_shp + w_exp_c @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        if face:
            if param.shape[0] == 62:
                vertex = p @ (u_base_ + w_shp_base_lp @ alpha_shp + w_exp_base_lp @ alpha_exp).reshape(3, -1, order='F') + offset
            elif param.shape[0] == 235:
                vertex = p @ (u_base_ + w_shp_base_ @ alpha_shp + w_exp_base_ @ alpha_exp).reshape(3, -1, order='F') + offset
            else:
                vertex = p @ (u_base_ + w_shp_base_c @ alpha_shp + w_exp_base_c @ alpha_exp).reshape(3, -1, order='F') + offset
        else:
            # vertex = p @ (u_base + w_shp_base_ @ alpha_shp + w_exp_base_ @ alpha_exp).reshape(3, -1, order='F') + offset
            vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex

def reconstruct_vertex_full(param, dense=True, face=True, transform=True, std_size=std_size):
    """
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    face: if True, only face region without ears and neck
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if param.shape[0] == 62:
        # 300w-lp
        param = param * param_std + param_mean
    
    p, offset, alpha_shp, alpha_exp = _parse_full_param(param)
    # p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        if param.shape[0] == 62:
            vertex = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
        elif param.shape[0] == 235:
            vertex = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
        else:
            vertex = p @ (u_c + w_shp_c @ alpha_shp + w_exp_c @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        if face:
            if param.shape[0] == 62:
                vertex = p @ (u_base_ + w_shp_base_lp @ alpha_shp + w_exp_base_lp @ alpha_exp).reshape(3, -1, order='F') + offset
            elif param.shape[0] == 235:
                vertex = p @ (u_base_ + w_shp_base_ @ alpha_shp + w_exp_base_ @ alpha_exp).reshape(3, -1, order='F') + offset
            else:
                vertex = p @ (u_base_ + w_shp_base_c @ alpha_shp + w_exp_base_c @ alpha_exp).reshape(3, -1, order='F') + offset
        else:
            vertex = p @ (u_base + w_shp_base_ @ alpha_shp + w_exp_base_ @ alpha_exp).reshape(3, -1, order='F') + offset
            # vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
            # vertex[0, :] -= 1  # for Python compatibility
            # vertex[2, :] -= 1
            # vertex[1, :] = std_size - vertex[1, :]

    return vertex

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']
    elif "model_state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['model_state_dict']

    # if "module.fc1.weight" in pretrained_dict:
    #     pretrained_dict = OrderedDict([('module.fc.weight', v) if k == 'module.fc1.weight' else (k, v) for k, v in pretrained_dict.items()])
    #     pretrained_dict = OrderedDict([('module.fc.bias', v) if k == 'module.fc1.bias' else (k, v) for k, v in pretrained_dict.items()])

    pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict)#, strict=False)

    return model


def sample_uv_map(uv_map):
    sampled_by_x = sorted(list(uv_map.keys()))[0:len(uv_map.keys()):2] # 17855
    sampled_by_y = sorted(sampled_by_x, key=lambda x: x[1])[0:len(sampled_by_x):2] # 8928
    new_uv_map = {}
    for uv in sampled_by_y:
        new_uv_map[uv] = uv_map[uv]

    f = open("HELEN_HELEN_3036412907_2_0_1_wo_pose_uvmap_sampled.pkl", 'wb')
    pickle.dump(new_uv_map, f)
    f.close()

def adjust_range_uv_map(uv_map):
    v_min = np.array(list(uv_map.keys()))[:, 1].min()
    v_max = np.array(list(uv_map.keys()))[:, 1].max()
    new_uv_map = {}
    for uv in uv_map.keys():
        new_v = (uv[1] - v_min)/(v_max-v_min)
        new_uv_map[(uv[0], new_v)] = uv_map[uv]
    
    f = open("HELEN_HELEN_3036412907_2_0_1_wo_pose_uvmap_full.pkl", 'wb')
    pickle.dump(new_uv_map, f)
    f.close()

def find_boundary_uvs(uv_map, face_contour, reference_mesh):
    boundary_verts = np.round(reference_mesh[:, face_contour], decimals=6)
    excluded_uvs = []
    for uv in uv_map.keys():
        for x in boundary_verts[:2, :][0]:
            for y in boundary_verts[:2, :][1]:
                if uv_map[uv] == [x, y]:
                    excluded_uvs.append(uv)

    return excluded_uvs

def get_neighbor_uvs(ori_uv_map, new_uv, r):
    new_u, new_v = new_uv
    neighbor_uvs = {}
    for old_u, old_v in ori_uv_map.keys():
        d = hypot(new_u - old_u, new_v - old_v)
        if d <= r:
            neighbor_uvs[(old_u, old_v)] = d
    
    print(new_uv, neighbor_uvs)
    return neighbor_uvs

def uniform_resample_uv_map(ori_uv_map):
    uniform_coord = np.linspace(0,1,120)
    r = 1 / 120 * 8 
    uv_pairs = {}
    for new_u in uniform_coord:
        for new_v in uniform_coord:
            new_uv = (new_u, new_v)
            neighbor_uvs = get_neighbor_uvs(ori_uv_map, (new_u,new_v), r)
            
    
            # if len(neighbor_uvs) >= 5:
                # find weight according to distance
                # calculate 
                # new_xyz(uv_i) = summation(w_i * xyz(uv_i))
                
            # else:
                # what to do?
                
            uv_pairs[(new_u, new_v)] = neighbor_uvs

    # for new_u in uniform_coord:
    #     for new_v in uniform_coord:
    #         new_uv = (new_u, new_v)


    return uv_pairs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFADataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader

    def _target_loader(self, index):
        target = self.params[index]

        return target

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)

        target = self._target_loader(index)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.lines)


class DDFATestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = img_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.lines)



class LpDataset:
    def __init__(self, file_list, root_dir, transform=None):
        self.data = open(file_list, "r").readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.imgs_path = list()
        self.coeffs = []

        for line in self.data:
            line = line.rstrip()
            if ".jpg" in line:
                path = self.root_dir + line
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                coeff = [float(x) for x in line]
                self.coeffs.append(coeff)

        self.coeffs = torch.from_numpy(np.array(self.coeffs))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        coeff = self.coeffs[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, coeff



class DDFAMeshDataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))

        self.param_mean = _numpy_to_tensor(param_full_mean)
        self.param_std = _numpy_to_tensor(param_full_std)

        self.u = _numpy_to_tensor(u_)#.double()
        self.w_shp = _numpy_to_tensor(w_shp_)#.double()
        self.w_exp = _numpy_to_tensor(w_exp_)#.double()
    

    def __getitem__(self, index):
        gt_param = self.params[index]
        gt_param = gt_param * self.param_std + self.param_mean
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param(gt_param.float())

        target_vert = (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg).view(-1, 3)

        return target_vert, target_vert

    def __len__(self):
        return len(self.lines)