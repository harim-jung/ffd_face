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
from math import cos, sin, sqrt
# from bernstein_ffd.ffd_utils import deform_matrix, control_points
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


# def _parse_full_param(params):
#     """one inference with numpy"""
#     pose = params[:7] # pitch, yaw, roll, t3dx, t3dy, t3dz, f
#     rot = pose[:3]
#     R = _numpy_to_tensor(get_rotation_matrix(rot))
#     offset = pose[3:6].reshape((-1, 1)) # move reshape?
#     scale = pose[6]
#     shp = params[7:206].reshape((-1, 1))
#     exp = params[206:].reshape((-1, 1))

#     return R * scale, offset, shp, exp


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
    r_hat = r / theta # N x 3

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
            vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

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