#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
import time
import numpy as np
import os.path as ospconda
import os
from collections import OrderedDict
import argparse
import pickle
from scipy.io import loadmat
import cv2

from benchmark_aflw2000 import calc_nme, ana_sampled
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw2000 import calc_nme_rescaled, calc_nmse, aflw_meshes, draw_landmarks
from utils.inference import _predict_vertices, dump_rendered_img, dump_to_ply, rescale_w_roi, get_landmarks
from utils.io import _load
from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, DDFADataset, LpDataset, reconstruct_vertex, reconstruct_vertex_full, \
get_rot_mat_from_axis_angle, get_rot_mat_from_axis_angle_np, get_rot_mat_from_axis_angle_batch, \
get_axis_angle_from_rot_mat, get_axis_angle_from_rot_mat_batch, get_axis_angle_s_t_from_rot_mat_batch, _parse_param, \
adjust_range_uv_map, find_boundary_uvs, uniform_resample_uv_map
from utils.params import *
from utils.render_simdr import render
from bernstein_ffd.ffd_utils import deformed_vert, cp_num, deformed_vert_w_pose, deformed_vert_w_pose_nurbs, \
uv_map, face_contour, reference_mesh, nurbs_cp_num
import models.mobilenet_v1_ffd as mobilenet_v1_ffd
import models.mobilenet_v1
import models.mobilenet_v1_ffd_lm
from efficientnet_pytorch import EfficientNet

# import bernstein_ffd.ffd_utils_patch

# new_uv_map = find_boundary_uvs(uv_map, face_contour, reference_mesh)
# new_uv_map = uniform_resample_uv_map(uv_map)

# root = '../Datasets/AFLW2000/Data/'
# aflw_gt = LpDataset('test_configs/aflw_gt.txt', root)
# file = aflw_gt.imgs_path[0]
# print(file)
# gt_param = aflw_gt[0][1].numpy()

# lms = reconstruct_vertex_full(gt_param, dense=False, face=False, transform=True, std_size=450)

# pts68_all = _load(osp.join('test_configs', 'AFLW2000-3D.pts68.npy'))
# gt_lms = pts68_all[0]

# from scipy.io import loadmat
# # temp test
# example_mat = loadmat("../Datasets/AFLW2000/Data/image04375.mat")
# alpha_shp = example_mat["Shape_Para"]
# alpha_exp = example_mat["Exp_Para"]
# pose_param = example_mat["Pose_Para"].T
# param = np.concatenate((pose_param, alpha_shp, alpha_exp))

# vertex = reconstruct_vertex(param, dense=True, face=False, transform=False, std_size=std_size)

# params = np.load("train.configs/param_all_full.pkl", allow_pickle=True)
# pose_param = torch.tensor([params[0][:12], params[1][:12]]).cuda()
# new_pose_param = get_axis_angle_s_t_from_rot_mat_batch(pose_param)

# r = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).float().cuda()
# rot_mat = get_rot_mat_from_axis_angle_batch(r)
# new_axis_angle = get_axis_angle_s_t_from_rot_mat_batch

# axis_angle = np.array([1.5, 2.5, 3.5])
# rot_mat = get_rot_mat_from_axis_angle_np(axis_angle)
# new_axis_angle = get_axis_angle_from_rot_mat(rot_mat)


root = '../Datasets/AFLW2000/Data/'
filelist = open('../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list', "r").read().split("\n")
roi_boxs = _load('test_configs/AFLW2000-3D_crop.roi_box.npy')

root_lp = '../Datasets/train_aug_120x120/'
filelist_lp = open('train.configs/train_aug_120x120.list.val.part', "r").read().split("\n")

aflw_root = '../Datasets/AFLW2000/Data/'
aflw_gt_params = LpDataset('test_configs/aflw_gt.txt', aflw_root)

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def extract_param(checkpoint_fp, root='', filelists=None, param_fp=None, arch='mobilenet_1', param_classes=cp_num+12, lm_classes=136, device_ids=[0],
                  batch_size=128, num_workers=4, cpu=False):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
 
    if arch.startswith("mobilenet"):
        if arch.endswith("v1"):
            model = getattr(mobilenet_v1_ffd, arch)(param_classes=param_classes)
        elif arch.endswith("v3"):
            model = torchvision.models.mobilenet_v3_large(pretrained=False, num_classes=param_classes)
    elif arch.startswith("resnet"):
        model = torchvision.models.resnet50(pretrained=False, num_classes=param_classes)
    elif arch.startswith("resnext"):
        model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=param_classes)
    elif arch.startswith("efficientnet"):
        # model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=param_classes)
        model = EfficientNet.from_name('efficientnet-b3', num_classes=param_classes)

    # if "module.fc1.weight" in checkpoint:
    #     checkpoint = OrderedDict([('module.fc.weight', v) if k == 'module.fc1.weight' else (k,v) for k, v in checkpoint.items()])
    #     checkpoint = OrderedDict([('module.fc.bias', v) if k == 'module.fc1.bias' else (k,v) for k, v in checkpoint.items()])

    # remove fc2 trained for lm regression
    # if "module.fc2.weight" in checkpoint:
    #     checkpoint = OrderedDict([(k,v) for k, v in checkpoint.items() if k != 'module.fc2.weight'])
    #     checkpoint = OrderedDict([(k,v) for k, v in checkpoint.items() if k != 'module.fc2.bias'])

    if cpu:
        checkpoint = remove_prefix(checkpoint, 'module.')
    else:
        torch.cuda.set_device(device_ids[0])
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint, strict=False)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs #.cuda()
            output = model(inputs)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()
                # param_prediction = output[i][:param_classes].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def extract_param_(checkpoint_fp, root='', filelists=None, param_fp=None, arch='mobilenet_1', param_classes=cp_num, lm_classes=136, device_ids=[0],
                  batch_size=128, num_workers=4, cpu=False):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']

    if arch.startswith('mobilenet'):
        # model = getattr(mobilenet_v1, arch)(num_classes=62)
        model = getattr(mobilenet_v1_ffd, arch)(param_classes=param_classes)#, lm_classes=lm_classes)
    elif arch.startswith('resnet'):
        model = torchvision.models.resnet50(num_classes=param_classes)

    if cpu:
        checkpoint = remove_prefix(checkpoint, 'module.')
    else:
        torch.cuda.set_device(device_ids[0])
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint, strict=False)

    dataset = DDFADataset(filelists=filelists, root=root, param_fp=param_fp,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs[0].cuda()
            output = model(inputs)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()
                gt = dataset[i][1][:62]
                outputs.append([param_prediction, gt])
        outputs = outputs

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def extract_feat(checkpoint_fp,arch='mobilenet_1', param_classes=cp_num, lm_classes=136, device_ids=[0],
                  batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']

    torch.cuda.set_device(device_ids[0])
    model = getattr(mobilenet_v1_ffd, arch)(param_classes=param_classes)#, lm_classes=lm_classes)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint) #, strict=False)

    cudnn.benchmark = True
    model.eval()

    outputs = []
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    with torch.no_grad():
        inputs = ['samples/outputs/300w-lp/LFPWFlip_LFPW_image_train_0812_0_1.jpg', 
        'samples/outputs/300w-lp/LFPWFlip_LFPW_image_train_0812_0_1_gt.jpg', 
        '../Datasets/train_aug_120x120/LFPWFlip_LFPW_image_train_0812_0_1.jpg']
        for input in inputs:
            input = cv2.imread(input)
            input = transform(input).unsqueeze(0)
            y = model(input)
            outputs.append(y)
    
    return outputs


def _benchmark_aflw2000(outputs, dense=False, dim=2):
    # return ana_alfw2000(calc_nme(outputs, dense=dense, all=True, dim=dim))
    # return ana_sampled(calc_nme(outputs, dense=dense, all=True, dim=dim))
    # from os import walk

    # nme_list = calc_nme(outputs, dense=dense, all=True, dim=dim)
    # d = "test_configs/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700/"
    # for (dirpath, dirnames, filenames) in walk(d):
    #     for filename in filenames:
    #         if filename.startswith("AFLW2000-3D.small-pose-"):
    #             small = filename
    #             num = ("").join(small.split("-")[-1])
    #             med = f"AFLW2000-3D.med-pose-{num}"
    #             lar = f"AFLW2000-3D.large-pose-{num}"
    #             filenames = [small, med, lar]
    #             print(filenames)
    #             ana_sampled(nme_list, filenames=filenames)

    # d = "test_configs/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700/"
    # for (dirpath, dirnames, filenames) in walk(d):
    #     for filename in filenames:
    #         if filename.startswith("AFLW2000-3D.med-pose-"):
    #             print(filename)
    #             ana_sampled(nme_list)

    # d = "test_configs/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700/"
    # for (dirpath, dirnames, filenames) in walk(d):
    #     for filename in filenames:
    #         if filename.startswith("AFLW2000-3D.large-pose-"):
    #             print(filename)
    #             ana_sampled(nme_list)

    # nme_list = calc_nme(outputs, dense=dense, all=True, dim=dim)
    # for i in range(20):
    #     ana_sampled(nme_list, save_folder="nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700_test_2")

    # nme_list = calc_nme(outputs, dense=dense, all=True, dim=dim)
    # mean_nme1 = []
    # mean_nme2 = []
    # mean_nme3 = []
    # mean_nme = []

    # for j in range(50):
    #     print(j)
    #     nme1s = []
    #     nme2s = []
    #     nme3s = []
    #     nme = []
    #     for i in range(10):
    #         mean_nme_1, mean_nme_2, mean_nme_3, mean, std = ana_sampled(nme_list, save_folder="nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700_test")
    #         nme1s.append(mean_nme_1)
    #         nme2s.append(mean_nme_2)
    #         nme3s.append(mean_nme_3)
    #         nme.append(mean)

    #     mean_nme1.append(np.mean(nme1s))
    #     mean_nme2.append(np.mean(nme2s))
    #     mean_nme3.append(np.mean(nme3s))
    #     mean_nme.append(np.mean(nme))
    #     # print("[ 0, 30]", np.mean(nme1s))
    #     # print("[30, 60]", np.mean(nme2s))
    #     # print("[60, 90]", np.mean(nme3s))
    #     # print("[0, 90]",np.mean(nme))

    # print(np.min(mean_nme1))
    # print(np.min(mean_nme2))
    # print(np.min(mean_nme3))
    # print(np.min(mean_nme))
    return ana_sampled(calc_nme(outputs, dense=dense, all=True, dim=dim))
    
def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])

    return _benchmark_aflw2000(outputs)


def benchmark_aflw2000_ffd_(deforms, dense=False, dim=2, rewhiten=False):
    outputs = []
    for i in range(deforms.shape[0]):
        img_fp = filelist[i]
        deform = deforms[i]
        if rewhiten:
            deform = deform * delta_p_std + delta_p_mean
        vert = deformed_vert(deform, transform=True) # 3 x N
        if dense:
            outputs.append(vert)
        else:
            lm = get_landmarks(vert)
            if dim == 2:
                outputs.append(lm[:2, :])
            else:
                outputs.append(lm)

    return _benchmark_aflw2000(outputs, dense=dense, dim=dim)


def benchmark_aflw2000_ffd(params, dense=False, dim=2, rewhiten=False, pose="axis_angle"):
    outputs = []
    for i in range(params.shape[0]):
        img_fp = filelist[i]
        pred_param = params[i]
        if pose is None:
            pred_vert = deformed_vert(pred_param, transform=True) # 3 x 38365
        else:
            # pred_vert = deformed_vert_w_pose(pred_param, transform=True, rewhiten=rewhiten, pose=pose) # 3 x 38365
            pred_vert = deformed_vert_w_pose_nurbs(pred_param, transform=True, rewhiten=rewhiten, pose=pose) # 3 x 38365

        if dense:
            outputs.append(pred_vert)
        else:
            lm = get_landmarks(pred_vert)
            if dim == 2:
                outputs.append(lm[:2, :])
            else:
                outputs.append(lm)

    return _benchmark_aflw2000(outputs, dense=dense, dim=dim)



def benchmark_aflw2000_ffd_full(deforms):
    outputs = []
    for i in range(deforms.shape[0]):
        deform = deforms[i]
        vert = deformed_vert(deform, transform=True, face=False) # 3 x 38365
        lm = get_landmarks(vert)
        # outputs.append(lm[:, :])
        outputs.append(lm[:2, :])

    return _benchmark_aflw2000(outputs)


def reconstruct_face_mesh(params, pose=None):
    outputs = []
    for i in range(params.shape[0]):
        img_ori = cv2.imread(root + filelist[i])

        gt_param = aflw_gt_params[i][1]
        gt_vert = aflw_meshes[i]
        # transform y axis (original gt mesh is on the mesh coordinate system, not image coordinate)
        gt_vert[1] = img_ori.shape[0] + 1 - gt_vert[1]
        pred_param = params[i]

        if pose is None:
            pred_vert = deformed_vert(pred_param.copy(), transform=True) # 3 x 38365
        else:
            # pred_vert = deformed_vert_w_pose(pred_param.copy(), transform=True, rewhiten=True, pose=pose) # 3 x 38365
            pred_vert = deformed_vert_w_pose_nurbs(pred_param.copy(), transform=True, rewhiten=True, pose=pose) # 3 x 38365
        # pred_vert = deformed_vert_w_pose(pred_param, transform=True, rewhiten=True, pose='rot_mat') # 3 x 38365
        # pred_vert = deformed_vert(pred_param, transform=True) # 3 x 38365
        
        pred_vert = rescale_w_roi(pred_vert, roi_boxs[i])

        dis = pred_vert[:2,:] - gt_vert[:2,:]
        print(i, filelist[i])
        # mouth loss
        print("mouth: ", np.mean(dis[:, [*upper_mouth, *lower_mouth]] ** 2))
        # eye loss
        print("eyes: ", np.mean(dis[:, [*left_eye, *right_eye]] ** 2))
        # nose loss
        print("nose: ", np.mean(dis[:, [*lower_nose, *upper_nose]] ** 2))
        # brow loss
        print("brow: ", np.mean(dis[:, [*left_brow, *right_brow]] ** 2))
        # contour loss
        print("contour: ", np.mean(dis[:, contour_boundary] ** 2))

        # # pose
        # pg, offsetg, alpha_shpg, alpha_expg = _parse_param(gt_param.numpy())

        # pred_param[:12] = pred_param[:12] * param_full_std[:12] + param_full_mean[:12]
        # p_ = pred_param[:12].reshape(3, -1)
        # p = p_[:, :3]
        # offset = p_[:, -1].reshape(3, 1)

        # print("rotation: ", (p - pg).mean())
        # print("offset: ", (offset[:2] - offsetg[:2]).mean())

        # reflip y axis
        pred_vert[1, :] = img_ori.shape[0] + 1 - pred_vert[1, :]
        wfp = None
        render(img_ori, [pred_vert], tri_.astype(np.int32), alpha=0.8, show_flag=True, wfp=wfp, with_bg_flag=True, transform=True)

        # pred_vert[1, :] = img_ori.shape[0] + 1 - pred_vert[1, :]
        # wfp = None
        # render(img_ori, [pred_vert], tri_.astype(np.int32), alpha=0.8, show_flag=True, wfp=wfp, with_bg_flag=True, transform=True)

        # wfp = f"samples/outputs/aflw/{filelist[i].replace('.jpg', '.ply')}"
        # dump_to_ply(vertex, tri_.T, wfp, transform=False)
        outputs.append(pred_vert)

    return outputs


def reconstruct_face_mesh_(params):
    outputs = []
    for i in range(len(params)):
        # deform = params[i][0]
        # vertex = deformed_vert(deform, transform=True, face=True) # 3 x 38365
        # vertex = rescale_w_roi(vert, roi_boxs[i])

        gt_param = np.array(params[i][1][:62])
        gt_vert = reconstruct_vertex(gt_param, dense=True, transform=True)

        wfp_gt = f"samples/outputs/300w_lp_test/{filelist_lp[i].replace('.jpg', '_gt.jpg')}"
        # wfp = f"samples/outputs/300w-lp/{filelist_lp[i]}"
        # dump_rendered_img(gt_vert, root_lp + filelist_lp[i], wfp=None, show_flag=True, alpha=0.8)

        img_ori = cv2.imread(root_lp + filelist_lp[i])
        # vertex[1, :] = img_ori.shape[0] + 1 - vertex[1, :]
        # render(img_ori, [vertex], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

        gt_vert[1, :] = img_ori.shape[0] + 1 - gt_vert[1, :]
        render(img_ori, [gt_vert], tri_, alpha=0.8, show_flag=True, wfp=wfp_gt, with_bg_flag=True, transform=True)

        wfp = wfp_gt.replace('.jpg', '.ply')
        dump_to_ply(gt_vert, tri_.T, wfp, transform=False)
        # outputs.append(vertex)

    return outputs


def reconstruct_full_mesh(params):
    outputs = []
    for i in range(params.shape[0]):
        deform = params[i]
        vert = deformed_vert(deform, transform=False, face=False) # 3 x 38365
        vertex = rescale_w_roi(vert, roi_boxs[i])
        dump_rendered_img(vertex, root + filelist[i], wfp=None, show_flag=True, face=False)
        # wfp = f"samples/outputs/{filelist[i].replace('.jpg', '.ply')}"
        # dump_to_ply(vertex, tri_.T, wfp, transform=True)
        outputs.append(vertex)

    return outputs


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [0]
    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop',
            filelists='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=128,
            cpu=False)

        return benchmark_aflw2000_params(params)

    return aflw2000()


def benchmark_pipeline_ffd(arch, checkpoint_fp, param_classes=1470, dense=False, dim=2, rewhiten=False, pose=None):
    device_ids = [0]
    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop',
        filelists='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        batch_size=128,
        param_classes=param_classes,
        cpu=False)

    # return benchmark_aflw2000_ffd_(params, dense=dense, dim=dim, rewhiten=rewhiten)
    return benchmark_aflw2000_ffd(params, dense=dense, dim=dim, rewhiten=rewhiten, pose=pose)


def aflw2000_mesh(arch, checkpoint_fp, param_classes=1470, pose=None):
    device_ids = [0]
    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop',
        filelists='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        param_classes=param_classes,
        batch_size=128)

    # params = np.zeros((64, 648))
    # return reconstruct_full_mesh(params)
    return reconstruct_face_mesh(params, pose=pose)


def lp_mesh(arch, checkpoint_fp, param_classes=1470):
    device_ids = [0]
    params = extract_param_(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/train_aug_120x120',
        filelists='train.configs/train_aug_120x120.list.val.part',
        param_fp='train.configs/param_lm_val.pkl',
        arch=arch,
        device_ids=device_ids,
        param_classes=param_classes,
        batch_size=128)

    # return reconstruct_full_mesh(params)
    return reconstruct_face_mesh_(params)


def benchmark_aflw2000_vert(verts):
    lms = []
    for vert in verts:
        lm = get_landmarks(vert)
        lms.append(lm[:2, :])
    return ana_alfw2000(calc_nme_rescaled(lms))


def render_face_mesh(verts):
    for i in range(len(verts)):
        dump_rendered_img(verts[i], root + filelist[i], wfp=None, show_flag=True)


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    # parser.add_argument('--arch', default='mobilenet_v3', type=str)
    parser.add_argument('--arch', default='resnet', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/phase2_wpdc_lm_vdc_all_checkpoint_epoch_19.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_resnet_vertex_lm_no_pose_norm_lr_0.37/ffd_resnet_vertex_lm_no_pose_norm_lr_0.37_checkpoint_epoch_23.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_990/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_990_checkpoint_epoch_40.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_756_0.37/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_756_0.37_checkpoint_epoch_30.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_700_checkpoint_epoch_45.pth.tar', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_resnet_vertex_lm_no_pose_norm_lr_0.37_checkpoint_epoch_23.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_resnet_vertex_lm_no_pose_norm_lr/ffd_resnet_vertex_lm_no_pose_norm_lr_checkpoint_epoch_49.pth.tar', type=str)
    args = parser.parse_args()

    # params = extract_feat(
    # checkpoint_fp=args.checkpoint_fp,
    # arch=args.arch,
    # device_ids=[0],
    # batch_size=128)


    # mat = 'train.configs/35709_tri.mat'
    # tri_ = loadmat(mat)
    # tri = tri_["tri"] - 1
    # with open("train.configs/35709_tri.pkl", "wb") as i:
    #     pickle.dump(tri, i)

    # with open("train.configs/35709_keypoints.txt", "r") as f:
    #     ind = f.read().split("\n")
    #     ind = np.array(ind).astype(np.int)
    #     ind = ind - 1
    #     with open("train.configs/35709_keypoints.pkl", "wb") as i:
    #         pickle.dump(ind, i)

    # mean_nme_1_all = []
    # mean_nme_2_all = []
    # mean_nme_3_all = []
    # for i in range(20):
    #     print("checkpoint: ", args.checkpoint_fp)
    #     mean_nme_1, mean_nme_2, mean_nme_3, mean, std = benchmark_pipeline_ffd(args.arch, args.checkpoint_fp, dense=False, param_classes=nurbs_cp_num+12, dim=2, pose="rot_mat", rewhiten=True)
    #     mean_nme_1_all.append(mean_nme_1)
    #     mean_nme_2_all.append(mean_nme_2)
    #     mean_nme_3_all.append(mean_nme_3)

    print("checkpoint: ", args.checkpoint_fp)
    # benchmark_pipeline_ffd(args.arch, args.checkpoint_fp, dense=False, param_classes=nurbs_cp_num+12, dim=2, pose="rot_mat",  rewhiten=True)
    benchmark_pipeline_ffd(args.arch, args.checkpoint_fp, dense=False, param_classes=cp_num+12, dim=2, pose="rot_mat",  rewhiten=True)

    # lp_mesh(args.arch, args.checkpoint_fp, param_classes=nurbs_cp_num+12)
    # aflw2000_mesh(args.arch, args.checkpoint_fp, param_classes=nurbs_cp_num+12, pose="rot_mat")
    # aflw2000_mesh(args.arch, args.checkpoint_fp, param_classes=cp_num+12, pose="rot_mat")

    # min_nme = 100
    # min_index = 0
    # min_1 = 100
    # min_1_index = 0
    # min_2 = 100
    # min_2_index = 0
    # min_3 = 100
    # min_3_index = 0
    # for i in range(30, 51):
    #     # checkpoint = f"snapshot/ffd_resnet_lm_19/ffd_resnet_lm_19_checkpoint_epoch_{i}.pth.tar"            
    #     # checkpoint = f"snapshot/ffd_resnet_region_lm_0.46_5000/ffd_resnet_region_lm_0.46_5000_checkpoint_epoch_{i}.pth.tar"
    #     # checkpoint = f"snapshot/ffd_resnet_region_lm_0.46_10500/ffd_resnet_region_lm_0.46_10500_checkpoint_epoch_{i}.pth.tar"
    #     # checkpoint = f"snapshot/ffd_resnext_vertex_lm_no_pose_norm/ffd_resnext_vertex_lm_no_pose_norm_checkpoint_epoch_{i}.pth.tar"
    #     checkpoint = f"snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_756_0.37/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_756_0.37_checkpoint_epoch_{i}.pth.tar"
    #     # checkpoint = f"snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_990/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_990_checkpoint_epoch_{i}.pth.tar"
    #     # checkpoint = f"snapshot/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr/nurbs_ffd_resnet_vertex_lm_no_pose_norm_lr_checkpoint_epoch_{i}.pth.tar"
    #     print(i, checkpoint)
    #     # mean_nme_1, mean_nme_2, mean_nme_3, mean, std = benchmark_pipeline_ffd(args.arch, checkpoint, param_classes=cp_num, dim=2, rewhiten=True, pose=None)
    #     mean_nme_1, mean_nme_2, mean_nme_3, mean, std = benchmark_pipeline_ffd(args.arch, checkpoint, param_classes=nurbs_cp_num+12, dim=2, rewhiten=True, pose='rot_mat')
    #     # mean_nme_1, mean_nme_2, mean_nme_3, mean, std = benchmark_pipeline_ffd(args.arch, checkpoint, param_classes=cp_num+12, dim=2, rewhiten=True, pose='rot_mat')
    #     if mean < min_nme:
    #         min_nme = mean
    #         min_index = i
    #     if mean_nme_1 < min_1:
    #         min_1 = mean_nme_1
    #         min_1_index = i
    #     if mean_nme_2 < min_2:
    #         min_2 = mean_nme_2 
    #         min_2_index = i
    #     if mean_nme_3 < min_3:
    #         min_3 = mean_nme_3
    #         min_3_index = i

    # print("min mean: ", min_index, min_nme)
    # print("min nme 1: ", min_1, min_1_index)
    # print("min nme 2: ", min_2, min_2_index)
    # print("min nme 3: ", min_3, min_3_index)


if __name__ == '__main__':
    main()

