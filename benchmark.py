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

from benchmark_aflw2000 import calc_nme
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw2000 import calc_nme_rescaled, calc_nmse, aflw_meshes, draw_landmarks
from utils.inference import _predict_vertices, dump_rendered_img, dump_to_ply, rescale_w_roi, get_landmarks
from utils.io import _load
from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, DDFADataset, reconstruct_vertex
from utils.params import *
from utils.render_simdr import render
from bernstein_ffd.ffd_utils import deformed_vert, cp_num, cp_num_
import mobilenet_v1_ffd
import mobilenet_v1
import mobilenet_v1_ffd_lm


root = '../Datasets/AFLW2000/Data/'
filelist = open('../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list', "r").read().split("\n")
roi_boxs = _load('test_configs/AFLW2000-3D_crop.roi_box.npy')

root_lp = '../Datasets/train_aug_120x120/'
filelist_lp = open('train.configs/train_aug_120x120.list.val.part', "r").read().split("\n")


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def extract_param(checkpoint_fp, root='', filelists=None, param_fp=None, arch='mobilenet_1', param_classes=cp_num_, lm_classes=136, device_ids=[0],
                  batch_size=128, num_workers=4, cpu=False):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']

    if arch.startswith('mobilenet'):
        # model = getattr(mobilenet_v1, arch)(num_classes=62)
        model = getattr(mobilenet_v1_ffd, arch)(param_classes=param_classes)#, lm_classes=lm_classes)
    elif arch.startswith('resnet'):
        model = torchvision.models.resnet50(num_classes=param_classes)
        # model = torchvision.models.resnet50(num_classes=param_classes)

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


def extract_param_(checkpoint_fp, root='', filelists=None, param_fp=None, arch='mobilenet_1', param_classes=cp_num_, lm_classes=136, device_ids=[0],
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
    return ana_alfw2000(calc_nme(outputs, dense=dense, all=True, dim=dim))
    # return ana_alfw2000(calc_nmse(outputs))


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])

    return _benchmark_aflw2000(outputs)


def benchmark_aflw2000_ffd(deforms, dense=False, dim=2, rewhiten=False):
    outputs = []
    for i in range(deforms.shape[0]):
        img_fp = filelist[i]
        deform = deforms[i]
        if rewhiten:
            deform = deform * delta_p_std - delta_p_mean
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



def benchmark_aflw2000_ffd_full(deforms):
    outputs = []
    for i in range(deforms.shape[0]):
        deform = deforms[i]
        vert = deformed_vert(deform, transform=True, face=False) # 3 x 38365
        lm = get_landmarks(vert)
        # outputs.append(lm[:, :])
        outputs.append(lm[:2, :])

    return _benchmark_aflw2000(outputs)


def reconstruct_face_mesh(params, rewhiten=False):
    outputs = []
    for i in range(params.shape[0]):
        gt_vert = aflw_meshes[i]
        deform = params[i]
        if rewhiten:
            deform = deform * delta_p_std - delta_p_mean
        vert = deformed_vert(deform, transform=True, face=True) # 3 x 38365
        vertex = rescale_w_roi(vert, roi_boxs[i])
        # vertex = _predict_vertices(params[i], roi_boxs[i], dense=True, transform=True) # image coordinate space
        wfp = None
        # wfp = f"samples/outputs/aflw_region_lm_0.46/{filelist[i]}"
        img_ori = cv2.imread(root + filelist[i])

        dis = vertex[:2,:] - gt_vert[:2,:]
        print(i, filelist[i])
        # mouth loss
        print("mouth: ", np.mean(np.abs(dis[:, [*upper_mouth, *lower_mouth]])))
        # eye loss
        print("eyes: ", np.mean(np.abs(dis[:, [*left_eye, *right_eye]])))
        # nose loss
        print("nose: ", np.mean(np.abs(dis[:, [*lower_nose, *upper_nose]])))
        # brow loss
        print("brow: ", np.mean(np.abs(dis[:, [*left_brow, *right_brow]])))
        # contour loss
        print("contour: ", np.mean(np.abs(dis[:, contour_boundary])))

        vertex[1, :] = img_ori.shape[0] + 1 - vertex[1, :]
        render(img_ori, [vertex], tri_.astype(np.int32), alpha=0.8, show_flag=True, wfp=wfp, with_bg_flag=True, transform=True)
        # wfp = f"samples/outputs/aflw/{filelist[i].replace('.jpg', '.ply')}"
        # dump_to_ply(vertex, tri_.T, wfp, transform=False)
        outputs.append(vertex)

    return outputs


def reconstruct_face_mesh_(params):
    outputs = []
    for i in range(len(params)):
        deform = params[i][0]
        vertex = deformed_vert(deform, transform=True, face=True) # 3 x 38365
        # vertex = rescale_w_roi(vert, roi_boxs[i])

        gt_param = np.array(params[i][1][:62])
        gt_vert = reconstruct_vertex(gt_param, dense=True, transform=True)

        # wfp_gt = f"samples/outputs/300w-lp/{filelist_lp[i].replace('.jpg', '_gt.jpg')}"
        # wfp = f"samples/outputs/300w-lp/{filelist_lp[i]}"
        # dump_rendered_img(gt_vert, root_lp + filelist_lp[i], wfp=None, show_flag=True, alpha=0.8)

        img_ori = cv2.imread(root_lp + filelist_lp[i])
        vertex[1, :] = img_ori.shape[0] + 1 - vertex[1, :]
        render(img_ori, [vertex], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

        gt_vert[1, :] = img_ori.shape[0] + 1 - gt_vert[1, :]
        render(img_ori, [gt_vert], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
        # wfp = f"samples/outputs/{filelist[i].replace('.jpg', '.ply')}"
        # dump_to_ply(vertex, tri_.T, wfp, transform=True)
        outputs.append(vertex)

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


def benchmark_pipeline_ffd(arch, checkpoint_fp, dense=False, dim=2, rewhiten=False):
    device_ids = [0]
    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop',
        filelists='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        batch_size=128,
        cpu=False)

    return benchmark_aflw2000_ffd(params, dense=dense, dim=dim, rewhiten=rewhiten)


def aflw2000_mesh(arch, checkpoint_fp, rewhiten=False):
    device_ids = [0]
    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop',
        filelists='../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        batch_size=128)

    # params = np.zeros((64, 648))
    # return reconstruct_full_mesh(params)
    return reconstruct_face_mesh(params, rewhiten=rewhiten)


def lp_mesh(arch, checkpoint_fp):
    device_ids = [0]
    params = extract_param_(
        checkpoint_fp=checkpoint_fp,
        root='../Datasets/train_aug_120x120',
        filelists='train.configs/train_aug_120x120.list.val.part',
        param_fp='train.configs/param_lm_val.pkl',
        arch=arch,
        device_ids=device_ids,
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
    parser.add_argument('--arch', default='mobilenet_1', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/phase2_wpdc_lm_vdc_all_checkpoint_epoch_19.pth.tar', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_mb_delta_p_norm/ffd_mb_delta_p_norm_checkpoint_epoch_50.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_resnet_region_lm_0.37_mse/ffd_resnet_region_lm_0.37_mse_checkpoint_epoch_23.pth.tar', type=str)
    # parser.add_argument('-c', '--checkpoint-fp', default='snapshot/ffd_mb_v2/ffd_mb_v2_checkpoint_epoch_37.pth.tar', type=str)
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

    # print("checkpoint: ", args.checkpoint_fp)
    # benchmark_pipeline_ffd(args.arch, args.checkpoint_fp, dense=False, dim=2)

    # lp_mesh(args.arch, args.checkpoint_fp)
    # aflw2000_mesh(args.arch, args.checkpoint_fp, rewhiten=True)

    min_nme = 100
    min_index = 0
    min_1 = 100
    min_1_index = 0
    min_2 = 100
    min_2_index = 0
    min_3 = 100
    min_3_index = 0
    for i in range(1, 51):
        # checkpoint = f"snapshot/ffd_resnet_lm_19/ffd_resnet_lm_19_checkpoint_epoch_{i}.pth.tar"            
        checkpoint = f"snapshot/ffd_mb_lr_delta_p/ffd_mb_lr_delta_p_checkpoint_epoch_{i}.pth.tar"
        # checkpoint = f"snapshot/ffd_resnet_region_ratio/ffd_resnet_region_ratio_checkpoint_epoch_{i}.pth.tar"
        # checkpoint = f"snapshot/ffd_resnet_lm/ffd_resnet_lm_checkpoint_epoch_{i}.pth.tar"
        print(i, checkpoint)
        mean_nme_1, mean_nme_2, mean_nme_3, mean, std = benchmark_pipeline_ffd(args.arch, checkpoint, dim=2, rewhiten=False)
        if mean < min_nme:
            min_nme = mean
            min_index = i
        if mean_nme_1 < min_1:
            min_1 = mean_nme_1
            min_1_index = i
        if mean_nme_2 < min_2:
            min_2 = mean_nme_2 
            min_2_index = i
        if mean_nme_3 < min_3:
            min_3 = mean_nme_3
            min_3_index = i

    print("min mean: ", min_index, min_nme)
    print("min nme 1: ", min_1, min_1_index)
    print("min nme 2: ", min_2, min_2_index)
    print("min nme 3: ", min_3, min_3_index)


if __name__ == '__main__':
    main()

