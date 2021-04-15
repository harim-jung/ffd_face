#!/usr/bin/env python3
# coding: utf-8

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1_ffd
import numpy as np
import cv2
import time
import torchvision
from os import walk
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from utils.inference import get_suffix, crop_img, predict_68pts, dump_to_ply, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, dump_rendered_img, rescale_w_roi, get_landmarks
from utils.ddfa import load_model
from utils.params import tri_, std_size
from utils.render_simdr import render
from retina_face.models.retinaface import RetinaFace
from retina_face.data.config import cfg_mnet
from retina_face.layers.functions.prior_box import PriorBox
from retina_face.utils.box_utils import decode, decode_landm
from retina_face.utils.nms.py_cpu_nms import py_cpu_nms
from bernstein_ffd.ffd_utils import deformed_vert

import argparse
import torch.backends.cudnn as cudnn


def get_box(img_raw):
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    bboxes = []
    for b in dets:
        if b[4] > args.vis_thres:
            bbox = [b[0], b[1], b[2], b[3]]
            bboxes.append(bbox)

    return bboxes



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', default=['samples/inputs/image00408.jpg'], type=list,
                        help='image files paths fed into network, single or multiple images')
    # parser.add_argument('-f', '--files', nargs='+',
    #                     help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')

    parser.add_argument('--detect-checkpoint', default='retina_face/weights/mobilenet0.25_Final.pth', type=str, metavar='PATH')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    # parser.add_argument('--recon-checkpoint', default='snapshot/ffd_resnet_region_lm_0.46_checkpoint_epoch_7.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--recon-checkpoint', default='snapshot/ffd_resnet_lm_adamw/ffd_resnet_lm_adamw_checkpoint_epoch_10.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--recon-model', default='resnet', type=str)
    # parser.add_argument('--param-classes', default=1470, type=int)
    parser.add_argument('--param-classes', default=1029, type=int)
    parser.add_argument('--dump_lm_img', default='false', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_vert_img', default='true', type=str2bool)
    parser.add_argument('--show_flag', default='false', type=str2bool, help='whether show the visualization result')



    args = parser.parse_args()

    # 1. load pre-tained model
    gpu_mode = args.mode == 'gpu'
    # detection model
    cpu_mode = not gpu_mode
    net = RetinaFace(cfg=cfg_mnet, phase='test')
    net = load_model(net, args.detect_checkpoint, cpu_mode)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu_mode else "cuda")
    net = net.to(device)
    resize = 1

    # reconstruction model
    if args.recon_model.startswith("resnet"):
        model = torchvision.models.resnet50(num_classes=args.param_classes)
    else:
        # model = getattr(mobilenet_v1, arch)(num_classes=62)
        model = getattr(mobilenet_v1_ffd, args.recon_model)(param_classes=args.param_classes)  # 62 = 12(pose) + 40(shape) +10(expression)
    load_model(model, args.recon_checkpoint, load_to_cpu=args.mode == 'cpu')
    
    if gpu_mode:
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()


    # 3. forward
    tri = tri_
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    # for img_fp in args.files:
    d = '../Datasets/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png/'
    save = '../Datasets/CelebA/results/ffd_resnet_lm_adamw/'
    # d = '../Datasets/300W_LP/Data/'
    # save = '../Datasets/300W_LP/results_ffd/'
    for (dirpath, dirnames, filenames) in walk(d):
        for img_fp in filenames[:5000]:
            if img_fp.endswith(".jpg") or img_fp.endswith(".png"):
                # for img_fp in args.files:
                img_ori = cv2.imread(d + img_fp)
                # img_ori = cv2.imread("samples/inputs/wilderface-5.jpg")

                boxes = get_box(img_ori)
                if len(boxes) == 0:
                    continue

                pts_res = []
                vertices_lst = []  # store multiple face vertices
                ind = 0
                suffix = get_suffix(img_fp)
                for bbox in boxes:
                    # use detected face bbox to find roi box
                    roi_box = parse_roi_box_from_bbox(bbox)

                    # crop image using roi box
                    img = crop_img(img_ori, roi_box)

                    # resize image to standard size
                    img = cv2.resize(img, dsize=(std_size, std_size), interpolation=cv2.INTER_LINEAR)
                    input = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        if gpu_mode:
                            input = input.cuda()
                        start = time.time()
                        param = model(input)
                        print("inference time: ", (time.time() - start)*1000, "ms")
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    root = "samples/outputs/"
                    wfp = root + img_fp.split("/")[-1]
                    # dense face 3d vertices
                    if args.dump_ply or args.dump_vert_img or args.dump_lm_img:
                        vertices = deformed_vert(param, transform=True, face=True)
                        vertices = rescale_w_roi(vertices, roi_box)
                        vertices[1, :] = img_ori.shape[0] + 1 - vertices[1, :]

                        vertices_lst.append(vertices)

                        # 68 landmarks
                        pts68 = get_landmarks(vertices, face=True)
                        pts_res.append(pts68)

                    # if args.dump_vert_img:
                    #     # saves rendered image for each face
                        
                    #    dump_rendered_img(vertices, img_fp, wfp='{}_{}.jpg'.format(wfp.replace(suffix, ''), ind), show_flag=args.show_flag)
                    if args.dump_ply:
                        # saves to ply file for each face
                        wfp = save + '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind)
                        dump_to_ply(vertices, tri.T, wfp, transform=False)
                    ind += 1
                if args.dump_lm_img:
                    # saves rendered landmarks for all faces
                    # wfp = wfp.replace(suffix, '_lms.jpg')
                    wfp = None
                    draw_landmarks(img_ori, pts_res, wfp=wfp, show_flag=args.show_flag, tranform=True)
                if args.dump_vert_img:
                    wfp = None
                    wfp = save + img_fp
                    render(img_ori, vertices_lst, tri_, alpha=0.8, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True, transform=True)
