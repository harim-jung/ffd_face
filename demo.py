#!/usr/bin/env python3
# coding: utf-8

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import time
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from utils.inference import get_suffix, crop_img, predict_68pts, dump_to_ply, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, dump_rendered_img
from utils.ddfa import load_model
from utils.params import tri_
from utils.params import std_size
from retina_face.models.retinaface import RetinaFace
from retina_face.data.config import cfg_mnet
from retina_face.layers.functions.prior_box import PriorBox
from retina_face.utils.box_utils import decode, decode_landm
from retina_face.utils.nms.py_cpu_nms import py_cpu_nms
from utils.render_simdr import render
from os import walk

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
    parser.add_argument('-f', '--files', default=['samples/inputs/image00066.jpg'], type=list,
                        help='image files paths fed into network, single or multiple images')
    # parser.add_argument('-f', '--files', nargs='+',
    #                     help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flag', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_lm_img', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('--dump_vert_img', default='true', type=str2bool)

    parser.add_argument('--detect-checkpoint', default='retina_face/weights/mobilenet0.25_Final.pth', type=str, metavar='PATH')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    # parser.add_argument('--recon-checkpoint', default='snapshot/ffd_adam/ffd_adam_checkpoint_epoch_30.pth.tar', type=str, metavar='PATH')
    parser.add_argument('--recon-checkpoint', default='snapshot/phase2_wpdc_lm_vdc_all_checkpoint_epoch_19.pth.tar', type=str, metavar='PATH')
    # parser.add_argument('--recon-checkpoint', default='snapshot/mb1_120x120.pth', type=str, metavar='PATH')
    parser.add_argument('--recon-model', default='mobilenet_1', type=str)



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
    model = getattr(mobilenet_v1, args.recon_model)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    load_model(model, args.recon_checkpoint, load_to_cpu=args.mode == 'cpu')
    if gpu_mode:
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()


    # 3. forward
    tri = tri_
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    d = '../Datasets/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png/'
    save = '../Datasets/CelebA/results/3ddfa/'
    for (dirpath, dirnames, filenames) in walk(d):
        for img_fp in filenames[:100]:
            img_ori = cv2.imread(d + img_fp)

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

                # 68 landmarks
                pts68 = predict_68pts(param, roi_box)
                pts_res.append(pts68)

                root = "samples/outputs/"
                wfp = root + img_fp.split("/")[-1]
                # dense face 3d vertices
                if args.dump_ply or args.dump_vert_img:
                    vertices = predict_dense(param, roi_box, transform=True)

                    vertices[1, :] = img_ori.shape[0] + 1 - vertices[1, :]

                    vertices_lst.append(vertices)
                if args.dump_ply:
                    # saves to ply file for each face
                    dump_to_ply(vertices, tri.T, '{}_{}.ply'.format(wfp.replace(suffix, ''), ind), transform=True)
                ind += 1
            if args.dump_lm_img:
                # saves rendered landmarks for all faces
                draw_landmarks(img_ori, pts_res, wfp=wfp.replace(suffix, '_lms.jpg'), show_flag=args.show_flag)
            if args.dump_vert_img:
                wfp = save + img_fp
                render(img_ori, vertices_lst, tri_, alpha=0.8, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True, transform=True)