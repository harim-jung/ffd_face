#!/usr/bin/env python3
# coding: utf-8

"""
Notation (2019.09.15): two versions of spliting AFLW2000-3D:
 1) AFLW2000-3D.pose.npy: according to the fitted pose
 2) AFLW2000-3D-new.pose: according to AFLW labels 
There is no obvious difference between these two splits.
"""


import os.path as osp
import numpy as np
import cv2
from math import sqrt
from utils.io import _load
from utils.params import std_size
from utils.ddfa import LpDataset, reconstruct_vertex
from utils.inference import dump_to_ply
from utils.params import *
from utils.render_simdr import render
import matplotlib.pyplot as plt

d = 'test_configs'

# [1312, 383, 305], current version
yaws_list = _load(osp.join(d, 'AFLW2000-3D.pose.npy'))

# [1306, 462, 232], same as paper 
# yaws_list = _load(osp.join(d, 'AFLW2000-3D-new.pose.npy'))

# origin
pts68_all = _load(osp.join(d, 'AFLW2000-3D.pts68.npy'))

roi_boxs = _load(osp.join(d, 'AFLW2000-3D_crop.roi_box.npy'))

aflw_meshes = _load(osp.join(d, 'aflw_gt_mesh_35709_z.pkl'))

root = '../Datasets/AFLW2000/Data/'
filelist = open('../Datasets/AFLW2000/test.data/AFLW2000-3D_crop.list', "r").read().split("\n")


def aflw_mesh():
    root = '../Datasets/AFLW2000/Data/'
    aflw_gt = LpDataset('test_configs/aflw_gt.txt', root)
    verts = []
    for i in range(len(aflw_gt)):
        param = aflw_gt.coeffs[i].numpy()
        vert = reconstruct_vertex(param, dense=True, face=True, transform=True, std_size=450)
        vert = np.array(vert).astype(np.float32)
        vert[2, :] -= np.min(vert[2, :]) 
        verts.append(vert)

        img_ori = cv2.imread(aflw_gt.imgs_path[i])
        render(img_ori, [vert], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
        wfp = f"samples/outputs/{aflw_gt.imgs_path[i].split('/')[-1].replace('.jpg', '.ply')}"
        dump_to_ply(vert, tri_.T, wfp, transform=True)

    return verts


def ana(nme_list):
    yaw_list_abs = np.abs(yaws_list)
    ind_yaw_1 = yaw_list_abs <= 30
    ind_yaw_2 = np.bitwise_and(yaw_list_abs > 30, yaw_list_abs <= 60)
    ind_yaw_3 = yaw_list_abs > 60

    nme_1 = nme_list[ind_yaw_1]
    nme_2 = nme_list[ind_yaw_2]
    nme_3 = nme_list[ind_yaw_3]

    mean_nme_1 = np.mean(nme_1) * 100
    mean_nme_2 = np.mean(nme_2) * 100
    mean_nme_3 = np.mean(nme_3) * 100
    mean_nme_all = np.mean(nme_list) * 100

    std_nme_1 = np.std(nme_1) * 100
    std_nme_2 = np.std(nme_2) * 100
    std_nme_3 = np.std(nme_3) * 100
    std_nme_all = np.std(nme_list) * 100

    mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
    mean = np.mean(mean_all)
    std = np.std(mean_all)

    s1 = '[ 0, 30]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
    s2 = '[30, 60]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
    s3 = '[60, 90]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
    # s4 = '[ALL]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_all, std_nme_all)
    s5 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: \x1b[31m{:.3f}\x1b[0m'.format(mean, std)

    s = '\n'.join([s1, s2, s3, s5])#, s4])
    print(s)

    return mean_nme_1, mean_nme_2, mean_nme_3, mean, std


def convert_to_ori(lms, i, dim=2):
    sx, sy, ex, ey = roi_boxs[i]
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    lms[0, :] = lms[0, :] * scale_x + sx
    lms[1, :] = lms[1, :] * scale_y + sy
    if dim == 3:
        s = (scale_x + scale_y) / 2
        lms[2, :] *= s
        lms[2, :] -= np.min(lms[2, :])

    return lms


def calc_nme(pts, dense=False, all=True, dim=2):
    if dense:
        nme = calc_nme_mesh(pts)
    else:
        nme = calc_nme_lm(pts, all=all, dim=dim)
    
    return nme


def calc_nme_lm(pts68_fit_all, all=True, dim=2):
    nme_list = []
    l1_list = []
    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        # shift z-coordinates (temp)
        if dim == 3:
            pts68_gt[2, :] -= np.min(pts68_gt[2, :])


        # rescale to original image size
        pts68_fit = convert_to_ori(pts68_fit, i, dim=dim)
        # z_diff = pts68_fit[2, 0] - pts68_gt[2, 0]
        # pts68_fit[2, :] -= z_diff


        # wfp = f'samples/outputs/aflw_lms_region_lm_0.46/{filelist[i]}'
        # draw_landmarks(root + filelist[i], pts68_fit[:2, :],  pts68_gt[:2, :], style='simple', wfp=wfp, show_flag=False)


        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        if all:
            # test for all 68 landmarks
            if dim == 2:
                dis = pts68_fit[:, :] - pts68_gt[:2, :]
            else:
                dis = pts68_fit[:, :] - pts68_gt[:, :]
        else:
            # test for 51 landmarks excluding jaw
            if dim == 2:
                dis = pts68_fit[:, 17:] - pts68_gt[:2, 17:]
            else:
                dis = pts68_fit[:, 17:] - pts68_gt[:, 17:]

        # l1 loss
        l1_list.append(np.mean(np.abs(dis)))
        # l1_list.append(np.mean(np.sum(np.abs(dis), 0)))

        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength

        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    print("L1 Loss ", np.array(l1_list, dtype=np.float32).mean())
    return nme_list


def calc_nme_mesh(vert, dim=3):
    # vert_gt = aflw_mesh()

    nme_list = []
    for i in range(len(roi_boxs)):
        vert_fit = vert[i]
        vert_gt = aflw_meshes[i]
        pts68_gt = pts68_all[i]

        # rescale to original image size
        vert_fit = convert_to_ori(vert_fit, i, dim=dim)

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        dis = vert_fit - vert_gt

        # temp
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


        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list


def calc_nmse(pts68_fit_all):
    nmse_list = []

    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        # rescale to original image size
        pts68_fit = convert_to_ori(pts68_fit, i)

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        # test for 51 landmarks excluding jaw
        dis = pts68_fit[:, 17:] - pts68_gt[:2, 17:]
        dis = np.sum(np.power(dis, 2), 0)
        dis = np.mean(dis)
        nmse = dis / llength
        nmse_list.append(nmse)

    nmse_list = np.array(nmse_list, dtype=np.float32)
    return nmse_list


def calc_nme_rescaled(pts68_fit_all, option='ori'):
    nme_list = []

    for i in range(len(pts68_fit_all)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        # test for 51 landmarks excluding jaw
        dis = pts68_fit[:, 17:] - pts68_gt[:2, 17:]
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list


def draw_landmarks(img_fp, pts, gt_pts, style='simple', wfp=None, show_flag=False, transform=False, **kwargs):
    """Draw landmarks using matplotlib"""
    img = cv2.imread(img_fp)
    height, width = img.shape[:2]
    plt.figure(figsize=(4, height / width * 4))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
        gt_pts = [gt_pts]
    for i in range(len(pts)):
        if style == 'simple':
            plt.plot(pts[i][0, :], pts[i][1, :], 'o', markersize=1, color='g')
            plt.plot(gt_pts[i][0, :], gt_pts[i][1, :], 'o', markersize=1, color='r')

        elif style == 'fancy':
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)

    if wfp is not None:
        plt.savefig(wfp, dpi=200)
        print('Save visualization result to {}'.format(wfp))
        plt.close()
    if show_flag:
        plt.show()


def main():
    pass


if __name__ == '__main__':
    main()

    import pickle 

    vert = aflw_mesh()
    # f = open('train.configs/aflw/aflw_gt_mesh_35709_.pkl', 'wb')
    # pickle.dump(vert, f)
    # f.close()