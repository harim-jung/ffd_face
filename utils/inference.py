#!/usr/bin/env python3
# coding: utf-8
__author__ = 'cleardusk'

from math import sqrt

import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2

# from .ddfa import reconstruct_vertex
from .params import tri_, std_size, tri, keypoints_, keypoints
from .render import cfg, _to_ctype
from utils.lighting import RenderPipeline

# temp
std_size = 120

def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def dump_to_ply(vertex, tri, wfp, transform=False):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    vertex = vertex.copy()
    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    if transform:
        vertex[1, :] = std_size + 1 - vertex[1, :]
    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1, idx2, idx3))
    print('Dump to {}'.format(wfp))


def dump_rendered_img(vertex, img_fp, wfp=None, show_flag=False, alpha=0.8, face=True):
    if face:
        triangles = _to_ctype(tri_).astype(np.int32)  # for type compatible
    else:
        triangles = _to_ctype(tri).astype(np.int32)

    img = imageio.imread(img_fp)
    # img = cv2.resize(img, (std_size, std_size))
    img_ = img.astype(np.float32) / 255.

    vertices = vertex.T  # 3xm
    vertices = _to_ctype(vertices)  # for type compatible

    app = RenderPipeline(**cfg)

    overlap = app(vertices, triangles, img_)
    img_render = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)


    if wfp is not None:
        imageio.imwrite(wfp, img_render)
        # cv2.imwrite(wfp, img_render)
        print("saved to ", wfp)

    if show_flag:
        plt.imshow(img_render)
        plt.show()

    return img_render


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = reconstruct_vertex(param, dense=dense, transform=transform)
    vertex = rescale_w_roi(vertex, roi_bbox)

    return vertex


def rescale_w_roi(vert, roi_bbox):
    vertex = vert.copy()
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box, transform=True):
    return _predict_vertices(param, roi_box, dense=True, transform=transform)


def get_landmarks(vert, face=True):
    if face:
        pts68 = keypoints_
    else:
        pts68 = keypoints
    # vert = vert.T.reshape(-1, 1)
    # lms = vert[pts68].reshape(-1, 3).T  # 3x68
    
    lms = vert[:,pts68] # 3x68
    return lms


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, transform=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if style == 'simple':
            plt.plot(pts[i][0, :], pts[i][1, :], 'o', markersize=4, color='g')

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
    if show_flag:
        plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
