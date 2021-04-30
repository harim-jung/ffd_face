# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ffd_face.Sim3DR import RenderPipeline


cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(3, height / width * 3))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True, transform=False): # transform = True in our experiment
    tri = _to_ctype(tri)

    if transform:
        img = cv2.flip(img, 0) # img = original image:  vertical flipping: flip the image around the x-axis, by passing the value 0 as second argument

    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap) # overlap = bg = depth buffer

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if transform:    
        res = cv2.flip(res, 0) # vertical flip the rendered image

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res
