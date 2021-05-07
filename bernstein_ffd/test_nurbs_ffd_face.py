#pip install plyfile

import sys

sys.path.append('F:\\Dropbox\\Anaconda\\envs\\ffd\\ffd_face')
sys.path.append("F:\\Dropbox\\Anaconda\\envs\\ffd\\ffd_face\\bernstein_ffd\\ffd")
sys.path.append('F:\\Dropbox\\Anaconda\\envs\\ffd\\ffd_face\\utils')

sys.path
import os
import numpy as np
import torch
from utils.params import *
# => e.g. d = make_abs_path('../train.configs')
#   68 landmarks
#   keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
# from params import *
from math import cos, sin, atan2, sqrt
from utils.inference import dump_to_ply
from utils.ddfa import get_rot_mat_from_axis_angle_np, get_rot_mat_from_axis_angle
# from utils.render_simdr import render
import cv2
from plyfile import PlyData, PlyElement
from utils.params import keypoints_

from bernstein_ffd.ffd import bernstein, deform, util


def test_face_ffd_nurbs(vertices, faces, U, V, W, P_lattice, sample_indices=None):
    # pdb.set_trace()
    b, p = _calculate_ffd_nurbs(vertices, faces, U, V, W, P_lattice, sample_indices)
    return b,p # dict(b=b, p=p)


def _calculate_ffd_nurbs(vertices, faces, U, V, W, P_lattice, sample_indices):
    # import bernstein_ffd.ffd.deform as ffd
    # import util3d.mesh.sample as sample
    # stu_origin, stu_axes = ffd.get_stu_params(vertices)
    if sample_indices is None:
        xyz = vertices
    else:
        xyz = vertices[sample_indices]

    return deform.get_reference_ffd_param_nurbs(xyz, U, V, W, P_lattice)


# def get_reference_ffd_param(vertices, dims, stu_origin=None, stu_axes=None):
#    if stu_origin is None or stu_axes is None:
#        if not (stu_origin is None and stu_axes is None):
#            raise ValueError(
#                'Either both or neither of stu_origin/stu_axes must be None')
#        stu_origin, stu_axes = get_stu_params(vertices)
#    b = get_deformation_matrix(vertices, dims, stu_origin, stu_axes)
#    p = get_control_points(dims, stu_origin, stu_axes)
#    return b, p


def sample_triangle(v, n=None):
    if hasattr(n, 'dtype'):
        n = np.asscalar(n)
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert (v.shape[-2] == 2)
    a = np.random.uniform(size=size)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a * v, axis=-2)


def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    tris = vertices[faces]
    n_faces = len(faces)
    d0 = tris[..., 0:1, :]
    ds = tris[..., 1:, :] - d0
    assert (ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1]) ** 2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)


def deformed_vert(deform, transform=False, face=True):
    if face:
        dm = deform_matrix
        cp = control_points
    else:
        dm = deform_matrix_f
        cp = control_points_f

    deform = deform.reshape(cp_num // 3, -1)
    deformed_vert = (dm @ (cp + deform)).T.astype(np.float32)
    if transform:
        deformed_vert[1, :] = std_size + 1 - deformed_vert[1, :]
    return deformed_vert


def deformed_vert_w_pose(params, transform=False, rewhiten=True, pose='rot_mat'):
    if rewhiten:
        params[:12] = params[:12] * param_full_std[:12] + param_full_mean[:12]

    if pose == 'rot_mat':
        p_ = params[:12].reshape(3, -1)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        deform = params[12:].reshape(cp_num // 3, -1)
    elif pose == 'axis_angle':
        # s = pose_param[:, 0].view(batch, 1)
        s = np.abs(params[0])
        axis_angle = params[1:4]
        offset = params[4:7].reshape(3, 1)
        r = get_rot_mat_from_axis_angle_np(axis_angle)
        # r_ = get_rot_mat_from_axis_angle(axis_angle)
        p = s * r
        deform = params[7:].reshape(cp_num // 3, -1)

    deformed_vert = p @ (deform_matrix @ (control_points + deform)).T + offset
    if transform:
        deformed_vert[1, :] = std_size + 1 - deformed_vert[1, :]
    return deformed_vert.astype(np.float32)


"""reference meshes"""

# the  face just below the nose
# face# 15997 : vert# (8084 8204 8203)
# face vert 0 : vert# 8084
# position [61.493000 63.313400 -32.137001]
# normal [-0.141119 -4.286755 4.582139]
# face vert 1 : vert# 8204
# position [62.125301 63.311699 -32.132500]
# normal [0.385106 -4.480810 4.356285]
# face vert 2 : vert# 8203
# position [62.130501 63.812500 -31.638201]
# normal [0.303801 -4.853062 3.926230]

# %%


"""Augmented LP reference mesh (only rigid part wo pose) (HELEN_HELEN_3036412907_2_0_1_wo_pose.ply)"""
plydata = PlyData.read('train.configs/HELEN_HELEN_3036412907_2_0_1_wo_pose.ply')
v = plydata['vertex']

vert = np.zeros((3, 35709))
for i, vt in enumerate(v):
    vert[:, i] = np.array(list(vt))

reference_mesh = vert


faces = tri_  # (76073, 3)

landmarks = keypoints_


