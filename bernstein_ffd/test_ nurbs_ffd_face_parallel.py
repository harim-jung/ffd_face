
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Add a folder that contains fastai package: ffd_face/ fastai/fastai; The current python file "test_nurbs_ffd_face_parallel.py"
# is in ffd_face/berstein_ffd

import sys

sys.path.append('F:\\Dropbox\\Anaconda\\envs\\ffd_face\\bernstein_ffd\\ffd') #Add subfolder ffd  which contains deform_parallel.py module, under the current folder, which is ffd_face/bernstein_ffd


sys.path.append('F:\\Dropbox\\Anaconda\\envs\\ffd_face\\fastai\\fastcore') #".." refers to the parent of the current folder, which is bernstein_ffd

sys.path

import deform_parallel
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

from ffd import bernstein, deform, util


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

    return deform_parallel.get_reference_ffd_param_nurbs(xyz, U, V, W, P_lattice)


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


def chamfer_distance_without_batch(p1, p2, debug=False):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])

    p1 = p1.repeat(p2.size(1), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(0, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0])

    p2 = p2.repeat(p1.size(0), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=1)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist


def chamfer_distance_with_batch(p1, p2, debug=False):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    if debug:
        print(p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))
        print(p1[0][0])

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 2)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0][0])

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0][0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=3)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=2)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist


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
plydata = PlyData.read('../train.configs/HELEN_HELEN_3036412907_2_0_1_wo_pose.ply')
v = plydata['vertex']

vert = np.zeros((3, 35709))
for i, vt in enumerate(v):
 vert[:, i] = np.array(list(vt))

reference_mesh = vert

faces = tri_  # (76073, 3)

landmarks = keypoints_

"""find B and P"""

# %%

# define nurbs surface: we use the notation used in paper https://asmedigitalcollection.asme.org/computingengineering/article-abstract/8/2/024001/465778/Freeform-Deformation-Versus-B-Spline?redirectedFrom=fulltext

# extra parameters:
#  a,b,c: a+1, b+1, c+1 are the numbers of control points in the x, y, z direcitons
#  p,m,n: ( 1< p <= a, 1< m <= b, 1< n <= c)  the degrees of the basis functionns in u,v,w; default: p,m,n = 3
#  knot vectors:
#  U = (u0, u1,...,uq), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
#  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

# xyz = R(s,t,u) = (R_1(s,t,u), R_2(s,t,u), R_3(s,t,u) ) =
#      = sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u) ( p_{ijk}[0], p_{ijk}[1],p_{ijk}[2] )
#     = ( sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[0],
#         sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[1],
#          sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[2] )


# import pdb

# pdb.set_trace()
# stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# import pdb; pdb.set_trace()
a = 6  # a+1 is the number of contol points in the x direction
b = 6
c = 6


# define U,V,W, and P

#  knot vectors:
#  U = (u0, u1,...,uq ), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
#  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

p = m = n = 3  # the degree of basis functions

q = a + p + 1
r = b + m + 1
s = c + n + 1

U = np.zeros(shape=(q + 1,), dtype=np.float)
V = np.zeros(shape=(r + 1,), dtype=np.float)
W = np.zeros(shape=(s + 1,), dtype=np.float)

for i in range(q + 1):  # 0,1,2,....q: the u knot index

 if i <= p:
  U[i] = 0
 else:
  if i <= (q - p - 1):
   U[i] = i - p
  else:
   U[i] = q - 2 * p

for i in range(r + 1):
 if i <= m:
  V[i] = 0
 else:
  if i <= (r - m - 1):
   V[i] = i - m
  else:
   V[i] = r - 2 * m

for i in range(s + 1):
 if i <= n:
  W[i] = 0
 else:
  if i <= (s - n - 1):
   W[i] = i - n
  else:
   W[i] = s - 2 * n

print('U=', U)
print('V=', V)
print('W=', W)

dims = (a, b, c)

stu_origin, stu_axes = deform_parallel.get_stu_params(reference_mesh.T[landmarks])

#stu_axes contains the directions and the extents of the bounding box of the face

#control_points_dist = np.array(
#       [  [ 0,  1.0, 0.2, 0.5, 0.7, 0.9,1.0 ],
#          [ 0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0 ],
#          [  0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ]
#       ]
#)

control_points_dist = np.array(
    [
    [  1.0/a * i for i in range(a+1)  ],
    [   1.0/b * i for i in range(b+1)   ] ,
    [   1.0/c * i for i in range(c+1)   ]
    ]
)


control_points_dist

#a_control_points = [ stu_axes[0] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]
#b_control_points = [ stu_axes[1] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]
#c_control_points =  [ stu_axes[2] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]

#P_lattice = deform_parallel.get_control_points_nurbs(dims, stu_origin, stu_axes)

P_lattice = deform_parallel.get_control_points_nurbs_nonuniform(dims, stu_origin, stu_axes, control_points_dist)

print('control points in 3d lattice form=', P_lattice)

print('xyz.shape=', reference_mesh.T[landmarks].shape)

print('xyz.dtype=', reference_mesh.T.dtype)

# np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# np.double == np.float64
# z = 2.0 's type is float64

# np.double(reference_mesh.T[landmarks])

# https://forums.fast.ai/t/use-of-parallel-function-in-fastai-core/35704/6
# parallel( func, arr:Collection, max_works:int=None)
# call func on every element of arr in parallel using max_workers
# func must accept two values: both the value of the array element and the index of the array element e.g. def my_func(value, index)


deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T[landmarks].astype(np.double), faces, U, V, W,
                                                    P_lattice, sample_indices=None)

# deform_matrix = dic["b"]  # (38365, 216)

# control_points = dic["p"]  # (216, 3)
cp_num = control_points.shape[0]

print('deform_matrix=', deform_matrix)
print('control_points=', control_points)
print('num of control points=', cp_num)

print('deform_matrix.shape=', deform_matrix.shape)
print('control_points.shape=', control_points.shape)

reconstructed_vertices = deform_matrix @ control_points

# deform_matrix: M x N
# P_lattice:  n1 x n2 x n3 x 3

mesh = np.zeros(shape=(deform_matrix.shape[0], 3), dtype=np.double)

for l in range(deform_matrix.shape[0]):
 for i in range(P_lattice.shape[0]):  # i=0...a
  for j in range(P_lattice.shape[1]):  # j=0...b
   for k in range(P_lattice.shape[2]):  # k = 0...c

    # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
    # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
    # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
    # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

    mesh[l, :] += deform_matrix[
                   l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i, j, k,
                                                                                                    :]

# for l in range(deform_matrix.shape[0]):
for l in range(reference_mesh.T[landmarks].shape[0]):
 print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[landmarks][l]))
 print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
 print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix @ control_points)[l]))

