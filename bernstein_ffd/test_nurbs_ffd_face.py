import sys

# sys.path.append('F:/Dropbox/Anaconda/envs/ffd_face')
# sys.path.append("F:/Dropbox/Anaconda/envs/ffd_face/bernstein_ffd/ffd")
# sys.path.append('F:/Dropbox/Anaconda/envs/ffd_face/utils')

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
plydata = PlyData.read('train.configs/HELEN_HELEN_3036412907_2_0_1_wo_pose.ply')
v = plydata['vertex']

vert = np.zeros((3, 35709))
for i, vt in enumerate(v):
    vert[:, i] = np.array(list(vt))

reference_mesh = vert


faces = tri_  # (76073, 3)

landmarks = keypoints_

"""find B and P"""
# dic = test_face_ffd(reference_mesh.T, faces, n=(9, 9, 9))
# dic = test_face_ffd(reference_mesh.T, faces, n=(3, 6, 3))
# dic = test_face_ffd(reference_mesh.T, faces, n=(6, 9, 6))


# upper_index
# lower_index

# The point just below the nose
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


# control_points1_in_3d_lattice[:, n1y,:] and control_points2_in_3d_lattice[:, 0,:] contain the common control points on the interface
# plane of grid1 (defined by stu_origin1 and stu_axes1) and grid2 (defined by stu_origin2 and stu_axes2).

# The size of deltaP vector of the neural net will be the number of control points counting the common control points
# just once; it will be N1 + N2

# N1 = (n1x + 1) x (n1y + 1) x (n1z +1)
# N2 = (n1x +1) x (n2y ) x (n1z +1); The number of common control points is (n1x * n1z)
# When you apply the control points1 and the control points2 to the Bernstein matrix B1 and B2:
# do:
#  control_points1_in_3d_lattice[:,:,:].view(N,3) += deltaP[: N1] # deltaP: N x 3
#  control_points2_in_3d_lattice[:,1:,:].view(N,3) += deltaP[N1: N1+N2]
# Assign the common control points:
#   control_points2_in_3d_lattice[:,0:,:].view(N,3) += deltaP[N1 - (( n1x+1)  * (n1z+1) ) : N1]
#

# The number of the common control points are (n1x + 1) * (n1z + 1), which are located in control_points1[ ], and control_points2[ ]
# The output paramters for deltaP of the neural net contain do not contain the redundant control parameters, so some part of deltaP should be assigned to both control_points1
# control_points2 when performing the computation of deformed_mesh[lower_indices] = B1@(P10 + deltaP1); deformed_mesh[upper_indices] = B2@(P20 + deltaP2)
# cp_num_[:, :6, :]
# cp_num_[:, 6:, :]


# coord_range = vertices[:, mouth_index]
# # upper = 44.53444489630172
# # lower = 43.128484579586235
# # upper_ = 44.53444489630172
# # lower_ = 43.128484579586235
# # # upper_ = 40
# # # lower_ = 47
# cps = []
# # upper_ind = []
# # lower_ind = []
# for i, cp in enumerate(control_points_):
#     if coord_range[0].min() <= cp[0] <= coord_range[0].max():
#         if coord_range[1].min() <= cp[1] <= coord_range[1].max():
#             cps.append(i)
# #         if upper_ <= cp[1] <= upper:
# #             upper_ind.append(i)
# #         if lower_ <= cp[1] <= lower:
# #             lower_ind.append(i)
# #         # if cp[1] == upper:
# #         #     upper_ind.append(i)
# #         # if cp[1] == lower:
# #         #     lower_ind.append(i)
# # print(cps)
# # print(upper_ind)
# # print(lower_ind)

# # reference_mesh = (deform_matrix_ @ control_points_).T.astype(np.float32)
# # dump_to_ply(reference_mesh, tri_.T, f"samples/outputs/reference_mesh.ply", transform=False)

# # upper_vert = [5132, 5133, 5134, 5135, 5256, 5257, 5258, 5259, 5260, 5261, 5262, 5263, 5264, 5384, 5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6163, 6279, 6280, 6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6289, 6290, 6291, 6405, 6406, 6407, 6408, 6409, 6410, 6411, 6412, 6413, 6414, 6415, 6416, 6528, 6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6539, 6650, 6651, 6652, 6653, 6654, 6655, 6656, 6657, 6658, 6659, 6660, 6661, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7375, 7376, 7377, 7378, 7379, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7624, 7625, 7626, 7627, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 8095, 8096, 8097, 8098, 8099, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8225, 8226, 8227, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8347, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8575, 8576, 8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944, 8945, 8946, 8947, 9055, 9056, 9057, 9058, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067, 9175, 9176, 9177, 9178, 9179, 9180, 9181, 9182, 9183, 9184, 9185, 9186, 9187, 9295, 9296, 9297, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9306, 9307, 9415, 9416, 9417, 9418, 9419, 9420, 9421, 9422, 9423, 9424, 9425, 9426, 9427, 9535, 9536, 9537, 9538, 9539, 9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9655, 9656, 9657, 9658, 9659, 9660, 9661, 9662, 9663, 9664, 9665, 9666, 9667, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783, 9784, 9785, 9786, 9787, 9896, 9897, 9898, 9899, 9900, 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030, 10141, 10142, 10143, 10144, 10145, 10146, 10147, 10148, 10149, 10150, 10151, 10152, 10153, 10267, 10268, 10269, 10270, 10271, 10272, 10273, 10274, 10275, 10276, 10277, 10394, 10395, 10396, 10397, 10398, 10399, 10400, 10401, 10402, 10403, 10404, 10523, 10524, 10525, 10526, 10527, 10528, 10529, 10530, 10531, 10532, 10533, 10534, 10535, 10653, 10654, 10655, 10656, 10657, 10658, 10659, 10660, 10661, 10662, 10663, 10664, 10783, 10784, 10785, 10786, 10787, 10788, 10789, 10790, 10791, 10792, 10793, 10913, 10914, 10915, 10916, 10917, 10918, 10919, 10920, 10921, 10922, 11043, 11044, 11045, 11046, 11047, 11048, 11049, 11050, 11051, 11173, 11174, 11175, 11176, 11177, 11178, 11179, 11180]
# # lower_vert = [5265, 5394, 5395, 5523, 5524, 5525, 5526, 5652, 5653, 5654, 5655, 5656, 5657, 5781, 5782, 5783, 5784, 5785, 5786, 5787, 5788, 5910, 5911, 5912, 5913, 5914, 5915, 5916, 5917, 5918, 6038, 6039, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 6047, 6048, 6167, 6168, 6169, 6170, 6171, 6172, 6173, 6174, 6175, 6176, 6177, 6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6417, 6420, 6421, 6422, 6423, 6424, 6425, 6426, 6427, 6428, 6540, 6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550, 6551, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6907, 6908, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035, 7036, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7628, 7629, 7630, 7631, 7632, 7633, 7634, 7635, 7636, 7748, 7749, 7750, 7751, 7752, 7753, 7754, 7755, 7756, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7988, 7989, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 8108, 8109, 8110, 8111, 8112, 8113, 8114, 8115, 8116, 8228, 8229, 8230, 8231, 8232, 8233, 8234, 8235, 8236, 8348, 8349, 8350, 8351, 8352, 8353, 8354, 8355, 8356, 8468, 8469, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8588, 8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8828, 8829, 8830, 8831, 8832, 8833, 8834, 8835, 8836, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 9068, 9069, 9070, 9071, 9072, 9073, 9074, 9075, 9076, 9188, 9189, 9190, 9191, 9192, 9193, 9194, 9195, 9196, 9308, 9309, 9310, 9311, 9312, 9313, 9314, 9315, 9316, 9428, 9429, 9430, 9431, 9432, 9433, 9434, 9435, 9436, 9548, 9549, 9550, 9551, 9552, 9553, 9554, 9555, 9556, 9668, 9669, 9670, 9671, 9672, 9673, 9674, 9675, 9676, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9910, 9911, 9912, 9913, 9914, 9915, 9916, 9917, 9918, 9919, 10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10156, 10157, 10158, 10159, 10160, 10161, 10162, 10163, 10164, 10165, 10278, 10280, 10281, 10282, 10283, 10284, 10285, 10286, 10287, 10288, 10289, 10290, 10405, 10406, 10407, 10408, 10409, 10410, 10411, 10412, 10413, 10414, 10415, 10416, 10417, 10536, 10537, 10538, 10539, 10540, 10541, 10542, 10543, 10544, 10545, 10665, 10666, 10667, 10668, 10669, 10670, 10671, 10672, 10673, 10794, 10795, 10796, 10797, 10798, 10799, 10923, 10924, 10925, 11052, 11053, 11181, 11309]
# # better
# # upper_vert = [4872, 4873, 4874, 4875, 4876, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 5256, 5257, 5258, 5259, 5260, 5261, 5262, 5263, 5264, 5384, 5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6163, 6164, 6165, 6166, 6279, 6280, 6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6289, 6290, 6291, 6292, 6293, 6405, 6406, 6407, 6408, 6409, 6410, 6411, 6412, 6413, 6414, 6415, 6416, 6417, 6418, 6528, 6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6539, 6540, 6650, 6651, 6652, 6653, 6654, 6655, 6656, 6657, 6658, 6659, 6660, 6661, 6662, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7375, 7376, 7377, 7378, 7379, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7624, 7625, 7626, 7627, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 8095, 8096, 8097, 8098, 8099, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8225, 8226, 8227, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8347, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8575, 8576, 8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944, 8945, 8946, 8947, 9055, 9056, 9057, 9058, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067, 9175, 9176, 9177, 9178, 9179, 9180, 9181, 9182, 9183, 9184, 9185, 9186, 9187, 9295, 9296, 9297, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9306, 9307, 9415, 9416, 9417, 9418, 9419, 9420, 9421, 9422, 9423, 9424, 9425, 9426, 9427, 9535, 9536, 9537, 9538, 9539, 9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9655, 9656, 9657, 9658, 9659, 9660, 9661, 9662, 9663, 9664, 9665, 9666, 9667, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783, 9784, 9785, 9786, 9787, 9896, 9897, 9898, 9899, 9900, 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030, 10141, 10142, 10143, 10144, 10145, 10146, 10147, 10148, 10149, 10150, 10151, 10152, 10153, 10154, 10267, 10268, 10269, 10270, 10271, 10272, 10273, 10274, 10275, 10276, 10277, 10278, 10279, 10394, 10395, 10396, 10397, 10398, 10399, 10400, 10401, 10402, 10403, 10404, 10405, 10406, 10523, 10524, 10525, 10526, 10527, 10528, 10529, 10530, 10531, 10532, 10533, 10534, 10535, 10653, 10654, 10655, 10656, 10657, 10658, 10659, 10660, 10661, 10662, 10663, 10664, 10783, 10784, 10785, 10786, 10787, 10788, 10789, 10790, 10791, 10792, 10793, 10913, 10914, 10915, 10916, 10917, 10918, 10919, 10920, 10921, 10922, 11043, 11044, 11045, 11046, 11047, 11048, 11049, 11050, 11051, 11173, 11174, 11175, 11176, 11177, 11178, 11179, 11180, 11303, 11304, 11305, 11306, 11307, 11308, 11309, 11433, 11434, 11435, 11436, 11437, 11565]
# # lower_vert = [5136, 5265, 5266, 5394, 5395, 5396, 5523, 5524, 5525, 5526, 5652, 5653, 5654, 5655, 5656, 5781, 5782, 5783, 5784, 5785, 5786, 5787, 5788, 5910, 5911, 5912, 5913, 5914, 5915, 5916, 5917, 5918, 6039, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 6047, 6048, 6167, 6168, 6169, 6170, 6171, 6172, 6173, 6174, 6175, 6176, 6177, 6294, 6295, 6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6419, 6420, 6421, 6422, 6423, 6424, 6425, 6426, 6427, 6428, 6541, 6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550, 6551, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6907, 6908, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035, 7036, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7628, 7629, 7630, 7631, 7632, 7633, 7634, 7635, 7636, 7748, 7749, 7750, 7751, 7752, 7753, 7754, 7755, 7756, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7988, 7989, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 8108, 8109, 8110, 8111, 8112, 8113, 8114, 8115, 8116, 8228, 8229, 8230, 8231, 8232, 8233, 8234, 8235, 8236, 8348, 8349, 8350, 8351, 8352, 8353, 8354, 8355, 8356, 8468, 8469, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8588, 8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8828, 8829, 8830, 8831, 8832, 8833, 8834, 8835, 8836, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 9068, 9069, 9070, 9071, 9072, 9073, 9074, 9075, 9076, 9188, 9189, 9190, 9191, 9192, 9193, 9194, 9195, 9196, 9308, 9309, 9310, 9311, 9312, 9313, 9314, 9315, 9316, 9428, 9429, 9430, 9431, 9432, 9433, 9434, 9435, 9436, 9548, 9549, 9550, 9551, 9552, 9553, 9554, 9555, 9556, 9668, 9669, 9670, 9671, 9672, 9673, 9674, 9675, 9676, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9909, 9910, 9911, 9912, 9913, 9914, 9915, 9916, 9917, 9918, 9919, 10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10155, 10156, 10157, 10158, 10159, 10160, 10161, 10162, 10163, 10164, 10165, 10280, 10281, 10282, 10283, 10284, 10285, 10286, 10287, 10288, 10289, 10290, 10407, 10408, 10409, 10410, 10411, 10412, 10413, 10414, 10415, 10416, 10417, 10536, 10537, 10538, 10539, 10540, 10541, 10542, 10543, 10544, 10545, 10665, 10666, 10667, 10668, 10669, 10670, 10671, 10672, 10673, 10794, 10795, 10796, 10797, 10923, 10924, 10925, 11052, 11053, 11181]
# # narrower
# upper_vert = [5512, 5513, 5514, 5640, 5641, 5642, 5643, 5644, 5645, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6163, 6164, 6165, 6166, 6279, 6280, 6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6289, 6290, 6291, 6292, 6293, 6405, 6406, 6407, 6408, 6409, 6410, 6411, 6412, 6413, 6414, 6415, 6416, 6417, 6418, 6528, 6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6539, 6540, 6650, 6651, 6652, 6653, 6654, 6655, 6656, 6657, 6658, 6659, 6660, 6661, 6662, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7375, 7376, 7377, 7378, 7379, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7624, 7625, 7626, 7627, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 8095, 8096, 8097, 8098, 8099, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8225, 8226, 8227, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8347, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8575, 8576, 8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944, 8945, 8946, 8947, 9055, 9056, 9057, 9058, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067, 9175, 9176, 9177, 9178, 9179, 9180, 9181, 9182, 9183, 9184, 9185, 9186, 9187, 9295, 9296, 9297, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9306, 9307, 9415, 9416, 9417, 9418, 9419, 9420, 9421, 9422, 9423, 9424, 9425, 9426, 9427, 9535, 9536, 9537, 9538, 9539, 9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9655, 9656, 9657, 9658, 9659, 9660, 9661, 9662, 9663, 9664, 9665, 9666, 9667, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783, 9784, 9785, 9786, 9787, 9896, 9897, 9898, 9899, 9900, 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030, 10141, 10142, 10143, 10144, 10145, 10146, 10147, 10148, 10149, 10150, 10151, 10152, 10153, 10267, 10268, 10269, 10270, 10271, 10272, 10273, 10274, 10275, 10276, 10277, 10394, 10395, 10396, 10397, 10398, 10399, 10400, 10401, 10402, 10403, 10404, 10523, 10524, 10525, 10526, 10527, 10528, 10529, 10530, 10531, 10532, 10533, 10653, 10654, 10655, 10656, 10657, 10658, 10659, 10783, 10784, 10785, 10786, 10913]
# lower_vert = [6167, 6168, 6169, 6170, 6171, 6172, 6173, 6174, 6175, 6176, 6177, 6294, 6295, 6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6419, 6420, 6421, 6422, 6423, 6424, 6425, 6426, 6427, 6428, 6541, 6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550, 6551, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6907, 6908, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035, 7036, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7628, 7629, 7630, 7631, 7632, 7633, 7634, 7635, 7636, 7748, 7749, 7750, 7751, 7752, 7753, 7754, 7755, 7756, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7988, 7989, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 8108, 8109, 8110, 8111, 8112, 8113, 8114, 8115, 8116, 8228, 8229, 8230, 8231, 8232, 8233, 8234, 8235, 8236, 8348, 8349, 8350, 8351, 8352, 8353, 8354, 8355, 8356, 8468, 8469, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8588, 8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8828, 8829, 8830, 8831, 8832, 8833, 8834, 8835, 8836, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 9068, 9069, 9070, 9071, 9072, 9073, 9074, 9075, 9076, 9188, 9189, 9190, 9191, 9192, 9193, 9194, 9195, 9196, 9308, 9309, 9310, 9311, 9312, 9313, 9314, 9315, 9316, 9428, 9429, 9430, 9431, 9432, 9433, 9434, 9435, 9436, 9548, 9549, 9550, 9551, 9552, 9553, 9554, 9555, 9556, 9668, 9669, 9670, 9671, 9672, 9673, 9674, 9675, 9676, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9909, 9910, 9911, 9912, 9913, 9914, 9915, 9916, 9917, 9918, 9919, 10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10157, 10158, 10159, 10160, 10161, 10162, 10163, 10164, 10165]

# for j in range(0, 1):
#     # deform = np.random.uniform(low=-30, high=-20, size=(18180,))
#     # deform_ = np.random.uniform(low=20, high=30, size=(18180,))
#     # d = np.append(deform, deform_)
#     # np.random.shuffle(d)
#     # deform = d.reshape(cp_num_//3, -1)

#     deform = np.zeros((12120, 3))

#     # deform[upper_ind, 1] = 70 #np.random.randint(20, 30)
#     # # deform[cps, 1] = 50
#     # deform[lower_ind, 1] = -70 #np.random.randint(-30, -20)

#     # deform = np.random.uniform(low=-5, high=5, size=(12120,1))
#     # deform_1 = np.append(np.random.uniform(low=20, high=30, size=(6060,1)), np.random.uniform(low=20, high=30, size=(6060,1)), axis=0)
#     # deform_2 = np.random.uniform(low=-5, high=5, size=(12120,1))

#     # de = np.append(deform, deform_1, axis=1)
#     # deform = np.append(de, deform_2, axis=1)
#     # np.random.shuffle(deform)
#     # for i, cp in enumerate(deform):
#     #     if i not in cps:
#     #         deform[i] = np.zeros(3)

#     deformed_vert = np.zeros((deform_matrix_.shape[0], 3))
#     # new_cp = control_points_ + deform
#     for i in range(deform_matrix_.shape[0]):
#         deform = np.append(np.random.uniform(low=-10, high=10, size=(12120,2)), np.zeros((12120,1)), axis=1)
#         # np.random.uniform(low=-10, high=10, size=(12120,2))

#         # deform[upper_ind, 1] = 30 #np.random.randint(20, 30)
#         # # deform[cps, 1] = 50
#         # deform[lower_ind, 1] = -30 #np.random.randint(-30, -20)
#         if i in upper_vert:
#             # new_cp[lower_ind] = 0
#             # control_points_[lower_ind] = 0
#             deform[lower_ind] = 0
#         elif i in lower_vert:
#             # new_cp[upper_ind] = 0
#             # control_points_[upper_ind] = 0
#             deform[upper_ind] = 0

#         deform_ = deform_matrix_[i] @ (control_points_ + deform)
#         deformed_vert[i] = deform_

#     deformed_vert = deformed_vert.T.astype(np.float32)
#     # deformed_vert = (deform_matrix_ @ (control_points_ + deform)).T.astype(np.float32)

#     # img_ori = cv2.imread('samples/inputs/image00066.jpg')
#     # wfp = 'samples/outputs/test_render.jpg'
#     # render(img_ori, [deformed_vert * 1.2], tri_, alpha=0.8, show_flag=True, wfp=wfp, with_bg_flag=True, transform=True)
#     dump_to_ply(deformed_vert, tri_.T, f"samples/outputs/deform_all.ply", transform=False)


# """full face"""
# full_vert = p @ mean.reshape(3, -1, order='F') # (3, 38365)
# full_vert[0] -= full_vert[0].min()
# full_vert[0] += (std_size - full_vert[0].max()) / 2
# full_vert[1] -= full_vert[1].min()
# full_vert[1] += (std_size - full_vert[1].max()) / 2
# # shift z to start from 0
# full_vert[2] -= full_vert[2].min()

# dic_full = test_face_ffd(full_vert.T, faces, n=5)
# deform_matrix_f = dic_full["b"] #(38365, 216)
# control_points_f = dic_full["p"] #(216, 3)
# cp_num_f = control_points_f.reshape(-1).shape[0]

# vert = deform_matrix_f @ control_points_f


# dic = test_face_ffd(vertices.T, faces, n=5) # (5 + 1) ** 3 = 216 control points
# dic = test_face_ffd(vertices.T, faces, n=3) # (3 + 1) ** 3 = 64 control points
# dic = test_face_ffd(vertices.T, faces, n=9) # (9 + 1) ** 3 = 1000 control points
# deform_matrix = dic["b"] #(38365, 64)
# control_points = dic["p"] #(64, 3)
# # vert = b @ p # (p + dp)


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


#import pdb

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

U = np.zeros(shape=(q + 1,), dtype=np.double)
V = np.zeros(shape=(r + 1,), dtype=np.double)
W = np.zeros(shape=(s + 1,), dtype=np.double)

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

stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T[landmarks])

P_lattice = deform.get_control_points_nurbs(dims, stu_origin, stu_axes)

print('control points in 3d lattice form=', P_lattice)

print('xyz.shape=', reference_mesh.T[landmarks].shape)

print('xyz.dtype=', reference_mesh.T.dtype )

#np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# np.double == np.float64
# z = 2.0 's type is float64

# np.double(reference_mesh.T[landmarks])
deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T[landmarks].astype(np.double), faces,  U, V, W, P_lattice, sample_indices = None)

#deform_matrix = dic["b"]  # (38365, 216)

#control_points = dic["p"]  # (216, 3)
cp_num = control_points.shape[0]

print('deform_matrix=', deform_matrix)
print('control_points=', control_points)
print('num of control points=', cp_num)

print('deform_matrix.shape=', deform_matrix.shape)
print('control_points.shape=', control_points.shape)

reconstructed_vertices = deform_matrix @ control_points

# deform_matrix: M x N
# P_lattice:  n1 x n2 x n3 x 3

mesh = np.zeros( shape=( deform_matrix.shape[0], 3), dtype=np.double)

for l in range( deform_matrix.shape[0]):
  for i in range(P_lattice.shape[0]):  # i=0...a
      for j in range(P_lattice.shape[1]):  # j=0...b
         for k in range(P_lattice.shape[2]):  # k = 0...c

            # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
            # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
            # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
            # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

            mesh[l,:] += deform_matrix[l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i,j,k,:]

#for l in range(deform_matrix.shape[0]):
for l in range( reference_mesh.T[landmarks].shape[0] ):
  print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[landmarks][l]))
  print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
  print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix@control_points)[l] ) )


