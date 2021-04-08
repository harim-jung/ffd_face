#!/usr/bin/env python3
# coding: utf-8
import os
from scipy.io import loadmat

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


import os.path as osp
import numpy as np
from .io import _load
import pickle


def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


"""includes neck and ears (40, 10 dim)"""
d = make_abs_path('../train.configs')
# 68 landmarks
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
# shape and expression basis
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
# mean shape and expression
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp

# w = np.concatenate((w_shp, w_exp), axis=1)
# w_base = w[keypoints]
# w_norm = np.linalg.norm(w, axis=0)
# w_base_norm = np.linalg.norm(w_base, axis=0)

# for landmarks
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]


"""neck and ears (199, 29 dim)"""
shape_mat = osp.join(d, "Model_Shape.mat")
exp_mat = osp.join(d, "Model_Exp.mat")
shape_params = loadmat(shape_mat)
exp_params = loadmat(exp_mat)

mean_shp = shape_params["mu_shape"]  # 159645x1 single
mean_exp = exp_params["mu_exp"]  # 159645x1 single
mean = mean_shp + mean_exp
w_shp_full = shape_params["w"]  # 159645x199 single
w_exp_full = exp_params["w_exp"]  # 159645x29 double
tri = (shape_params["tri"] - 1).T # 105840x3 double # subtract 1 to match python index


"""face only (3 x 35709) (199, 29 dim)"""
# sampling_indices = _load(osp.join(d, '35709_indices.npy')).astype(int)
# u_ = np.reshape(mean, [-1, 3, 1])[sampling_indices, :].reshape([-1, 1]).astype(np.float32)
# w_shp_ = np.reshape(w_shp_full, [-1, 3, 199])[sampling_indices, :, :].reshape([-1, 199]).astype(np.float32)
# w_exp_ = np.reshape(w_exp_full, [-1, 3, 29])[sampling_indices, :, :].reshape([-1, 29]).astype(np.float32)
# tri_ = loadmat(osp.join(d, "35709_tri.mat"))["tri"].astype(np.int) - 1  # (70789, 3)
# keypoints_ = _load(osp.join(d, '35709_keypoints.pkl'))


"""face only (3 x 38365) (40, 10 dim)"""
# bfm = _load(osp.join(d, 'bfm_noneck_v3.pkl'))
# u_ = bfm.get('u').astype(np.float32)
# w_shp_ = bfm.get('w_shp').astype(np.float32)[..., :40]
# w_exp_ = bfm.get('w_exp').astype(np.float32)[..., :10]
# w_ = np.concatenate((w_shp, w_exp), axis=1)
# tri_ = _load(osp.join(d, 'face_tri.pkl')).T # (76073, 3)
# keypoints_ = _load(osp.join(d, '38365_keypoints.pkl'))


# temp
# sampled = []
# for pt in w_shp_:
#     index = np.where(w_shp_full == pt[0])
#     sampled.append(index[0][0])

# with open(osp.join(d, '38365_indices_flat.pkl'), "wb") as f:
#     pickle.dump(np.array(sampled), f)


"""Coarse Data - neck and ears (199, 29 dim)"""
shape_mat = osp.join(d, "Model_Shape.mat")
exp_mat = osp.join(d, "Model_Exp.mat")
shape_params = loadmat(shape_mat)
exp_params = loadmat(exp_mat)

mean_shp = shape_params["mu_shape"]  # 159645x1 single
mean_exp = exp_params["mu_exp"]  # 159645x1 single
mean = mean_shp + mean_exp
w_shp_full = shape_params["w"]  # 159645x199 single
w_exp_full = exp_params["w_exp"]  # 159645x29 double
tri = (shape_params["tri"] - 1).T # 105840x3 double # subtract 1 to match python index

# load Exp_Pca.bin
root = '../Datasets/Coarse_Dataset/'
with open(root + "Exp_Pca.bin", mode='rb') as file: # b is important -> binary
    pca = np.fromfile(file, np.float32)
    pca = pca[1:] # 12771600

    mean_exp_c = pca[:3*53215].reshape(-1, 1) # expression mean (159645,1)
    w_exp_full_c = pca[3*53215:].reshape(3*53215, 79, order='F') # expression basis (159645, 79)

w_shp_full_c = w_shp_full[:, :100]
mean_c = mean_shp + mean_exp_c


"""Coarse Data - face only (3 x 35709) (199, 29 dim)
https://github.com/microsoft/Deep3DFaceReconstruction/tree/master/BFM"""

sampling_indices = _load(osp.join(d, '35709_indices.npy')).astype(int)
u_c = np.reshape(mean_c, [-1, 3, 1])[sampling_indices, :].reshape([-1, 1]).astype(np.float32)
w_shp_c = np.reshape(w_shp_full_c, [-1, 3, 100])[sampling_indices, :, :].reshape([-1, 100]).astype(np.float32)
w_exp_c = np.reshape(w_exp_full_c, [-1, 3, 79])[sampling_indices, :, :].reshape([-1, 79]).astype(np.float32)


"""300W-LP - face only (3 x 35709) (199, 29 dim)"""
u_ = np.reshape(mean, [-1, 3, 1])[sampling_indices, :].reshape([-1, 1]).astype(np.float32)
w_shp_ = np.reshape(w_shp_full, [-1, 3, 199])[sampling_indices, :, :].reshape([-1, 199]).astype(np.float32)
w_exp_ = np.reshape(w_exp_full, [-1, 3, 29])[sampling_indices, :, :].reshape([-1, 29]).astype(np.float32)


"""300W-LP - face only (3 x 35709) (40, 10 dim)"""
w_shp_lp = w_shp_[:, :40]
w_exp_lp = w_exp_[:, :10]


tri_ = loadmat(osp.join(d, "35709_tri.mat"))["tri"].astype(np.int) - 1  # (70789, 3)
keypoints_ = _load(osp.join(d, '35709_keypoints.pkl'))


"""for landmarks (3 x 35709)"""
keypoints_flat = _load(osp.join(d, "35709_keypoints_flat.pkl"))
u_base_ = u_[keypoints_flat].reshape(-1, 1)
w_shp_base_ = w_shp_[keypoints_flat]
w_exp_base_ = w_exp_[keypoints_flat]

# 300w-lp
w_shp_base_lp = w_shp_base_[:, :40]
w_exp_base_lp = w_exp_base_[:, :10]

# coarse data
u_base_c = u_c[keypoints_flat].reshape(-1, 1)
w_shp_base_c = w_shp_base_[:, :100]
w_exp_base_c = w_exp_c[keypoints_flat]


"""param_mean and param_std are used for re-whitening"""
params = _load(osp.join(d, 'param_whitening.pkl'))
param_mean = params.get('param_mean')
param_std = params.get('param_std')

params = _load(osp.join(d, 'param_all_full_mean_std.pkl'))
param_full_mean = params.get('param_mean')
param_full_std = params.get('param_std')


delta_p_params = _load('train.configs/delta_p_full_mean_std.pkl')
delta_p_mean = delta_p_params["delta_p_mean"]
delta_p_std = delta_p_params["delta_p_std"]

"""300w-lp V2"""
# params = _load(osp.join(d, 'param_mean_std_62d_120x120.pkl'))
# param_mean = params.get('mean')
# param_std = params.get('std')


"""
cropped 120 augmented ver
235 parameters (pitch, yaw, roll, scale, shape params, expression params)
"""
# img_dic = _load(osp.join(d, 'cropped_120/lp_cropped_img_mean_std_120.pkl'))
# lp_mean = img_dic["img_mean"] # BGR order (cv2)
# lp_std = img_dic["img_std"] # BGR order (cv2)

# coarse_dic = _load(osp.join(d, 'coarse_data/coarse_data_img_mean_std.pkl'))
# coarse_mean = coarse_dic["img_mean"]
# coarse_std = coarse_dic["img_std"]


"""Mouth area index (manually selected in blender)"""
# 382 indices
mouth_whole_index = _load(osp.join(d, '35709_mouth_index.pkl'))
mouth_index = _load(osp.join(d, '35709_mouth_inner.pkl'))
eye_index = _load(osp.join(d, '35709_eyes.pkl'))

"""landmark indices by region"""
upper_mouth = [5522, 5909, 10537, 10795, 8215, 8935, 10395, 6025, 7495, 9064, 8223, 7384]
lower_mouth = [5522, 5909, 10537, 10795, 6915, 7629, 7636, 8229, 8236, 8829, 8836, 9555]
upper_nose = [8161, 8177, 8187, 8192]
lower_nose = [6515, 7243, 8204, 9163, 9883]
left_brow = [28111, 28787, 29177, 29382, 29549]
right_brow = [30288, 30454, 30662, 31056, 31716]
left_eye = [2215, 3640, 3886, 4801, 4920, 5828]
right_eye = [10455, 11353, 11492, 12383, 12653, 14066]
contour_boundary = [16264, 16467, 16644, 16888, 27208, 27440, 27608, 27816, 32244, 32939, 33375, 33654, 33838, 34022, 34312, 34766, 35472]

upper_index = _load(osp.join(d, '35709_upper_index.pkl'))
lower_index = _load(osp.join(d, '35709_lower_index.pkl'))


# image resize
std_size = 120



# with open("train.configs/35709_upper_index.txt", "r") as f:
#     indices = np.array(f.read().split(",")).astype(int)
#     with open("train.configs/35709_upper_index.pkl", "wb") as p:
#         pickle.dump(indices, p)

# with open("train.configs/35709_lower_index.txt", "r") as f:
#     indices = np.array(f.read().split(",")).astype(int)
#     with open("train.configs/35709_lower_index.pkl", "wb") as p:
#         pickle.dump(indices, p)




# #!/usr/bin/env python3
# # coding: utf-8
# import os
# from scipy.io import loadmat

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# import os.path as osp
# import numpy as np
# from .io import _load
# import pickle

# def make_abs_path(d):
#     return osp.join(osp.dirname(osp.realpath(__file__)), d)


# """includes neck and ears (40, 10 dim)"""
# d = make_abs_path('../train.configs')
# # 68 landmarks
# keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
# # shape and expression basis
# w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
# w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
# # mean shape and expression
# u_shp = _load(osp.join(d, 'u_shp.npy'))
# u_exp = _load(osp.join(d, 'u_exp.npy'))
# u = u_shp + u_exp
# w = np.concatenate((w_shp, w_exp), axis=1)
# w_base = w[keypoints]
# w_norm = np.linalg.norm(w, axis=0)
# w_base_norm = np.linalg.norm(w_base, axis=0)

# # for landmarks
# u_base = u[keypoints].reshape(-1, 1)
# w_shp_base = w_shp[keypoints]
# w_exp_base = w_exp[keypoints]


# """neck and ears (199, 29 dim)"""
# shape_mat = osp.join(d, "Model_Shape.mat")
# exp_mat = osp.join(d, "Model_Exp.mat")
# shape_params = loadmat(shape_mat)
# exp_params = loadmat(exp_mat)

# mean_shp = shape_params["mu_shape"]  # 159645x1 single
# mean_exp = exp_params["mu_exp"]  # 159645x1 single
# mean = mean_shp + mean_exp
# w_shp_full = shape_params["w"]  # 159645x199 single
# w_exp_full = exp_params["w_exp"]  # 159645x29 double
# tri = (shape_params["tri"] - 1).T # 105840x3 double # subtract 1 to match python index


# """face only (3 x 38365) (40, 10 dim)"""
# bfm = _load(osp.join(d, 'bfm_noneck_v3.pkl'))
# u_ = bfm.get('u').astype(np.float32)
# w_shp_ = bfm.get('w_shp').astype(np.float32)[..., :40]
# w_exp_ = bfm.get('w_exp').astype(np.float32)[..., :10]
# w_ = np.concatenate((w_shp, w_exp), axis=1)
# tri_ = _load(osp.join(d, 'face_tri.pkl')).T # (76073, 3)
# keypoints_ = _load(osp.join(d, '38365_keypoints.pkl'))

# # keypoints_ = bfm.get('keypoints').astype(np.long)


# """face only (3 x 35709) (199, 29 dim)"""
# # sampling_indices = _load(osp.join(d, '35709_indices.npy')).astype(int)
# # u_ = np.reshape(mean, [-1, 3, 1])[sampling_indices, :].reshape([-1, 1]).astype(np.float32)
# # w_shp_ = np.reshape(w_shp_full, [-1, 3, 199])[sampling_indices, :, :].reshape([-1, 199]).astype(np.float32)
# # w_exp_ = np.reshape(w_exp_full, [-1, 3, 29])[sampling_indices, :, :].reshape([-1, 29]).astype(np.float32)
# # tri_ = loadmat(osp.join(d, "35709_tri.mat"))["tri"].astype(np.int) - 1  # (70789, 3)
# # keypoints_ = _load(osp.join(d, '35709_keypoints.pkl'))


# # param_mean and param_std are used for re-whitening
# params = _load(osp.join(d, 'param_whitening.pkl'))
# param_mean = params.get('param_mean')
# param_std = params.get('param_std')


# """
# cropped 120 augmented ver
# 235 parameters (pitch, yaw, roll, scale, shape params, expression params)
# """
# img_dic = pickle.load(open('train.configs/cropped_120/lp_cropped_img_mean_std_120.pkl', "rb"))
# lp_mean = img_dic["img_mean"] # BGR order (cv2)
# lp_std = img_dic["img_std"] # BGR order (cv2)


# # image resize
# std_size = 120
