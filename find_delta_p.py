import numpy as np
from bernstein_ffd.ffd_utils import *

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz, reconstruct_vertex, LpDataset, _parse_param, _parse_ffd_param
import cv2
from utils.render_simdr import render
from utils.io import _load, _numpy_to_cuda
from utils.params import *
import time
# from multiprocessing import Pool
import multiprocessing

filelists_train ='train.configs/train_aug_120x120.list.train'
filelists_val = 'train.configs/train_aug_120x120.list.val'
root ='../Datasets/train_aug_120x120/'
param_fp_train = 'train.configs/param_all_full.pkl'
# param_fp_train = 'train.configs/param_all.pkl'
param_fp_val ='train.configs/param_all_val_full.pkl'

normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization


train_dataset = DDFADataset(
    root=root,
    filelists=filelists_train,
    param_fp=param_fp_train,
    transform=transforms.Compose([ToTensorGjz()])#, normalize])
)
val_dataset = DDFADataset(
    root=root,
    filelists=filelists_val,
    param_fp=param_fp_val,
    transform=transforms.Compose([ToTensorGjz()])#, normalize])
)


"""bfm mean shape as reference mesh"""
# todo - decide dimension

def svd_solve(a, b):
    [U, s, Vt] = np.linalg.svd(a, full_matrices=False)
    r = max(np.where(s >= 1e-12)[0])
    # Sigma = np.diag(s[:r], np.zeros((s[r:].shape)))
    temp = np.dot(U[:, :r].T, b) / s[:r]
    return np.dot(Vt[:r, :].T, temp)


# reference_mesh = u_.reshape(3, -1, order='F') # 3 x 35709
# # dic = test_face_ffd(reference_mesh.T, faces, n=(6, 6, 6)) 
# dic = test_face_ffd(reference_mesh.T, faces, n=(3, 3, 3)) 
# deform_matrix = dic["b"] #(38365, 216)
# control_points = dic["p"] #(216, 3)
# cp_num = control_points.reshape(-1).shape[0]

# new_mesh = reference_mesh.copy()
# new_mesh[:, keypoints_] += 1

# cp_estimated = np.linalg.lstsq(deform_matrix, reference_mesh.T)[0]
# delta_p = cp_estimated - control_points# 343 x 3

# def save_delta_p(dataset):
#     for i in range(len(dataset)):
#         find_delta_p(i)

# def find_delta_p(gt_param):
#     # start = time.time()
#     # gt_param = train_dataset[i][1].numpy()

#     p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
#     gt_vert = (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F')
#     # gt_transformed_vert = p @ gt_vert + offset

#     # Ax = b
#     # begin = time.time()
#     cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)[0]
#     delta_p = cp_estimated - control_points# 343 x 3
#     # print("lstsq took: ", time.time() - begin)

#     delta_p = delta_p.reshape(1, -1)
#     # delta_ps[i] = delta_p

#     # print("/636252 ", time.time() - start)


# if __name__ == '__main__':
#     param_lst = list(train_dataset.params.numpy())[:10]
#     delta_ps = np.zeros((len(train_dataset), cp_num))
#     # for i in range(len(train_dataset)):
#     start = time.time()
#     pool = multiprocessing.Pool(processes=10)
#     pool.map(find_delta_p, param_lst)
#     pool.close()
#     pool.join()
#     print(time.time() - start)


# HELEN_HELEN_3036412907_2_0_1.jpg
delta_ps = np.zeros((len(train_dataset), (12+cp_num)))
for i in range(0, len(train_dataset), 1000):
    start = time.time()
    # i = 410807
    gt_param = train_dataset[i][1].numpy()

    p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
    # gt_vert = (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F')
    gt_vert = (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F')
    gt_transformed_vert = p @ gt_vert + offset

    print(train_dataset.lines[i], ":", gt_transformed_vert[0, :].min(), gt_transformed_vert[0, :].max(), gt_transformed_vert[1, :].min(), gt_transformed_vert[1, :].max())
    # img = cv2.imread(root + train_dataset.lines[i])
    # render(img, [gt_transformed_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
    # dump_to_ply(gt_vert, tri_.T, "train.configs/HELEN_HELEN_3036412907_2_0_1_wo_pose.ply", transform=False)

    # # Ax = b
    # begin = time.time()
    # cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)[0]
    # delta_p = cp_estimated - control_points# 343 x 3
    # # print("lstsq took: ", time.time() - begin)

    # delta_p = delta_p.reshape(1, -1)
    # new_gt = np.concatenate((gt_param[:12].reshape(1, -1), delta_p), 1)
    # delta_ps[i] = new_gt
    
    # """temp"""
    # pg, offsetg, delta_p = _parse_ffd_param(new_gt.T)
    # gt_vert = pg @ deformed_vert(delta_p, transform=False) + offsetg

    # dump_to_ply(gt_vert, tri_.T, "samples/outputs/test_696.ply", transform=False)
    # if i > 0 and i % 10000 == 0:
    #     with open(f"train.configs/delta_p_non_rigid_train_{i}.pkl", "wb") as f:
    #         pickle.dump(delta_ps, f, protocol=4)

    # print(i, "/636252 ", time.time() - start)

# with open(f"train.configs/delta_p_non_rigid_train.pkl", "wb") as f:
#     pickle.dump(delta_ps, f, protocol=4)


delta_ps_val = np.zeros((len(val_dataset), (12+cp_num))) 
for i in range(len(val_dataset)):
    start = time.time()
    gt_param = train_dataset[i][1].numpy()

    p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
    gt_vert = (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F')
    # gt_transformed_vert = p @ gt_vert + offset

    # Ax = b
    begin = time.time()
    cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)[0]
    delta_p = cp_estimated - control_points# 343 x 3
    # print("lstsq took: ", time.time() - begin)

    delta_p = delta_p.reshape(1, -1)
    new_gt = np.concatenate((gt_param[:12].reshape(1, -1), delta_p), 1)
    delta_ps_val[i] = new_gt


    if i > 0 and i % 10000 == 0:
        with open(f"train.configs/delta_p_non_rigid_val_{i}.pkl", "wb") as f:
            pickle.dump(delta_ps_val, f, protocol=4)

    print(i, "/", len(val_dataset), time.time() - start)

# with open(f"train.configs/delta_p_non_rigid_val.pkl", "wb") as f:
#     pickle.dump(delta_ps_val, f, protocol=4)


    # reconstructed = p @ (deform_matrix @ (control_points + delta_p)).T + offset

    # img = cv2.imread(root + train_dataset.lines[i])
    # render(img, [gt_transformed_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
    
    # indices = []
    # for i in range(cp_estimated.shape[0]):
    #     if gt_vert[0].min()*2 <= cp_estimated[i, 0] <= gt_vert[0].max()*2 \
    #     and gt_vert[1].min()*2 <= cp_estimated[i, 1] <= gt_vert[1].max()*2 \
    #     and gt_vert[2].min() - (gt_vert[2].max() - gt_vert[2].min()) / 2 <= cp_estimated[i, 2] <= gt_vert[2].max() + (gt_vert[2].max() - gt_vert[2].min()) / 2:
    #         indices.append(i)
    # print(indices)
    # print(len(indices) / cp_estimated.shape[0])

    # x_count = 0
    # for i in range(cp_estimated[:, 0].shape[0]):
    #     if gt_vert[0].min() <= cp_estimated[i, 0] <= gt_vert[0].max():
    #         x_count += 1 
    # x_fraction = x_count / cp_estimated[:, 0].shape[0]

    # y_count = 0
    # for i in range(cp_estimated[:, 1].shape[0]):
    #     if gt_vert[1].min() <= cp_estimated[i, 1] <= gt_vert[1].max():
    #         y_count += 1 
    # y_fraction = y_count / cp_estimated[:, 0].shape[0]

    # z_count = 0
    # for i in range(cp_estimated[:, 0].shape[0]):
    #     if gt_vert[2].min() <= cp_estimated[i, 2] <= gt_vert[2].max():
    #         z_count += 1 
    # z_fraction = z_count / cp_estimated[:, 0].shape[0]

    # BdeltaP = v - BP
    # begin = time.time()
    # delta_p_x = svd_solve(deform_matrix, (gt_vert[0].T - deform_matrix @ control_points[:, 0]))
    # delta_p_y = svd_solve(deform_matrix, (gt_vert[1].T - deform_matrix @ control_points[:, 1]))
    # delta_p_z = svd_solve(deform_matrix, (gt_vert[2].T - deform_matrix @ control_points[:, 2]))
    # delta_p_ = np.stack((delta_p_x, delta_p_y, delta_p_z), 1)
    # print("svd took: ", time.time() - begin)