import numpy as np
from bernstein_ffd.ffd_utils import *

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz, reconstruct_vertex, LpDataset, _parse_param
import cv2
from utils.render_simdr import render
from utils.io import _load, _numpy_to_cuda
from utils.params import *
import time

filelists_train ='train.configs/train_aug_120x120.list.train'
filelists_val = 'train.configs/train_aug_120x120.list.val'
root ='../Datasets/train_aug_120x120/'
param_fp_train = 'train.configs/param_all_new.pkl'
param_fp_val ='train.configs/param_all_val_new.pkl'

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

# delta_ps = np.zeros((636252, 1029)) #1470
# for i in range(len(train_dataset)):
#     gt_param = train_dataset[i][1].numpy()

#     p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
#     gt_vert = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
#     # gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
#     # gt_vert[2, :] -= gt_vert[2, :].min()

#     # img = cv2.imread(root + train_dataset.lines[i])
#     # render(img, [gt_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

#     # Ax = b
#     cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)[0]
#     delta_p = cp_estimated - control_points_ # 343 x 3
#     delta_p = delta_p.reshape(1, -1)
#     delta_ps[i] = delta_p

#     print(i, "/636252")

# with open("train.configs/delta_p_full_train.pkl", "wb") as f:
#     pickle.dump(delta_ps, f)

w_shp_ = _numpy_to_cuda(w_shp_)
w_exp_ = _numpy_to_cuda(w_exp_)
u_ = _numpy_to_cuda(u_)
deform_matrix = _numpy_to_cuda(deform_matrix).float()
control_points_ = _numpy_to_cuda(control_points_).float()
# normal function to run on cpu                           
delta_ps = np.zeros((636252, 1029)) #1470
for i in range(len(train_dataset)):
    start = time.time()
    gt_param = train_dataset[i][1].float().cuda()

    p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
    # gt_vert = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
    # gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
    gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).view(-1, 3).T + offset
    # gt_vert[2, :] -= gt_vert[2, :].min()

    # img = cv2.imread(root + train_dataset.lines[i])
    # render(img, [gt_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

    # Ax = b
    cp_estimated = torch.lstsq(gt_vert.T, deform_matrix)[0]
    delta_p = cp_estimated - control_points_ # 343 x 3
    delta_p = delta_p.reshape(1, -1)
    delta_ps[i] = delta_p

    if i > 0 and i % 10000 == 0:
        with open(f"train.configs/delta_p_full_train_{i}.pkl", "wb") as f:
            pickle.dump(delta_ps, f, protocol=4)
    print(i, "/636252 ", time.time() - start)

with open("train.configs/delta_p_full_train.pkl", "wb") as f:
    pickle.dump(delta_ps, f, protocol=4)
  

delta_ps = np.zeros((51602, 1029)) #1470
for i in range(len(val_dataset)):
    start = time.time()
    gt_param = val_dataset[i][1].numpy()

    p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
    gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset

    # img = cv2.imread(root + val_dataset.lines[i])
    # render(img, [gt_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

    # Ax = b
    cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)
    delta_p = cp_estimated - control_points_ # 343 x 3
    delta_p = delta_p.reshape(1, -1)
    delta_ps[i] = delta_p

    if i > 0 and i % 10000 == 0:
        with open(f"train.configs/delta_p_full_val_{i}.pkl", "wb") as f:
            pickle.dump(delta_ps, f, protocol=4)

    print(i, "/51602", time.time() - start)
    

with open("train.configs/delta_p_full_val.pkl", "wb") as f:
    pickle.dump(delta_ps, f, protocol=4)
