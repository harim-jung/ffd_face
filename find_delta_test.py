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
param_fp_train = 'train.configs/delta_p_full_train_norm.pkl'
param_fp_val ='train.configs/delta_p_full_val_norm.pkl'

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

def calc_mean_std(loader):
    # imp - calculate only for training set but apply to val and test data as well
    ndata = 0
    sum = 0.0
    for i_batch, batch_target in enumerate(loader):
        gt_param = batch_target[1].reshape(-1) # torch.Size([1, 240])
        p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
        # gt_vert = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
        gt_vert = p.numpy() @ (u_ + w_shp_ @ alpha_shp.numpy() + w_exp_ @ alpha_exp.numpy()).reshape(3, -1, order='F') + offset.numpy()
        # ndata += gt_param.size(0)
        sum += gt_vert.reshape(-1, 1)#.sum(0)
        print(i_batch)

    mean = sum / len(train_dataset)
    print(mean)

    # diff = 0.0
    # for i_batch, batch_target in enumerate(loader):
    #     batch = batch_target[1] # torch.Size([10, 235])
    #     diff += ((batch - mean)**2).sum(0)
    #     print(i_batch)

    # var = diff / nimages
    # std = torch.sqrt(var)
    # print(std)

    # dic = {}
    # dic["mean_mesh"] = np.array(mean).reshape(-1)
    f = open("train.configs/mesh_mean.pkl", "wb")
    pickle.dump(mean, f)
    print("saved mean mesh")

# loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
# calc_mean_std(loader)
dic = _load('train.configs/delta_p_mean_std.pkl')
delta_mean = dic["delta_p_mean"]
delta_std = dic["delta_p_std"]
# delta_ps = np.zeros((636252, 1029)) #1470
for i in range(len(train_dataset)):
    gt_param = train_dataset[i][1].numpy()
    gt_param = gt_param * delta_std + delta_mean
    # p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
    # gt_vert = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
    # gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
    # gt_vert[2, :] -= gt_vert[2, :].min()

    gt_vert = deformed_vert(gt_param, transform=False)

    img = cv2.imread(root + train_dataset.lines[i])
    render(img, [gt_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

    # Ax = b
    # cp_estimated = np.linalg.lstsq(deform_matrix_, gt_vert.T)[0]
    # delta_p = cp_estimated - control_points_ # 343 x 3
    # delta_p = delta_p.reshape(1, -1)
    # delta_ps[i] = delta_p

    print(i, "/636252")

# with open("train.configs/delta_p_full_train.pkl", "wb") as f:
#     pickle.dump(delta_ps, f)


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
#     cp_estimated = np.linalg.lstsq(deform_matrix_, gt_vert.T)[0]
#     delta_p = cp_estimated - control_points_ # 343 x 3
#     delta_p = delta_p.reshape(1, -1)
#     delta_ps[i] = delta_p

#     print(i, "/636252")

# with open("train.configs/delta_p_full_train.pkl", "wb") as f:
#     pickle.dump(delta_ps, f)

def svd_solve(a, b):
    [U, s, Vt] = np.linalg.svd(a, full_matrices=True)
    r = max(np.where(s >= 1e-6)[0])
    # Sigma = np.diag(s[:r], np.zeros((s[r:].shape)))
    temp = np.dot(U[:, :r].T, b) / s[:r]
    return np.dot(Vt[:r, :].T, temp)


# delta_ps = np.zeros((636252, 1029)) #1470
# for i in range(0, len(train_dataset), 1000):
#     start = time.time()
#     i = 4186
#     gt_param = train_dataset[i][1].numpy()

#     p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
#     # gt_vert = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
#     gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset
#     # gt_vert[0] -= gt_vert[0].min()
#     # gt_vert[1] -= gt_vert[1].min()
#     # gt_vert[2] -= gt_vert[2].min()

#     # test
#     # cp_estimated = np.linalg.lstsq(deform_matrix, gt_vert.T)[0]
#     # v = BP # Ax = b
#     # v = B * (P + deltaP)
#     # v = BP + B*deltaP
#     # BdeltaP = v - BP
#     # 
#     # delta_p = np.linalg.lstsq(deform_matrix, (gt_vert.T - deform_matrix @ control_points))[0]
#     delta_p_x = svd_solve(deform_matrix, (gt_vert[0].T - deform_matrix @ control_points[:, 0]))
#     delta_p_y = svd_solve(deform_matrix, (gt_vert[1].T - deform_matrix @ control_points[:, 1]))
#     delta_p_z = svd_solve(deform_matrix, (gt_vert[2].T - deform_matrix @ control_points[:, 2]))
#     print("")
#     # Ax = b
#     # cp_estimated = np.linalg.lstsq(deform_matrix_, gt_vert.T)[0]
#     # delta_p = cp_estimated - control_points_ # 343 x 3
#     # delta_p = delta_p.reshape(1, -1)
#     # delta_ps[i] = delta_p

#     # if i > 0 and i % 10000 == 0:
#     #     with open(f"train.configs/delta_p_full_train_{i}.pkl", "wb") as f:
#     #         pickle.dump(delta_ps, f, protocol=4)

#     # print(i, "/636252 ", time.time() - start)

# # with open("train.configs/delta_p_full_train.pkl", "wb") as f:
# #     pickle.dump(delta_ps, f, protocol=4)
  

# delta_ps = np.zeros((51602, 1029)) #1470
# for i in range(len(val_dataset)):
#     start = time.time()
#     gt_param = val_dataset[i][1].numpy()

#     p, offset, alpha_shp, alpha_exp = _parse_param(gt_param)
#     gt_vert = p @ (u_ + w_shp_ @ alpha_shp + w_exp_ @ alpha_exp).reshape(3, -1, order='F') + offset

#     # img = cv2.imread(root + val_dataset.lines[i])
#     # render(img, [gt_vert.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

#     # Ax = b
#     cp_estimated = np.linalg.lstsq(deform_matrix_, gt_vert.T)
#     delta_p = cp_estimated - control_points_ # 343 x 3
#     delta_p = delta_p.reshape(1, -1)
#     delta_ps[i] = delta_p

#     if i > 0 and i % 10000 == 0:
#         with open(f"train.configs/delta_p_full_val_{i}.pkl", "wb") as f:
#             pickle.dump(delta_ps, f, protocol=4)

#     print(i, "/51602", time.time() - start)
    

# with open("train.configs/delta_p_full_val.pkl", "wb") as f:
#     pickle.dump(delta_ps, f, protocol=4)
