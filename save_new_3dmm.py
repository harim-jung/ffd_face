import numpy as np
from bernstein_ffd.ffd_utils import *

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz, reconstruct_vertex, LpDataset, _parse_param
import cv2
from utils.render_simdr import render
from utils.io import _load
import time

filelists_train ='train.configs/train_aug_120x120.list.train'
filelists_val = 'train.configs/train_aug_120x120.list.val'
root ='../Datasets/train_aug_120x120/'
param_fp_train = 'train.configs/param_all.pkl'
param_fp_val ='train.configs/param_all_val.pkl'

filelists_train_f = 'train.configs/300w_lp_train_gt.txt'
filelists_val_f = 'train.configs/300w_lp_val_gt.txt'
root_f = '../Datasets/300W_LP/Data/'

normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

train_dataset_f = LpDataset(
    filelists_train_f,
    root_f,
    transform=transforms.Compose([ToTensorGjz()])#, normalize])
)

val_dataset_f = LpDataset(
    filelists_val_f,
    root_f,
    transform=transforms.Compose([ToTensorGjz()])#, normalize])
)

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

# temp
# index = np.where(train_dataset.lines == 'HELEN_HELEN_3036412907_2_0_1.jpg')
# index = np.where(val_dataset.lines == 'HELEN_HELEN_3036412907_2_0_1.jpg')



dic = {'LFPWFlip': 'LFPW_Flip', 'IBUGFlip': 'IBUG_Flip', 'HELENFlip': 'HELEN_Flip', 'AFWFlip': 'AFW_Flip', 
        'LFPW': 'LFPW', 'IBUG': 'IBUG', 'HELEN': 'HELEN', 'AFW': 'AFW'}

train_paths = [] 
val_paths = []

for img in train_dataset_f.imgs_path:
    train_paths.append("/".join(img.split("/")[-2:]))
for img in val_dataset_f.imgs_path:
    val_paths.append("/".join(img.split("/")[-2:]))

new_gts = np.zeros((636252, 240))
# for i in range(100):
for i in range(len(train_dataset)):
    start = time.time()
    clipped_param = train_dataset[i][1].numpy()
    img_fp = train_dataset.lines[i]
    img_full = dic[img_fp.split("_")[0]] + "/" + "_".join(img_fp.split("_")[1:-1]) + ".jpg"
    
    index = np.where(np.array(train_paths) == img_full)[0]
    if index.size == 1:
        index = index[0]
        full_param = train_dataset_f[index][1]
    else:
        index = np.where(np.array(val_paths) == img_full)[0][0]
        full_param = val_dataset_f[index][1]

    p, offset, alpha_shp, alpha_exp = _parse_param(clipped_param)
    p_f, offset_f, alpha_shp_f, alpha_exp_f = _parse_param(full_param.numpy())

    new_gt = np.concatenate((clipped_param[:12].reshape(-1,1), alpha_shp_f, alpha_exp_f)).reshape(1, -1)
    new_gts[i] = new_gt

    if i > 0 and i % 10000 == 0:
        with open(f"train.configs/param_all_new_{i}.pkl", "wb") as f:
            pickle.dump(new_gts, f)
    print(i, "/636252 ", time.time() - start)

with open("train.configs/param_all_new.pkl", "wb") as f:
    pickle.dump(new_gts, f)


new_gts = np.zeros((51602, 240))
for i in range(len(val_dataset)):
    clipped_param = val_dataset[i][1].numpy()
    img_fp = val_dataset.lines[i]
    img_full = dic[img_fp.split("_")[0]] + "/" + "_".join(img_fp.split("_")[1:-1]) + ".jpg"
    
    index = np.where(np.array(train_paths) == img_full)[0]
    if index.size == 1:
        index = index[0]
        full_param = train_dataset_f[index][1]
    else:
        index = np.where(np.array(val_paths) == img_full)[0][0]
        full_param = val_dataset_f[index][1]

    p, offset, alpha_shp, alpha_exp = _parse_param(clipped_param)
    p_f, offset_f, alpha_shp_f, alpha_exp_f = _parse_param(full_param.numpy())

    new_gt = np.concatenate((clipped_param[:12].reshape(-1,1), alpha_shp_f, alpha_exp_f)).reshape(1, -1)
    new_gts[i] = new_gt
    print(i, "/51602")

    # vertex = p @ (u_ + w_shp_lp @ alpha_shp + w_exp_lp @ alpha_exp).reshape(3, -1, order='F') + offset
    # vertex_full = p @ (u_ + w_shp_ @ alpha_shp_f + w_exp_ @ alpha_exp_f).reshape(3, -1, order='F') + offset

    # img = cv2.imread(root + train_dataset.lines[i])
    # render(img.copy(), [vertex], tri_, alpha=0.7, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
    # render(img.copy(), [vertex_full.astype(np.float32)], tri_, alpha=0.7, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

with open("train.configs/param_all_val_new.pkl", "wb") as f:
    pickle.dump(new_gts, f)

# delta_ps = []
# for i in range(100):
#     gt_param = train_dataset[i][1].numpy()

#     # render(img, [vertex_gt.astype(np.float32)], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

#     gt_vert = reconstruct_vertex(gt_param, dense=True, transform=False)
#     # Ax = b
#     cp_estimated = np.linalg.lstsq(deform_matrix_, gt_vert.T)
#     delta_p = cp_estimated - control_points_
#     delta_ps.append(delta_p)


