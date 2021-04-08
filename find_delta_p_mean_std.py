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
param_fp_train = 'train.configs/delta_p_pose_train.pkl'
param_fp_val ='train.configs/delta_p_pose_val.pkl'

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
    nimages = 0
    sum = 0.0
    for i_batch, batch_target in enumerate(loader):
        start = time.time()
        batch = batch_target[1] # torch.Size([10, 235])
        nimages += batch.size(0)
        sum += batch.sum(0)
        print(i_batch, start - time.time())

    mean = sum / nimages
    print(mean)

    diff = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[1] # torch.Size([10, 235])
        diff += ((batch - mean)**2).sum(0)
        print(i_batch)

    var = diff / nimages
    std = torch.sqrt(var)
    print(std)

    dic = {}
    dic["delta_p_mean"] = np.array(mean).reshape(-1)
    dic["delta_p_std"] = np.array(std).reshape(-1)
    f = open("train.configs/delta_p_pose_mean_std.pkl", "wb")
    pickle.dump(dic, f)
    print("saved delta_p_mean_std")

    # # imp - calculate only for training set but apply to val and test data as well
    # ndata = 0
    # sum = 0.0
    # for i_batch, batch_target in enumerate(loader):
    #     delta_p = batch_target[1].reshape(-1) # torch.Size([1, 240])
    #     sum += delta_p
    #     print(i_batch)

    # mean = sum / len(train_dataset)
    # print(mean)

    # # diff = 0.0
    # # for i_batch, batch_target in enumerate(loader):
    # #     batch = batch_target[1] # torch.Size([10, 235])
    # #     diff += ((batch - mean)**2).sum(0)
    # #     print(i_batch)

    # # var = diff / nimages
    # # std = torch.sqrt(var)
    # # print(std)

    # # dic = {}
    # # dic["mean_mesh"] = np.array(mean).reshape(-1)
    # f = open("train.configs/delta_p_mean_std.pkl", "wb")
    # pickle.dump(mean, f)
    # print("saved mean mesh")

loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
calc_mean_std(loader)

def save_normalized_delta_p():
    f = _load('train.configs/delta_p_full_train.pkl')
    mean_std = _load('train.configs/delta_p_mean_std.pkl')
    normalized = (f - mean_std['delta_p_mean']) / mean_std['delta_p_std']
    with open("train.configs/delta_p_full_train_norm.pkl", "wb") as new:
        pickle.dump(normalized, new)
    
    f = _load('train.configs/delta_p_full_val.pkl')
    normalized = (f - mean_std['delta_p_mean']) / mean_std['delta_p_std']
    with open("train.configs/delta_p_full_val_norm.pkl", "wb") as new:
        pickle.dump(normalized, new)

# save_normalized_delta_p()