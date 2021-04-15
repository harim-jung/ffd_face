# fit 3dmm parameters from Coarse Data

from utils.params import *
from utils.ddfa import reconstruct_vertex#, _parse_param
from utils.inference import dump_rendered_img, dump_to_ply, get_suffix, parse_roi_box_from_landmark
from utils.render_simdr import render
import numpy as np
import bernstein_ffd.ffd_utils
from os import walk
import cv2
import random
import pickle, torch
from plyfile import PlyData, PlyElement

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.io import _load

def create_file_list(root):
    files = open("train.configs/coarse_data/coarse_data_list.txt", "w")
    for (dirpath, dirnames, filenames) in walk(root):
        for file in filenames:
            if file.endswith(".jpg"):
                print(file)
                files.write(os.path.join(dirpath.split("/")[-1], file) + "\n")       
    
    files.close()


def create_train_val():
    lp_train = open("train.configs/coarse_data/300w_lp_train.txt", "r")
    lp_val = open("train.configs/coarse_data/300w_lp_val.txt", "r")

    train_file = open("train.configs/coarse_data/coarse_data_train_list.txt", "w")
    val_file = open("train.configs/coarse_data/coarse_data_val_list.txt", "w")
    file_txt = open("train.configs/coarse_data/coarse_data_list.txt", "r")
    file_list = file_txt.read().split("\n")

    train_data = []
    val_data = []
    train_num = 0.9
    
    from itertools import groupby

    # def parent_fp(img):
    #     # return img.split("/")[0]
    #     return os.path.split(img)[0]

    def image_fp(img):
        return os.path.split(img)[0]

    dirs = []
    for (dirpath, dirnames, filenames) in walk('D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData/'):
        for d in dirnames:
            dirs.append(d)

    dirs = sorted(list(set(dirs)))

    lp_train_ls = lp_train.read().split("\n")
    lp_val_ls = lp_val.read().split("\n")

    train_id = []
    for i, img in enumerate(lp_train_ls):
        lp = "_".join(os.path.split(img)[1].split("_")[:-1]).lower()
        train_id.append(lp)
        print(i)
    
    train_lp = sorted(list(set(train_id)))
    
    val_id = []
    for i, img in enumerate(lp_val_ls):
        lp = "_".join(os.path.split(img)[1].split("_")[:-1]).lower()
        val_id.append(lp)
        print(i)
    
    val_lp = sorted(list(set(val_id)))


    train_dir = []
    val_dir = []

    for d in dirs:
        if d in train_lp:
            train_dir.append(d)
        elif d in val_lp:
            val_dir.append(d)

    print(len(train_dir) + len(val_dir))


    root = 'D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData/'
    for d in train_dir:
        for (dirpath, dirnames, filenames) in walk(os.path.join(root, d)):
            for file in filenames:
                if file.endswith("jpg"):
                    train_file.write(os.path.join(d, file) + "\n")
    for d in val_dir:
        for (dirpath, dirnames, filenames) in walk(os.path.join(root, d)):
            for file in filenames:
                if file.endswith("jpg"):
                    val_file.write(os.path.join(d, file) + "\n")

    train_file.close()
    val_file.close()
    file_txt.close()
    lp_train.close()
    lp_val.close()


def parse_params(params):
    alpha_shp = params[:100].reshape((-1, 1))
    alpha_exp = params[100:179].reshape((-1, 1))
    pose = params[179:] #  pitch, yaw, roll, t2dx, t2dy, f
    rot = pose[:3]
    offset = np.append(pose[3:5]).reshape(-1, 1)
    scale = pose[5]

    return scale, offset, alpha_shp, alpha_exp


def crop_img(img, roi_box, param):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]

    x_shift = sx
    y_shift = img.shape[0] - (sy + res.shape[0])
    t2d = np.array([x_shift, y_shift])

    param_ = param.copy()
    param_[182:184] -= t2d

    return res, param_


def resize_and_scale(cropped_img, param):
    resized_img = cv2.resize(cropped_img, (std_size, std_size))
    ratio = resized_img.shape[0] / cropped_img.shape[0]

    param_ = param.copy()
    param_[182:185] = param_[182:185] * ratio  # t2dx, t2dy, scale

    return resized_img, param_


def crop_imgs(img_fp, param, root='D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData/', save_root='D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData_cropped/'):
    img = cv2.imread(os.path.join(root, img_fp))
    # fit landmarks
    lms = reconstruct_vertex(param, dense=False, transform=True, std_size=450).astype(np.float32) 
    # get roi box
    roi_box = parse_roi_box_from_landmark(lms)
    # crop_img
    cropped, param_= crop_img(img, roi_box, param)
    # resize img
    resized, param_= resize_and_scale(cropped, param_)

    # vertex = reconstruct_vertex(param_, dense=True, transform=False, std_size=450).astype(np.float32) # (3, 53215)
    
    path = os.path.join(save_root, os.path.split(img_fp)[0])
    if not os.path.isdir(path):
        os.mkdir(path)

    wfp = os.path.join(path, os.path.split(img_fp)[-1])
    # wfp = None
    
    cv2.imwrite(wfp, resized)
    print("saved", wfp)
    # render(resized, [vertex], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)

    return param_


def save_params():
    root = 'D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData/'

    train_file = open("train.configs/coarse_data/coarse_data_train_list.txt", "r")
    val_file = open("train.configs/coarse_data/coarse_data_val_list.txt", "r")
    train_param = open("train.configs/coarse_data/coarse_data_train_params.pkl", "wb")
    val_param = open("train.configs/coarse_data/coarse_data_val_params.pkl", "wb")

    train_list = train_file.read().split("\n")
    val_list = val_file.read().split("\n")

    params = []
    for i, file in enumerate(train_list):
        param_txt = file.replace(get_suffix(file), ".txt")
        with open(root + param_txt, "r") as f:
            param = np.array(f.read().replace("\n", "").rstrip().split(" ")).astype(np.float32)
        
        print(i)
        param_ = crop_imgs(file, param)
        params.append(param_)

    pickle.dump(np.array(params), train_param)

    params = []
    for i, file in enumerate(val_list):
        param_txt = file.replace(get_suffix(file), ".txt")
        with open(root + param_txt, "r") as f:
            param = np.array(f.read().replace("\n", "").rstrip().split(" ")).astype(np.float32)
        
        print(i)
        param_ = crop_imgs(file, param)
        params.append(param_)

    pickle.dump(np.array(params), val_param)

    train_file.close()
    val_file.close()
    train_param.close()
    val_param.close()


def calc_param_mean_std(loader):
    # imp - calculate only for training set but apply to val and test data as well
    nimages = 0
    sum = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[1] # torch.Size([10, 235])
        nimages += batch.size(0)
        sum += batch.sum(0)
        print(i_batch)

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
    dic["param_mean"] = np.array(mean).reshape(-1)
    dic["param_std"] = np.array(std).reshape(-1)
    f = open("train.configs/param_all_full_mean_std.pkl", "wb")
    pickle.dump(dic, f)
    print("saved coarse_data_img_mean_std")


def calc_img_mean_std(loader):
    # imp - calculate only for training set but apply to val and test data as well
    nimages = 0
    sum = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0] # images: torch.Size([10, 3, 128, 128])
        # batch = batch_target # images: torch.Size([10, 3, 128, 128])
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1) # torch.Size([10, 3, 16384])
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        sum += batch.mean(2).sum(0)
        print(i_batch)

    mean = sum / nimages
    print("mean:", mean)

    diff = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0] # images: torch.Size([10, 3, 128, 128])
        # batch = batch_target # images: torch.Size([10, 3, 128, 128])
        batch = batch.view(batch.size(0), batch.size(1), -1)  # torch.Size([10, 3, 16384])
        diff += ((batch.mean(2) - mean)**2).sum(0)
        print(i_batch)

    var = diff / nimages
    std = torch.sqrt(var)
    print("mean:", mean, " std:", std)

    # dic = {}
    # dic["img_mean"] = np.array(mean).reshape(-1)
    # dic["img_std"] = np.array(std).reshape(-1)
    # f = open("train.configs/coarse_data/coarse_data_img_mean_std.pkl", "wb")
    # pickle.dump(dic, f)
    # print("saved coarse_data_img_mean_std")
    # f.close()


def rewhiten_lp_data():
    param_train = _load("train.configs/param_all_norm.pkl")
    param_train = param_train * param_std + param_mean
    f = open("train.configs/param_all.pkl", "wb")
    pickle.dump(param_train, f)
    f.close()

    param_val = _load("train.configs/param_all_norm_val.pkl")
    param_val = param_val * param_std + param_mean
    f = open("train.configs/param_all_val.pkl", "wb")
    pickle.dump(param_val, f)
    f.close()


def save_normalized_params():
    f = _load('train.configs/param_all_full.pkl')
    mean_std = _load('train.configs/param_all_full_mean_std.pkl')
    normalized = (f - mean_std['param_mean']) / mean_std['param_std']
    with open("train.configs/param_all_full_norm.pkl", "wb") as new:
        pickle.dump(normalized, new)
    
    f = _load('train.configs/param_all_val_full.pkl')
    normalized = (f - mean_std['param_mean']) / mean_std['param_std']
    with open("train.configs/param_all_val_full_norm.pkl", "wb") as new:
        pickle.dump(normalized, new)


if __name__== "__main__":
    # filelists_train ='train.configs/train_aug_120x120.list.train'
    # filelists_val = 'train.configs/train_aug_120x120.list.val'
    # root ='../Datasets/train_aug_120x120'
    # param_fp_train = 'train.configs/param_all_norm.pkl'
    # param_fp_val ='train.configs/param_all_norm_val.pkl'

    # normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    # train_dataset = DDFADataset(
    #     root=root,
    #     filelists=filelists_train,
    #     param_fp=param_fp_train,
    #     transform=transforms.Compose([ToTensorGjz(), normalize])
    # )
    # val_dataset = DDFADataset(
    #     root=root,
    #     filelists=filelists_val,
    #     param_fp=param_fp_val,
    #     transform=transforms.Compose([ToTensorGjz(), normalize])
    # )

    # z_coord = []
    # for i in range(100):
    #     gt_param = train_dataset[i][1].numpy()
    #     gt_vert = reconstruct_vertex(gt_param, dense=True, transform=False)
    #     z_coord.append(gt_vert[2].min())
    #     print(gt_vert[2].min(), gt_vert[2].max())


    plydata = PlyData.read('train.configs/reference_mesh_lp_new.ply')
    v = plydata['vertex']

    vert = np.zeros((3, 53215))
    for i, vt in enumerate(v):
        vert[:, i] = np.array(list(vt))

    vert = vert[:, sampling_indices]

    vert_ = vert * 0.4 #0.27
    vert_[0] -= vert_[0].min()
    vert_[0] += (std_size - vert_[0].max()) / 2
    vert_[1] -= vert_[1].min()
    vert_[1] += (std_size - vert_[1].max()) / 2
    vert_[2] -= vert_[2].min()
    
    dump_to_ply(vert_, tri_.T, f"train.configs/reference_mesh_lp_new_.ply", transform=False)


    # rewhiten_lp_data()

    # create_train_val()

    # save_params()

    # train_files = "train.configs/train_aug_120x120.list.train"
    # train_param = "train.configs/param_all_full.pkl"
    # root = '../Datasets/train_aug_120x120/'

    # train_dataset = DDFADataset(
    #     root=root,
    #     filelists=train_files,
    #     param_fp=train_param,
    #     transform=transforms.Compose([ToTensorGjz()])#, normalize])
    # )

    # loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

    # # calc_img_mean_std(loader)
    # calc_param_mean_std(loader)


    save_normalized_params()


    # jpg = 'afw_1295311477_1/afw_1295311477_1_aug_25.jpg'
    # txt = 'afw_1295311477_1/afw_1295311477_1_aug_25.txt'
    # root = 'D:/ai/3d face reconstruction/Datasets/Coarse_Dataset/CoarseData/'

    # with open(root + txt, "r") as f:
    #     params = np.array(f.read().rstrip().split(" ")).astype(np.float32)

    # crop_imgs(jpg, params)

    # txt_root = '../Datasets/Coarse_Dataset/CoarseData/'
    # img_root = '../Datasets/Coarse_Dataset/CoarseData_cropped/'

    # with open("train.configs/coarse_data/coarse_data_train_list.txt", "r") as f:
    #     jpgs = f.read().split("\n")
    
    # params = _load('train.configs/coarse_data/coarse_data_train_params.pkl')

    # for i, jpg in enumerate(jpgs[100:200]):
    #     # txt = jpg.replace(get_suffix(jpg), ".txt")

    #     # # jpg = 'afw_1295311477_1/afw_1295311477_1_aug_25.jpg'
    #     # # txt = 'afw_1295311477_1/afw_1295311477_1_aug_25.txt'

    #     # with open(txt_root + txt, "r") as f:
    #     #     params = np.array(f.read().rstrip().split(" ")).astype(np.float32)
    #     param = params[i+100]
    #     img = cv2.imread(img_root + jpg)
    #     # img_ = cv2.resize(img, (std_size, std_size))
    #     # cv2.imwrite('samples/inputs/afw_1295311477_1_aug_25.jpg', img_)

    #     # ratio = std_size / 450
    #     # params[182:185] = params[182:185] * ratio

    #     vertex = reconstruct_vertex(param, dense=True, transform=False, std_size=std_size).astype(np.float32) # (3, 53215)
        
    #     wfp = f"samples/outputs/{jpg.split('/')[-1]}"
    #     # dump_rendered_img(vertex, root + jpg, wfp=None, show_flag=True, alpha=0.8, face=True)
    #     render(img, [vertex], tri_, alpha=0.8, show_flag=True, wfp=None, with_bg_flag=True, transform=True)
        
        # dump_to_ply(vertex, tri_.T, f"samples/outputs/{jpg.split('/')[-1][:-4]}.ply", transform=True)

    # todo 
    # match identity with 300w-lp-aug training & validation set and properly  ivide
    # save file list and gt data list 
    # fix gt params according to resizing to 120x120 (scale factor & translation)
    # make a dataloader for coarse dataset
