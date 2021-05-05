from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from models.stn import stnFFD
from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # plt.ion()   # interactive mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    # Test dataset
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    root ='../Datasets/train_aug_120x120/'
    filelists_val = 'train.configs/train_aug_120x120_test.list.val'
    param_fp_val ='train.configs/param_all_val_full_test.pkl'

    val_dataset = DDFADataset(
        root=root,
        filelists=filelists_val,
        param_fp=param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4,
                            shuffle=False, pin_memory=True)

    model = stnFFD().to(device)

    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # checkpoint = torch.load('models/mnist_stn_checkpoint_epoch_20.pth.tar', map_location=lambda storage, loc: storage)['state_dict']
    # model.load_state_dict(checkpoint)

    for input, target in val_loader:
        print(input)
        print(target)
        output = model(input.to(device))

    # Visualize the STN transformation on some input batch
    # visualize_stn()

    # plt.ioff()
    # plt.show()