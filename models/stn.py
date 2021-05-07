# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from utils.ddfa import get_axis_angle_from_rot_mat, get_rot_mat_from_axis_angle_batch
from bernstein_ffd.ffd_utils import sampled_uv_map, cp_num
import os

# import sys
# sys.path.append('..')
# from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz

# from ..utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz

class stnFFD(nn.Module):
    def __init__(self, channel=3, param_classes=cp_num):
        super(stnFFD, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for scale, axis angle, offset
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 26 * 26, 32),
            nn.ReLU(True),
            nn.Linear(32, 7)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # s, axis_angle, offset
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float))

        # resnet50 for delta P regression
        resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=param_classes)
        self.resnet50 = torch.nn.Sequential(*(list(resnet50.children())))

    def pixel_coord(self, uv, target_size):
        u, v = uv
        # pixel_x = u * 2 - 1
        # pixel_y = v * 2 - 1
        pixel_coord = torch.tensor([u * target_size, v * target_size])
        pixel_coord = torch.trunc(pixel_coord).int()
        
        return pixel_coord

    def convert_to_normal_square(self, xy_source, img_size):
        # height and width normalized coordinates
        # xs, ys = torch.abs(xy_source[:, 0]), torch.abs(xy_source[:, 1])
        xs, ys = xy_source[:, 0], xy_source[:, 1]
        xsNorm = (xs / img_size * 2) - 1
        ysNorm = -((ys / img_size * 2) - 1) # change (-1,-1) to be at left top 

        return xsNorm, ysNorm

    def affine_grid(self, T_theta, input_size):
        """
        # Generates a 2D flow field (sampling grid), given a batch of affine matrices theta.
        # grid specifies the sampling pixel locations normalized by the input spatial dimensions
        # the size-2 vector grid[n, h, w] specifies input pixel locations x and y (size N×H×W×2)
        for (u,v) in uvMapForRefMesh.uv:
            (x,y,z) = uvMapForRefMesh.xyz(u,v)
            (xs ys) = Convert_to_normal_square( T_cam( x, y,z) ) # convert_to_normal_square() 는 (-1, -1) 에서 (1,1)까지 변화게 좌표변환.
            grid[(u,v)] = (xs, ys)
        """
        target_size = len(sampled_uv_map)
        grid = torch.zeros(input_size[0], target_size, target_size, 2).cuda() # torch.Size([5, 8928, 8928, 2])
        img_size = input_size[2]
        pixel_coords = []
        for u, v in sampled_uv_map:
            x, y, z = sampled_uv_map[(u, v)]
            xy_s = (T_theta.double() @ torch.tensor([x,y,z,1]).cuda())[:, :2]
            xsNorm, ysNorm = self.convert_to_normal_square(xy_s, img_size) # [-1, 1]^2
            
            pixel_coord = self.pixel_coord((u,v), target_size)
            pixel_x, pixel_y = pixel_coord[0].item(), pixel_coord[1].item()
            pixel_coords.append((pixel_x, pixel_y))
            grid[:, pixel_x, pixel_y] = torch.cat((xsNorm, ysNorm)).view(2, -1).T

        return grid

    # Spatial transformer network forward function
    def stn(self, x):
        # x = torch.Size([N, 3, 120, 120]) # image input
        xs = self.localization(x) # torch.Size([N, 10, 26, 26])
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3]) # torch.Size([N, 6760])
        theta = self.fc_loc(xs) # torch.Size([N, 7])
        N = x.shape[0]
        s = torch.abs(theta[:, 0]).view(N, 1)
        axis_angle = theta[:, 1:4]
        offset = theta[:, 4:].view(N, 3, 1)
        p = get_rot_mat_from_axis_angle_batch(axis_angle) # N x 3 x 3

        rot_mat = torch.einsum('ab,acd->acd', s, p)
        T_theta = torch.cat((rot_mat, offset), 2) # Nx3x4 affine transformation matrix

        # target output image size same as input size
        # outputs flow-field grid of size N×H×W×2
        grid = self.affine_grid(T_theta, x.size()) # torch.Size([N, 120, 120, 2])
        
        # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
        # For each output location output[n, :, h, w], the size-2 vector grid[n, h, w] specifies input pixel locations x and y, 
        # normalized by the input spatial dimensions, which are used to interpolate the output value output[n, :, h, w]. 
        # It should have most values in the range of [-1, 1]. 
        # Values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
        x = F.grid_sample(x, grid, padding_mode='zeros') # torch.Size([64, 1, 120, 120])

        return T_theta, x

    def forward(self, x):
        # transform the input
        T_theta, aligned_x = self.stn(x) # torch.Size([5, 3, 8928, 8928])

        # Perform the usual forward pass
        delta_P = self.resnet50(aligned_x)

        return torch.cat((T_theta.view(-1, 12, 1), delta_P), dim=1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure the STN performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f'Save checkpoint to {filename}')


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')



if __name__=="__main__":
    # plt.ion()   # interactive mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test dataset
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    val_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    model = Net().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    checkpoint = torch.load('models/mnist_stn_checkpoint_epoch_20.pth.tar', map_location=lambda storage, loc: storage)['state_dict']
    model.load_state_dict(checkpoint)

    for input, target in val_loader:
        print(input)
        print(target)

    # Visualize the STN transformation on some input batch
    # visualize_stn()

    # plt.ioff()
    # plt.show()