#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

from utils.io import _numpy_to_cuda, _tensor_to_cuda


_to_tensor = _numpy_to_cuda

class CosineFeatLoss(nn.Module):
    def __init__(self):
        super(CosineFeatLoss, self).__init__()

    def forward(self, input, target, z_shift=False):
        # feature vector extracted from rendered images
        loss = nn.CosineSimilarity()

        N = target.shape[0]

        cos_loss = loss(input, target)
        return cos_loss


if __name__ == '__main__':
    pass
