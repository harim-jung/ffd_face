# coding: utf-8

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from utils.io import _load, _numpy_to_cuda, _numpy_to_tensor

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class BFMModel_ONNX(nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, param_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()

        _to_tensor = _numpy_to_tensor

        # load bfm
        bfm = _load(bfm_fp)

        u = _to_tensor(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

        param = _load(param_fp)
        # param_mean and param_std used for re-whitening
        self.param_mean = _to_tensor(param.get('param_mean'))
        self.param_std = _to_tensor(param.get('param_std'))

    def forward(self, param):
        # R, offset, alpha_shp, alpha_exp, roi_box = param # 66-d
        param = param.view(-1)
        param_ = param[:62] * self.param_std + self.param_mean
        R_ = param_[:12].view(3, -1)
        R = R_[:, :3]
        offset = R_[:, -1].view(3, 1)
        alpha_shp = param_[12:52].view(-1, 1)
        alpha_exp = param_[52:62].view(-1, 1)
        alpha = torch.cat((alpha_shp, alpha_exp))

        pts3d = R @ (self.u + self.w.matmul(alpha).view(3, -1)) + offset

        # transform to image coordinate space
        pts3d = torch.cat((pts3d[0, :], torch.add(-pts3d[1, :], 121), pts3d[2, :]), dim=0).view(3, -1)

        # re-scaling
        roi_box = param[62:]
        sx, sy, ex, ey = roi_box[0], roi_box[1], roi_box[2], roi_box[3]
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        s = (scale_x + scale_y) / 2
        vert = torch.cat((torch.add(torch.mul(pts3d[0, :], scale_x), sx), torch.add(torch.mul(pts3d[1, :], scale_y), sy), torch.mul(pts3d[2, :], s)), dim=0).view(3, -1)
        vert = torch.transpose(vert, 0, 1)

        return vert

def convert_bfm_to_onnx(bfm_onnx_fp, param_fp, shape_dim=40, exp_dim=10):
    bfm_fp = bfm_onnx_fp.replace('.onnx', '.pkl')
    bfm_decoder = BFMModel_ONNX(bfm_fp=bfm_fp, param_fp=param_fp, shape_dim=shape_dim, exp_dim=exp_dim)
    bfm_decoder.eval()

    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 66)

    bfm_onnx_fp = 'train.configs/bfm_noneck_v3_transposed.onnx'
    torch.onnx._export(bfm_decoder, inputs, bfm_onnx_fp, export_params=True, verbose=True,
                       input_names=input_names, output_names=output_names, opset_version=10)

    print(f'Convert {bfm_fp} to {bfm_onnx_fp} done.')

if __name__ == '__main__':
    convert_bfm_to_onnx('train.configs/bfm_noneck_v3.onnx', 'train.configs/param_whitening.pkl')