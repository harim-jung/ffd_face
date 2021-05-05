import argparse
import torch
import models.mobilenet_v1
from utils.ddfa import load_model

arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']

parser = argparse.ArgumentParser(description='onnx')
parser.add_argument('--checkpoint', default='snapshot/phase2_wpdc_lm_vdc_all_checkpoint_epoch_19.pth.tar', type=str, metavar='PATH')
parser.add_argument('--arch', default='mobilenet_1', type=str, choices=arch_choices)
parser.add_argument('--num-classes', default=62, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = getattr(mobilenet_v1, args.arch)(num_classes=args.num_classes)
    model = load_model(model, args.checkpoint, True)
    model.eval()
    device = torch.device("cpu")
    # ------------------------ export -----------------------------
    output_onnx = args.checkpoint.split("/")[-1].replace('.pth.tar', '.onnx')
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, 120, 120).to(device)

    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=True,
                                   input_names=input_names, output_names=output_names, opset_version=10)

