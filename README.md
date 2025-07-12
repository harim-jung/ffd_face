### Official implementation of:

**Harim Jung, Myeong-Seok Oh, Seong-Whan Lee, “Learning Free-Form Deformation for 3D Face Reconstruction from In-The-Wild Images,” SMC 2021.** [[Paper (arXiv)](https://arxiv.org/pdf/2206.08509)]

This project proposes a learning-based method to reconstruct 3D face meshes from 2D images using **Free-Form Deformation** (FFD), overcoming the representational limitations of traditional PCA-based 3D Morphable Models (3DMM). This method estimates **deviations of control points** within a grid, offering both high reconstruction quality and practical editability.

## Install
```
# Create virtual env
conda create -n ffd_face python=3.8 -y
conda activate ffd_face

# Install (change according to your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run code
```
# train
python train_ffd.py --devices-id 0 --filelists-train train.configs/train_aug_120x120.list.train --filelists-val train.configs/train_aug_120x120.list.val --root ../Datasets/train_aug_120x120 --param-fp-train train.configs/param_lm_train.pkl --param-fp-val train.configs/param_lm_val.pkl --log-file training/logs/ffd_train.log --epochs 50 --batch-size 128 --arch mobilenet_1
```
