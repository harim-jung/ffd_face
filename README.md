### Official implementation of:

**Harim Jung, Myeong-Seok Oh, Seong-Whan Lee, “Learning Free-Form Deformation for 3D Face Reconstruction from In-The-Wild Images,” SMC 2021.** [[Paper (arXiv)](https://arxiv.org/pdf/2206.08509)]

The 3D Morphable Model (3DMM), which is a Principal Component Analysis (PCA) based statistical model that represents a 3D face using linear basis functions, has shown promising results for reconstructing 3D faces from single-view in-the-wild images. However, 3DMM has restricted representation power due to the limited number of 3D scans and global linear basis. To address the limitations of 3DMM, we propose a straightforward learning-based method that reconstructs a 3D face mesh through Free-Form Deformation (FFD) for the first time. FFD is a geometric modeling method that embeds a reference mesh within a parallelepiped grid and deforms the mesh by moving the sparse control points of the grid. As FFD is based on mathematically defined basis functions, it has no limitation in representation power. Thus, we can recover accurate 3D face meshes by estimating the appropriate deviation of control points as deformation parameters. Although both 3DMM and FFD are parametric models, deformation parameters of FFD are easier to interpret in terms of their effect on the final shape. This practical advantage of FFD allows the resulting mesh and control points to serve as a good starting point for 3D face modeling, in that ordinary users can fine-tune the mesh by using widely available 3D software tools. Experiments on multiple datasets demonstrate how our method successfully estimates the 3D face geometry and facial expressions from 2D face images, achieving comparable performance to the state-of-the-art methods.

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
