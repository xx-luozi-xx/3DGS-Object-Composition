# Real-Time Object Composition in 3D Gaussian Splatting via Physics-Driven Light Transport Factorization

**Physically based, real-time object composition for 3D Gaussian Splatting.**

[English](README.md) | [简体中文](README_zh-CN.md)

![Teaser](https://github.com/xx-luozi-xx/3DGS-Object-Composition/blob/main/assets/teaser.png?raw=true)

### [[Paper]](https://raw.githubusercontent.com/xx-luozi-xx/3DGS-Object-Composition/main/assets/paper.pdf)

[Boyan Du](https://github.com/xx-luozi-xx), [Junke Zhu](https://github.com/junkzhu), [Haochuan Dang](), [Ang Li](https://github.com/Alan-sp), [Jixiang Hu](https://github.com/HuJX2005), [Sheng Li](https://lishengpku.github.io/), [Zhangjin Huang](http://staff.ustc.edu.cn/~zhuang/)

## Overview

This work enables **real-time, training-free object composition in pretrained 3D Gaussian Splatting (3DGS) scenes**.

Given a pretrained 3DGS scene and known PBR Gaussian assets, our light-transport reformulation efficiently accounts for scene illumination, object self-occlusion, and object-to-scene shadow casting—without retraining the scene or relying on computationally expensive voxel-based ray marching.

Our framework achieves **over 40 FPS on consumer hardware**, enabling interactive virtual-real composition under unconstrained lighting conditions.

## Installation

### Device and Software Environment

This project was tested in the following environment:
- Operating System: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 3090
- NVIDIA Driver Version: 580.95.05
- CUDA Version Supported by the Driver: 13.0
- CUDA Toolkit Version Installed for the Project: 12.1
- Python Environment: Conda, Python 3.10

It is recommended to use an NVIDIA discrete GPU and ensure that the VRAM meets the requirements for running the project. Although the current NVIDIA driver supports CUDA 13.0, the PyTorch, CUDA Toolkit, and related CUDA extensions used in this project are all installed based on CUDA 12.1.

### Note

Please execute all installation commands below in the same terminal session and do not interrupt the process.

During installation, you may temporarily encounter a NumPy version compatibility warning similar to the following:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash. To support both 1.x and 2.x versions of NumPy, modules must be compiled with NumPy 2.0. Some module may need to rebuild instead e.g. with 'pybind11>=2.12'. 

If you are a user of the module, the easiest solution will be to downgrade to 'numpy<2' or try to upgrade the affected module. We expect that some modules will need time to support NumPy 2.
```

This warning may appear temporarily during environment setup and can be safely ignored. The installation process will ultimately install `numpy==1.26.4`. After the deployment is completed successfully, warnings of this type should no longer appear.


### Deployment

```bash
git clone https://github.com/xx-luozi-xx/3DGS-Object-Composition.git

cd 3DGS-Object-Composition

conda create -n GSOC python=3.10 -y

conda activate GSOC

conda install -y \
    pytorch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    pytorch-cuda=12.1 \
    -c pytorch \
    -c nvidia \
    -c conda-forge \

conda install -y \
    -c nvidia/label/cuda-12.1.1 \
    -c defaults \
    cuda-nvcc=12.1.105 \

export CUDA_HOME="${CONDA_PREFIX}"

export PATH="${CUDA_HOME}/bin:${PATH}"

conda install -y \
    -c conda-forge \
    iopath=0.1.10 \
    fvcore=0.1.5.post20221221 \

python -m pip install \
    numpy==1.26.4 \
    gsplat==1.5.2 \
    opencv-contrib-python==4.9.0.80 \
    plyfile==1.1.3 \
    packaging==25.0 \
    OpenEXR==3.4.7 \
    Pillow==12.0.0 \
    scikit-learn==1.7.2

cd submodules/diff-gaussian-rasterization-panorama

conda install -y \
    -c conda-forge \
    setuptools=80.9.0 \

conda install -y \
    -c nvidia \
    cuda-nvcc=12.1 \
    cuda-toolkit=12.1 \

conda install -y --force-reinstall \
    -c nvidia/label/cuda-12.1.1 \
    cuda-cccl=12.1.109 \

python -m pip install . --no-build-isolation -v

cd ../..
```

## Running the Examples

This repository provides two demos illustrating the insertion of opaque and transparent objects into real-world scenes.

The complete workflow consists of two stages:

1. Probe baking
2. Rendering

For simplicity, the demo scripts bake only a single probe at the object insertion location. During the first execution, the baking components are compiled just in time, which may introduce additional startup time. Please wait until the compilation process has completed.

The prepared Gaussian models can be downloaded from the [download link](https://drive.google.com/drive/folders/1L8GOXoYajccNMWdMFU1LGZZmgd8lcztl?usp=sharing) and placed in the `./model/` directory.

The expected directory structure is:

```text
./model
├── armadillo_3.ply
├── bicycle_d_1.ply
├── garden_d_1.ply
└── hotdog.ply
```

### Opaque Object Composition

To insert an opaque object into a real-world scene, run:

```bash
python main_garden_video.py
```

### Transparent Object Composition

To insert a transparent object into a real-world scene, run:

```bash
python main_bicycle_video.py
```


