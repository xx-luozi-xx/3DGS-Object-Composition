# 基于物理驱动光传输分解的 3D Gaussian Splatting 实时物体合成

**面向 3D Gaussian Splatting 的物理真实感实时物体合成方法。**

[English](README.md) | [简体中文](README_zh-CN.md)

![示意图](https://github.com/xx-luozi-xx/3DGS-Object-Composition/blob/main/assets/teaser.png?raw=true)

## 项目简介

本项目支持在预训练的 3D Gaussian Splatting（3DGS）场景中进行**实时、无需重新训练的物体合成**。

给定一个预训练的 3DGS 场景以及已知的 PBR Gaussian 物体资产，PhysCompose 通过重新构建光传输过程，高效地处理场景光照、物体自遮挡以及物体对场景产生的阴影。该方法无需重新训练场景，也不依赖计算开销较高的基于体素的光线行进。

本框架在消费级硬件上可以达到**超过 40 FPS 的运行速度**，从而支持在非受限光照条件下进行交互式虚实物体合成。

## 安装

请在同一个终端会话中按顺序执行以下所有安装命令，执行过程中请不要中断。

在部署过程中，可能会暂时出现类似以下内容的 NumPy 版本兼容性警告：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash. To support both 1.x and 2.x versions of NumPy, modules must be compiled with NumPy 2.0. Some module may need to rebuild instead e.g. with 'pybind11>=2.12'. 

If you are a user of the module, the easiest solution will be to downgrade to 'numpy<2' or try to upgrade the affected module. We expect that some modules will need time to support NumPy 2.
```

该警告属于部署过程中的临时提示，可以忽略。最终环境会安装 `numpy==1.26.4`。部署成功完成后，通常不应再出现此类警告。

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

## 运行示例

本项目提供两个示例，分别展示如何将不透明物体和透明物体合成到真实场景中。

完整工作流程包括以下两个阶段：

1. 探针烘焙
2. 渲染

为简化示例流程，示例代码仅在物体插入位置烘焙单个探针。首次运行时，烘焙相关组件会进行即时编译，因此启动阶段可能需要额外等待一段时间。请耐心等待编译过程完成。

你可以通过[下载链接](https://drive.google.com/drive/folders/1L8GOXoYajccNMWdMFU1LGZZmgd8lcztl?usp=sharing)下载我们准备好的 Gaussian 模型，并将其放置在工作目录 `./model/` 下。

目录结构应如下所示：

```text
./model
├── armadillo_3.ply
├── bicycle_d_1.ply
├── garden_d_1.ply
└── hotdog.ply
```

### 在真实场景中插入不透明物体

运行以下命令：

```bash
python main_garden_video.py
```

### 在真实场景中插入透明物体

运行以下命令：

```bash
python main_bicycle_video.py
```


