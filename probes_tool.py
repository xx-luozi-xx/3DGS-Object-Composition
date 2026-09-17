import os
import json

import cv2
import numpy as np
import torch
import OpenEXR
import Imath

def save_exr(tensor: torch.Tensor, path: str):
    """
    保存 (H, W, 3) 的 float32 torch.Tensor 为 EXR 文件
    使用官方 OpenEXR 库，无损保存 32-bit Float 能量
    """
    arr = tensor.detach().cpu().float().numpy()
    h, w = arr.shape[:2]

    # 1. 强制拆分 RGB 通道，并转换为底层 C++ 认识的 bytes 字节流
    R = arr[:, :, 0].tobytes()
    G = arr[:, :, 1].tobytes()
    B = arr[:, :, 2].tobytes()

    # 2. 构建 EXR 文件头，严格指定宽高和 FLOAT(32-bit) 精度
    header = OpenEXR.Header(w, h)
    FLOAT = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': FLOAT, 'G': FLOAT, 'B': FLOAT}

    # 3. 写入文件
    out_file = OpenEXR.OutputFile(path, header)
    out_file.writePixels({'R': R, 'G': G, 'B': B})
    out_file.close()

def load_hdr(path: str, device="cpu") -> torch.Tensor:
    """
    读取 HDR 文件，返回 (H, W, 3) 的 float32 torch.Tensor
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 HDR 文件: {path}")

    # 1. 使用 OpenCV 读取，IMREAD_UNCHANGED 确保保留 HDR 的 32 位浮点数特性
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取 HDR 文件，请检查文件是否损坏: {path}")

    # 2. OpenCV 默认加载为 BGR 通道顺序，需要转换回 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. 转换为 PyTorch Tensor，并移动到目标设备
    # OpenCV 读进来的 HDR 默认已经是 numpy 的 float32 格式
    return torch.from_numpy(img).to(device)

def load_exr(path: str, device="cpu") -> torch.Tensor:
    """
    读取 EXR 文件，返回 (H, W, 3) 的 float32 torch.Tensor
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 EXR 文件: {path}")

    # 1. 读取文件头，获取原图宽高
    in_file = OpenEXR.InputFile(path)
    header = in_file.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # 2. 分别提取 RGB 通道的字节流，并用 numpy 还原为 float32 矩阵
    R = np.frombuffer(in_file.channel('R', FLOAT), dtype=np.float32).reshape(h, w)
    G = np.frombuffer(in_file.channel('G', FLOAT), dtype=np.float32).reshape(h, w)
    B = np.frombuffer(in_file.channel('B', FLOAT), dtype=np.float32).reshape(h, w)

    # 3. 组合并转为 PyTorch Tensor
    arr = np.stack([R, G, B], axis=-1)
    
    in_file.close()
    return torch.from_numpy(arr).to(device)

def save_probe(
    probe_idx: int,
    pos: list,      # (3,) 探针的 xyz 坐标
    env_d: torch.Tensor,    # (H, W, 3) 漫反射全景图
    mips: list,             # list of (H_i, W_i, 3)，长度为 num_mips
    save_dir: str,
):
    """
    存储格式：
      {n}_d.exr         - 漫反射分量
      {n}_m{i}.exr      - 镜面反射 mip 第 i 张
      {n}_p.json        - 探针元数据 (pos + num_mips)
    """
    os.makedirs(save_dir, exist_ok=True)
    n = probe_idx
    num_mips = len(mips)

    # --- 漫反射分量 ---
    save_exr(env_d, os.path.join(save_dir, f"{n}_d.exr"))

    # --- 各级 Mip ---
    for i, mip in enumerate(mips):
        save_exr(mip, os.path.join(save_dir, f"{n}_m{i}.exr"))

    # --- 元数据 JSON ---
    meta = {
        "pos": pos,   # [x, y, z]
        "num_mips": num_mips,
    }
    with open(os.path.join(save_dir, f"{n}_p.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_probe(probe_idx: int, save_dir: str, device="cpu"):
    """
    返回:
      pos     : torch.Tensor (3,)
      env_d   : torch.Tensor (H, W, 3)   漫反射分量
      mips    : list of torch.Tensor      各级 mip
    """
    n = probe_idx

    # --- 元数据 ---
    with open(os.path.join(save_dir, f"{n}_p.json"), "r") as f:
        meta = json.load(f)

    pos = meta["pos"]
    num_mips = meta["num_mips"]

    # --- 漫反射 ---
    env_d = load_exr(os.path.join(save_dir, f"{n}_d.exr"), device=device)

    # --- Mip 链 ---
    mips = [
        load_exr(os.path.join(save_dir, f"{n}_m{i}.exr"), device=device)
        for i in range(num_mips)
    ]

    return pos, env_d, mips

def load_probe_pos(probe_idx: int, save_dir: str, device="cpu"):
    """
    返回:
      pos     : torch.Tensor (3,)
    """
    n = probe_idx
    # --- 元数据 ---
    with open(os.path.join(save_dir, f"{n}_p.json"), "r") as f:
        meta = json.load(f)
    return meta["pos"]

def save_all_probes(
    pos_list: torch.Tensor,   # (N, 3)
    env_ds: list,             # N x (H, W, 3)
    mipss: list,              # N x num_mips x (H, W, 3)
    save_dir: str,
):
    N = len(env_ds)
    for n in range(N):
        save_probe(
            probe_idx=n,
            pos=pos_list[n],
            env_d=env_ds[n],
            mips=mipss[n],
            save_dir=save_dir,
        )
    print(f"[LightProbe] 已保存 {N} 个探针到 {save_dir}")

def load_all_probes_pos(device, save_dir: str):
    """
    自动扫描 save_dir 里存了多少个探针（按 *_p.json 计数），全部读回
    """
    json_files = sorted([
        f for f in os.listdir(save_dir) if f.endswith("_p.json")
    ], key=lambda f: int(f.split("_p.json")[0]))   # 按数字序号排序

    pos_list = []
    for f in json_files:
        n = int(f.split("_p.json")[0])
        pos = load_probe_pos(n, save_dir, device=device)
        pos_list.append(pos)

    # pos_tensor = torch.stack(pos_list, dim=0)   # (N, 3)
    print(f"[LightProbe] 已读取 {len(pos_list)} 个探针pos from {save_dir}")
    return pos_list
