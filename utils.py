import os
import json
import math
import struct

from typing import Literal
from PIL import Image
import OpenEXR
import Imath
import torch
import numpy as np
from torch import Tensor
from sklearn.neighbors import NearestNeighbors


def linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    将线性颜色空间(Linear)精确转换为 sRGB 颜色空间
    使用 IEC 61966-2-1 标准公式
    """
    # EXR是HDR格式，值域可能超过1.0，sRGB转换前需要先 clamp 到 [0, 1]（或执行Tone Mapping）
    # 这里使用标准的 clamp 以匹配普通LDR显示
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    mask = tensor <= 0.0031308
    srgb_tensor = torch.empty_like(tensor)
    
    # 线性部分
    srgb_tensor[mask] = tensor[mask] * 12.92
    # Gamma 曲线部分
    srgb_tensor[~mask] = 1.055 * torch.pow(tensor[~mask], 1.0 / 2.4) - 0.055
    
    return srgb_tensor

def srgb_to_linear(x: Tensor)-> torch.Tensor:
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow((x + 0.055) / 1.055, 2.4)
    )

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def fibonacci_sphere(n):
    """
    生成一个在单位球上均匀分布的点集（n个点）
    返回np数组 (n, 3)
    """
    i = np.arange(0, n)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
    y = 1 - (2*i) / (n - 1)             # y 从 1 到 -1
    radius = np.sqrt(1 - y*y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    pts = np.stack([x, y, z], axis=1)
    return pts

def cv_show(img:torch.Tensor, win_name:str="", scale=1.0):
    """
    显示单张PyTorch张量图像
    如果是0~1值域, 确保clip
    """
    import numpy as np
    import cv2

    if img.ndim == 4:
        img = img[0]

    # 分离计算图，转到CPU，转为numpy
    img = img.detach().cpu().numpy()
    
    if scale != 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # 缩放到0-255
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # RGB转BGR
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(win_name, img)
    return cv2.waitKey(1) & 0xFF

def plt_show(imgs_nxm):
    """
    用plt显示numpy的n*m图像组, 可以用列表组织图像
    """
    import matplotlib.pyplot as plt
    
    rows = len(imgs_nxm)
    cols = max([len(row) for row in imgs_nxm])

    fig, axs = plt.subplots(rows, cols, squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs_nxm[i][j])
            axs[i, j].axis("off")
    plt.show()

def create_camera_intrinsic(image_size=(512, 512), fov=60.0, device=None):
    """
    创建相机内参矩阵 K
    
    参数:
    - image_size: (height, width) 图像尺寸
    - fov: 视野角度 (degrees)
    
    返回:
    - K:[3, 3] Tensor相机内参矩阵
    """
    height, width = image_size
    
    # 计算焦距 (以像素为单位)

    if not isinstance(fov, torch.Tensor):
        fov = torch.tensor(fov, dtype=torch.float32, device=device)
    fov_rad = fov * (torch.pi / 180.0)
    focal_length = width / (2 * torch.tan(fov_rad / 2))
    
    # 构建内参矩阵
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length  # fx
    K[1, 1] = focal_length  # fy
    K[0, 2] = width / 2     # cx
    K[1, 2] = height / 2    # cy
    
    return K

def create_view_matrix(position, look_at, up, device=None):
    """
    Create a world-to-camera (view) matrix for gsplat.
    c = Mw
    """
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, device=device).float()
    if not isinstance(look_at, torch.Tensor):
        look_at = torch.tensor(look_at, device=device).float()
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, device=device).float()

    # Compute camera basis
    z_axis = torch.nn.functional.normalize(look_at - position, dim=0)  # camera forward
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=0)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis), dim=0)
    
    assert torch.norm(x_axis) > 1e-6, f"视线方向与up向量接近共线, x_axis模长: {torch.norm(x_axis)}"
    # while torch.norm(x_axis) == 0: # 视线和up共线，引入微小扰动
    #     position[2] += 0.00001
    #     z_axis = torch.nn.functional.normalize(look_at - position, dim=0)  # camera forward
    #     x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=0)
    #     y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis), dim=0)

    # Create rotation matrix and translation
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
    translation = -rotation.T @ position

    # Assemble view matrix (world-to-camera)
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = rotation.T
    view_matrix[:3, 3] = translation

    return view_matrix

def create_nerf_dataset_structure(output_dir="data/0"):
    dirs_to_create = [
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "val"), 
        os.path.join(output_dir, "test")
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def save_pos_nerf_dataset(output_dir, sub_type: Literal["train", "test", "val"], poses:torch.Tensor, fov):
    """
    poses should be c2w
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_dir = os.path.join(output_dir, sub_type)

    dataset = {
        "camera_angle_x": float(np.deg2rad(fov)),
        "frames": [],
    }

    for i, pos in enumerate(poses):
        # 转换为Python列表
        frame = {
            "file_path": f'./{sub_type}/r_{i:05d}',
            "rotation": float((i/poses.shape[0]) *2* math.pi),
            "transform_matrix": pos.tolist()
        }
        
        dataset["frames"].append(frame)
    
    output_path = os.path.join(output_dir, f"transforms_{sub_type}.json")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    return output_path    

def save_img_nerf_dataset(output_dir, sub_type: Literal["train", "test", "val"], imgs:torch.Tensor):
    
    output_dir = os.path.join(output_dir, sub_type)
    
    for i, img in enumerate(imgs):
        # 将 tensor 转换为 numpy array
        img_array = img.cpu().numpy()
        # 如果值范围是 [0, 1]，转换为 [0, 255]
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # 创建 PIL 图像并保存
        save_path = os.path.join(output_dir,f'r_{i:05d}.png')
        img = Image.fromarray(img_array)
        img.save(save_path)
        
def load_colmap_points3d_bin(path: str, device="cpu"):
    """
    读取 COLMAP 的 points3D.bin 文件
    返回: 
        xyz: (N, 3) float32 Tensor
        rgb: (N, 3) float32 Tensor, 归一化到 [0, 1]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "rb") as fid:
        # 1. 读取点云数量 (8 bytes, unsigned long long)
        num_points = struct.unpack("<Q", fid.read(8))[0]

        xyzs = np.empty((num_points, 3), dtype=np.float64)
        rgbs = np.empty((num_points, 3), dtype=np.uint8)

        # 2. 遍历每个点进行解析
        for i in range(num_points):
            # 读取基础属性: 
            # POINT3D_ID(8 bytes), X(8), Y(8), Z(8), R(1), G(1), B(1), ERROR(8) = 总计 43 bytes
            # 格式化字符串 "<QdddBBBd" 对应: 
            # < = 小端序, Q = uint64, d = float64, B = uint8
            props_data = fid.read(43)
            props = struct.unpack("<QdddBBBd", props_data)
            
            xyzs[i] = props[1:4] # 提取 X, Y, Z
            rgbs[i] = props[4:7] # 提取 R, G, B
            
            # 读取 Track 长度 (这个点在多少张图里被看到了)
            track_length = struct.unpack("<Q", fid.read(8))[0]
            
            # 跳过 Track 详细信息 (每个 track 包含 IMAGE_ID(4 bytes) 和 POINT2D_IDX(4 bytes) = 8 bytes)
            # 因为渲染/初始化不需要这些匹配信息，直接 seek 跳过可以大幅提升读取速度
            fid.seek(track_length * 8, 1) # 1 表示从当前位置偏移

    # 3. 转换为 PyTorch Tensor
    xyz_tensor = torch.from_numpy(xyzs).to(torch.float32).to(device)
    rgb_tensor = torch.from_numpy(rgbs).to(torch.float32).to(device) / 255.0

    print(f"[COLMAP Loader] 成功加载 points3D.bin, 包含 {num_points} 个点")
    return xyz_tensor, rgb_tensor

def load_exr(path: str, channels: str = "RGB") -> torch.Tensor:
    """
    通用且鲁棒的 EXR 读取函数
    """
    in_file = OpenEXR.InputFile(path)
    dw = in_file.header()['dataWindow']
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1
    F  = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # 获取该 EXR 文件中实际存在的通道名称列表
    available_channels = list(in_file.header()['channels'].keys())
    
    channels = channels.upper() 
    channel_list = []
    
    for c in channels:
        actual_c = c
        # 单通道直接读
        if actual_c not in available_channels:
            if len(available_channels) == 1:
                actual_c = available_channels[0]
            else:
                in_file.close()
                raise ValueError(
                    f"读取失败！请求通道 '{c}'，但 {path} 中只有: {available_channels}"
                )
                
        # 逐个读取请求的通道
        c_data = np.frombuffer(in_file.channel(actual_c, F), dtype=np.float32).reshape(h, w)
        channel_list.append(c_data)
        
    in_file.close()
    
    # 无论传入的是 "R" (1个通道) 还是 "RGBA" (4个通道)，stack 都能完美处理
    # copy() 是为了防止 NumPy 数组和 PyTorch 张量底层的内存连续性问题
    stacked_data = np.stack(channel_list, axis=-1).copy()
    
    return torch.from_numpy(stacked_data)  # 返回 [H, W, len(channels)]

def load_png(path: str, channels: str = "RGB") -> torch.Tensor:
    """
    通用 PNG 读取函数，与 load_exr 的接口和输出格式保持严格对齐。
    
    Args:
        path (str): PNG 文件路径
        channels (str): 需要读取的通道组合，例如 "RGB", "RGBA", "R", "BA", "A"
        
    Returns:
        torch.Tensor: 形状为 [H, W, C] 的张量, C 为请求的通道数量, float32 数据类型。
                      值范围已经被归一化到 [0.0, 1.0]。
    """
    # 使用 PIL 读取可以很好地保留 PNG 的原始通道数 (RGB(3) 或 RGBA(4))
    img = Image.open(path)
    img_np = np.array(img)
    
    # 防御性处理：如果是单通道灰度图，补齐通道维度 -> [H, W, 1]
    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=-1)
        
    num_channels = img_np.shape[2]
    
    # 统一转为大写
    channels = channels.upper() 
    
    # 定义通道到索引的映射关系
    channel_map = {'R': 0, 'G': 1, 'B': 2, 'A': 3}
    
    channel_list = []
    for c in channels:
        if c not in channel_map:
            raise ValueError(f"不支持的通道标识: {c}")
            
        ch_idx = channel_map[c]
        
        # 容错处理：如果请求了 A (Alpha) 通道，但原图只是普通的 RGB 图
        if ch_idx >= num_channels:
            if c == 'A':
                # 没有 Alpha 通道时，默认完全不透明 (255)
                c_data = np.full((img_np.shape[0], img_np.shape[1]), 255.0, dtype=np.float32)
            else:
                # 请求了越界的 RGB 通道（比如灰度图强行要 B 通道）
                c_data = img_np[..., 0].astype(np.float32) 
        else:
            c_data = img_np[..., ch_idx].astype(np.float32)
            
        channel_list.append(c_data)
        
    # 无论是一个通道还是多个通道，都能完美拼接
    stacked_data = np.stack(channel_list, axis=-1)
    
    # PNG 取值范围是 0-255，我们需要将其转为与渲染器匹配的 0.0-1.0 浮点数
    stacked_data = stacked_data / 255.0
    
    # contiguous() 解决底层内存不连续问题（类似 NumPy 的 .copy()）
    return torch.from_numpy(stacked_data).contiguous()

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    将轴角向量转换为旋转矩阵，仅依赖 PyTorch。

    参数:
        axis_angle: 浮点 Tensor，形状 (..., 3)。
                    方向表示旋转轴，长度表示旋转角度（弧度）。

    返回:
        形状 (..., 3, 3) 的旋转矩阵。
        输出的 dtype、device 与输入相同。
        零向量对应单位矩阵。
    """
    if not isinstance(axis_angle, torch.Tensor):
        raise TypeError("axis_angle 必须是 torch.Tensor")

    if axis_angle.dim() < 1 or axis_angle.shape[-1] != 3:
        raise ValueError(
            f"axis_angle 的形状必须是 (..., 3)，实际为 {axis_angle.shape}"
        )

    if axis_angle.dtype not in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64
    ):
        raise TypeError("axis_angle 必须是浮点 Tensor")

    original_dtype = axis_angle.dtype

    # 半精度输入使用 float32 完成内部计算，最后转回原 dtype。
    if original_dtype in (torch.float16, torch.bfloat16):
        v = axis_angle.float()
    else:
        v = axis_angle

    theta2 = (v * v).sum(dim=-1)  # (...,)
    small = theta2 < 1e-4

    # 必须先保护 sqrt 和除法的输入，不能算出 NaN 后再用 where 掩盖。
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta_safe = torch.sqrt(theta2_safe)

    # 泰勒展开只使用小角度数据，避免大角度在未选中的分支中溢出。
    t = torch.where(small, theta2, torch.zeros_like(theta2))

    # A = sin(theta) / theta
    A_small = 1.0 + t * (-1.0 / 6.0 + t * (1.0 / 120.0 - t / 5040.0))
    A = torch.where(small, A_small, torch.sin(theta_safe) / theta_safe)

    # B = (1 - cos(theta)) / theta^2
    # 非小角度使用半角恒等式，避免直接计算 1 - cos(theta)。
    B_small = 0.5 + t * (-1.0 / 24.0 + t * (1.0 / 720.0 - t / 40320.0))
    B = torch.where(
        small,
        B_small,
        2.0 * (torch.sin(0.5 * theta_safe) / theta_safe) ** 2,
    )

    x, y, z = v.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z

    # 展开 R = I + A [v]× + B [v]×²，避免额外的矩阵乘法。
    R = torch.stack(
        (
            1.0 - B * (yy + zz), B * xy - A * z,       B * xz + A * y,
            B * xy + A * z,       1.0 - B * (xx + zz), B * yz - A * x,
            B * xz - A * y,       B * yz + A * x,       1.0 - B * (xx + yy),
        ),
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))

    return R.to(dtype=original_dtype)
