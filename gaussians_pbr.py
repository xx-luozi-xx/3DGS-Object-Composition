import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyElement, PlyData

from gaussians import Gaussians

class Gaussians_PBR(Gaussians):
    def __init__(self):
        super().__init__()
        self.normal    = None  # [N, 3]  原始未归一化向量，渲染前需 F.normalize
        self.roughness = None  # [N,]    logit 空间，渲染前需 sigmoid
        self.metallic  = None  # [N,]    logit 空间，渲染前需 sigmoid

    def _load_from_ply(self, ply_path: str):
        """继承基类和 Gaussians 的几何 + SH 加载，追加 PBR 字段"""
        super()._load_from_ply(ply_path)
        v = self.plydata['vertex'].data

        available = set(v.dtype.names)

        # ------ normal [N, 3] ------
        if all(f"normal_{i}" in available for i in range(3)):
            self.normal = torch.from_numpy(
                np.stack([v['normal_0'], v['normal_1'], v['normal_2']], axis=1)
            ).float()  # 训练空间：未归一化原始向量
        else:
            # PLY 里没有 normal 字段：用零向量占位，等待训练
            N = self.xyz.shape[0]
            self.normal = torch.zeros(N, 3)
            self.normal[:, 2] = 1.0  # 默认朝 +Z 方向

        # ------ roughness [N,] ------
        if 'roughness' in available:
            self.roughness = torch.from_numpy(
                v['roughness'].astype(np.float32)
            ).float()  # 训练空间：logit 值
        else:
            N = self.xyz.shape[0]
            self.roughness = torch.zeros(N)  # logit(0.5) = 0.0

        # ------ metallic [N,] ------
        if 'metallic' in available:
            self.metallic = torch.from_numpy(
                v['metallic'].astype(np.float32)
            ).float()  # 训练空间：logit 值
        else:
            N = self.xyz.shape[0]
            self.metallic = torch.full((N,), torch.logit(torch.tensor(0.1)).item())

    def _init_from_pointcloud(self, points: torch.Tensor, sh_degree: int = 3):
        """继承 Gaussians 的几何 + SH 初始化，追加 PBR 字段随机初始化"""
        # 1. 父类负责 xyz, scale, rot, opacity, color
        super()._init_from_pointcloud(points, sh_degree=sh_degree)

        N = self.num_points()

        # ------ normal [N, 3] ------
        raw_normal = torch.randn(N, 3)
        self.normal = F.normalize(raw_normal, dim=-1)  # [N, 3]

        # ------ roughness [N,] ------
        init_roughness = 0.5
        self.roughness = torch.full(
            (N,), torch.logit(torch.tensor(init_roughness)).item()
        )  # [N,]   训练空间值全为 0.0 = logit(0.5)

        # ------ metallic [N,] ------
        init_metallic = 0.1
        self.metallic = torch.full(
            (N,), torch.logit(torch.tensor(init_metallic)).item()
        )  # [N,]   训练空间值全为约 -2.197

    def add_background_gaussians(self):
        raise NotImplementedError("没有")

    def write_ply(self, ply_path: str, binary: bool = True):
        """
        将当前 Gaussians_PBR 对象写入 PLY 文件。
        此处仅负责剥离 Tensor 并转为 CPU numpy，实际写入交由外部接口 gs_pbr_save_ply 完成。

        参数:
            ply_path (str): 输出文件路径
            binary (bool): 是否以 binary_little_endian 写出，默认 True
        """
        N = self.num_points()

        xyz       = self.xyz.detach().cpu().numpy()        # (N,3)
        scale     = self.scale.detach().cpu().numpy()      # (N,3)
        rot       = self.rot.detach().cpu().numpy()        # (N,4)
        opacity   = self.opacity.detach().cpu().numpy()    # (N,)
        shs       = self.color.detach().cpu().numpy()      # (N,16,3)
        
        normal    = self.normal.detach().cpu().numpy()     # (N,3)
        roughness = self.roughness.detach().cpu().numpy()  # (N,)
        metallic  = self.metallic.detach().cpu().numpy()   # (N,)

        gs_pbr_save_ply(
            ply_path=ply_path,
            N=N,
            xyz=xyz,
            scale=scale,
            rot=rot,
            opacity=opacity,
            shs=shs,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            binary=binary
        )


def gs_pbr_save_ply(ply_path, N, xyz, scale, rot, opacity, shs, normal, roughness, metallic, binary=True):
    """
    将 PBR 高斯数据写入 PLY 文件，字段顺序与标准格式完全一致。
    支持作为外部接口独立调用。

    参数:
        ply_path (str): 输出文件路径
        N: int          高斯点数
        xyz:            (N,3)      numpy in cpu
        scale:          (N,3)      numpy in cpu
        rot:            (N,4)      numpy in cpu
        opacity:        (N,)       numpy in cpu
        shs:            (N,16,3)   numpy in cpu
        normal:         (N,3)      numpy in cpu (PBR)
        roughness:      (N,)       numpy in cpu (PBR)
        metallic:       (N,)       numpy in cpu (PBR)
        binary (bool):  是否以 binary 写出，默认 True
    """

    # -----------------------------------------
    # 1. 拆分球谐系数并恢复通道顺序
    # -----------------------------------------
    f_dc = shs[:, 0]          # (N,3)
    f_rest = shs[:, 1:]       # (N,15,3)

    # 展开并恢复原 ply 的 channel-major 写法 (R1..R15, G1..G15, B1..B15)
    f_rest_flat = f_rest.reshape(N, 15 * 3)
    idx = np.array([i + 15 * j for i in range(15) for j in range(3)])    
    inv_idx = np.argsort(idx)
    f_rest_original = f_rest_flat[:, inv_idx]   # (N,45)

    # -----------------------------------------
    # 2. 构建 ply 顶点结构化数组的数据类型定义
    # -----------------------------------------
    dtype_list = [
        ('x',        'f4'),
        ('y',        'f4'),
        ('z',        'f4'),
        ('scale_0',  'f4'),
        ('scale_1',  'f4'),
        ('scale_2',  'f4'),
        ('rot_0',    'f4'),
        ('rot_1',    'f4'),
        ('rot_2',    'f4'),
        ('rot_3',    'f4'),
        ('opacity',  'f4'),
        ('f_dc_0',   'f4'),
        ('f_dc_1',   'f4'),
        ('f_dc_2',   'f4'),
    ]

    # 添加 SH 剩余部分
    for i in range(45):
        dtype_list.append((f"f_rest_{i}", 'f4'))

    # 追加 PBR 字段定义
    dtype_list.extend([
        ('normal_0',  'f4'),
        ('normal_1',  'f4'),
        ('normal_2',  'f4'),
        ('roughness', 'f4'),
        ('metallic',  'f4'),
    ])

    vertex_data = np.zeros(N, dtype=dtype_list)

    # -----------------------------------------
    # 3. 填充值
    # -----------------------------------------
    # 几何与基础属性
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]

    vertex_data['scale_0'] = scale[:, 0]
    vertex_data['scale_1'] = scale[:, 1]
    vertex_data['scale_2'] = scale[:, 2]

    vertex_data['rot_0'] = rot[:, 0]
    vertex_data['rot_1'] = rot[:, 1]
    vertex_data['rot_2'] = rot[:, 2]
    vertex_data['rot_3'] = rot[:, 3]

    vertex_data['opacity'] = opacity

    # 颜色属性
    vertex_data['f_dc_0'] = f_dc[:, 0]
    vertex_data['f_dc_1'] = f_dc[:, 1]
    vertex_data['f_dc_2'] = f_dc[:, 2]

    for i in range(45):
        vertex_data[f"f_rest_{i}"] = f_rest_original[:, i]

    # PBR 属性
    vertex_data['normal_0']  = normal[:, 0]
    vertex_data['normal_1']  = normal[:, 1]
    vertex_data['normal_2']  = normal[:, 2]
    vertex_data['roughness'] = roughness
    vertex_data['metallic']  = metallic

    # -----------------------------------------
    # 4. 写出 PLY 文件
    # -----------------------------------------
    ply_el = PlyElement.describe(vertex_data, 'vertex')

    if binary:
        PlyData([ply_el], text=False).write(ply_path)
    else:
        PlyData([ply_el], text=True).write(ply_path)


