import math

import torch
import numpy as np
from plyfile import PlyElement, PlyData

import utils

class Gaussian_Base:
    def __init__(self):
        """
        基础构造函数。只声明属性，不进行任何赋值。
        真实的初始化交由 @classmethod 工厂方法来完成。
        """
        self.xyz = None
        self.scale = None
        self.rot = None
        self.opacity = None

    @classmethod
    def from_ply(cls, ply_path: str):
        """工厂方法 1：从 PLY 文件初始化"""
        instance = cls() # 创建当前类（或子类）的实例
        instance._load_from_ply(ply_path)
        return instance

    @classmethod
    def from_pointcloud(cls, points: torch.Tensor, *args, **kwargs):
        """工厂方法 2：从 [N, 3] 的点云张量初始化"""
        instance = cls()
        instance._init_from_pointcloud(points, *args, **kwargs)
        return instance

    def _load_from_ply(self, ply_path):
        """读取 ply 的基础属性逻辑"""
        self.plydata = PlyData.read(ply_path)
        v = self.plydata['vertex'].data

        self.xyz = torch.from_numpy(np.stack([v['x'], v['y'], v['z']], axis=1)).float()
        self.scale = torch.from_numpy(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)).float()
        self.rot = torch.from_numpy(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)).float()
        self.opacity = torch.from_numpy(v['opacity']).float()

    def _init_from_pointcloud(self, points: torch.Tensor):
        """从点云初始化的基础属性(随机)"""

        N = points.shape[0]
        self.xyz = points.float()

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (utils.knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        self.scale = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]  

        self.rot = torch.rand((N, 4))  # [N, 4]
        self.rot[:, 3] = 1.0

        init_opacity = 0.1
        self.opacity = torch.logit(torch.full((N,), init_opacity))  # [N,]   

    def num_points(self):
        """动态返回当前高斯点的数量"""
        if self.xyz is None:
            return 0
        return self.xyz.shape[0]

    def to(self, device):
        """
        [智能基类方法]：自动遍历对象的所有属性，把所有 Tensor 都移到指定 device
        子类不需要重写此方法！
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
        return self

    def down_sampling(self, ratio=0.1):
        """
        [智能基类方法]：自动对所有第一维度长度为 N 的 Tensor 进行一致的下采样
        子类不需要重写此方法！
        """
        N = self.xyz.shape[0]
        M = int(N * ratio)
        idx = np.random.choice(N, size=M, replace=False)

        # 遍历所有属性，只要是 Tensor 且长度等于 N，就切片
        for attr_name, attr_value in list(self.__dict__.items()):
            if isinstance(attr_value, torch.Tensor) and attr_value.shape[0] == N:
                setattr(self, attr_name, attr_value[idx])

        print(f"[{self.__class__.__name__}] Down-sampled: {N} -> {M}")

    def apply_uniform_scale(self, scale_factor: float):
        """
        [智能基类方法]：对整个高斯模型进行均匀缩放 (放大或缩小)。
        参数:
            scale_factor (float): 缩放倍数。例如 2.0 表示放大一倍，0.5 表示缩小一半。
        """
        if scale_factor <= 0:
            raise ValueError("缩放因子必须大于 0")
            
        if self.xyz is not None and self.scale is not None:
            with torch.no_grad():
                # 1. 位置在物理空间，直接相乘
                self.xyz *= scale_factor
                
                # 2. 尺度在对数空间，表现为加上 ln(scale_factor)
                self.scale += math.log(scale_factor)
                
        return self

    def __repr__(self):
        """动态生成字符串表示，列出所有 Tensor 属性"""
        tensors_info = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                tensors_info.append(f"{k}:{tuple(v.shape)}")
        
        info_str = " ".join(tensors_info)
        return (
            f"<{self.__class__.__name__}: {len(self.xyz)} points | {info_str} | "
            f"x:[{self.xyz[:, 0].min():.2f}, {self.xyz[:, 0].max():.2f}]>"
        )

    def write_ply(self, ply_path, binary=True):
        """留给子类实现，因为每个子类要保存的特定字段不同"""
        raise NotImplementedError("子类必须实现具体的 write_ply 方法！")
        
    def add_background_gaussians(self, *args, **kwargs):
        """留给子类实现，因为背景点的颜色/材质初始化方式不同"""
        raise NotImplementedError("子类必须实现 add_background_gaussians 方法！")

class Gaussians(Gaussian_Base):
    def __init__(self):
        super().__init__()
        # 声明 PBR 专属属性
        self.color = None
        self.sh_degree = None
       
    def _load_from_ply(self, ply_path):
        super()._load_from_ply(ply_path)
        
        # 2. 解析特有的颜色属性
        v = self.plydata['vertex'].data
        f_dc = np.vstack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).T   

        self.sh_degree = 3
        N = len(v)
        f_rest = np.zeros((N, 45), dtype=np.float32)
        available_names = set(v.dtype.names) 
        for i in range(45):
            key = f"f_rest_{i}"
            if key in available_names:
                f_rest[:, i] = v[key] 

        idx = np.array([i + 15*j for i in range(15) for j in range(3)])  
        f_rest = f_rest[:, idx]  
        f_rest = f_rest.reshape(-1, 15, 3)
        
        shs = np.concatenate([f_dc[:, np.newaxis, :], f_rest], axis=1)
        self.color = torch.from_numpy(shs).float()

    def _init_from_pointcloud(self, points: torch.Tensor, sh_degree = 3):
        super()._init_from_pointcloud(points)
        self.sh_degree = sh_degree

        N = self.num_points()
        rgbs = torch.rand((N, 3))
        # color is SH coefficients.
        color = torch.zeros((N, (self.sh_degree + 1) ** 2, 3))  # [N, K, 3]
        color[:, 0, :] = utils.rgb_to_sh(rgbs)
        self.color = color

    def add_background_gaussians(self, num_bg=1000, bg_color=[1,1,1], bg_dis=10, hemisphere = False):
        """
        添加背景高斯点
        """
        center = self.xyz.mean(dim=0)  # (3,)
        r = torch.norm(self.xyz - center[None, :], dim=1).max()
        R = bg_dis*r

        # 生成背景点
        bg_xyz = utils.fibonacci_sphere(num_bg) 
        bg_xyz = torch.from_numpy(bg_xyz).float().to(self.xyz.device)
         # ====== 极简半球处理 ======
        if hemisphere:  # 假设你加了这个参数
            bg_xyz = bg_xyz[bg_xyz[:, 1] >= 0]  # 剔除 y < 0 的点
            num_bg = bg_xyz.shape[0]            # ⚠️ 关键：必须更新 num_bg，防止后续维度报错
        # ==========================

        bg_xyz = bg_xyz* R + center[None, :]  # 平移到中心
        
        bg_scale = torch.ones(num_bg, 3).float().to(self.scale.device)
        bg_scale = bg_scale * torch.sqrt((8*torch.pi*(R**2))/(torch.sqrt(torch.tensor(3.0))*num_bg))
        bg_scale = torch.log(bg_scale) #像素空间→训练空间。渲染前会转回来

        bg_rot = torch.zeros(num_bg, 4).float().to(self.rot.device)
        bg_rot[:, 0] = 1.0 #无旋转

        bg_opacity = torch.ones(num_bg).float().to(self.opacity.device)
        bg_opacity = torch.logit(bg_opacity) #像素空间→训练空间。渲染前会转回来

        bg_colors = torch.zeros(num_bg, 16, 3).float().to(self.color.device)
        bg_colors[:, 0, :] = torch.tensor(bg_color).float()
        C0 = 0.28209479177387814 # C0 = 1/torch.sqrt(torch.tensor(4.0 * torch.pi))
        bg_colors[:, 0, :] = (bg_colors[:, 0, :]-0.5)/C0  #像素空间→3dgs球谐空间

        # 合并背景点与原始点
        self.xyz = torch.cat([self.xyz, bg_xyz], dim=0)
        self.scale = torch.cat([self.scale, bg_scale], dim=0)
        self.rot = torch.cat([self.rot, bg_rot], dim=0)
        self.opacity = torch.cat([self.opacity, bg_opacity], dim=0)
        self.color = torch.cat([self.color, bg_colors], dim=0)

    def write_ply(self, ply_path, binary=True):
        """
        将当前 Gaussians 对象写入 PLY 文件，字段顺序与读取的格式完全一致。

        参数:
            ply_path (str): 输出文件路径
            binary (bool): 是否以 binary_little_endian 写出，默认 True
        """

        N = self.xyz.shape[0]

        # -----------------------------------------
        # 1. 基础字段准备
        # -----------------------------------------
        xyz = self.xyz.cpu().numpy()               # (N,3)
        scale = self.scale.cpu().numpy()           # (N,3)
        rot = self.rot.cpu().numpy()               # (N,4)
        opacity = self.opacity.cpu().numpy()       # (N,)
        shs = self.color.cpu().numpy()             # (N,16,3)

        gs_save_ply(
            ply_path=ply_path,
            N=N,
            xyz=xyz,
            scale=scale,
            rot=rot,
            opacity=opacity,
            shs=shs,
            binary=binary
        )

def gs_save_ply(ply_path, N, xyz, scale, rot, opacity, shs, binary=True):
    """
    将当前 Gaussians 对象写入 PLY 文件，字段顺序与读取的格式完全一致。

    参数:
        ply_path (str): 输出文件路径

        N:int       高斯点数
        xyz       # (N,3)      numpy in cpu
        scale     # (N,3)      numpy in cpu
        rot       # (N,4)      numpy in cpu
        opacity   # (N,)       numpy in cpu
        shs       # (N,16,3)   numpy in cpu
        
        binary (bool): 是否以 binary_little_endian 写出，默认 True
    """

    # -----------------------------------------
    # 2. 拆分球谐系数：DC + f_rest(45)
    #    注意：你读取时把 f_rest 重排列成 [R1,G1,B1 | R2,G2,B2 | ...]
    #    写回时要恢复“原始顺序”:
    #       [R1..R15 | G1..G15 | B1..B15]
    # -----------------------------------------

    f_dc = shs[:, 0]          # (N,3)
    f_rest = shs[:, 1:]       # (N,15,3)

    # 将 f_rest(N,15,3) reshape 成 (N,45)
    # 当前顺序是 freq-major: [R1,G1,B1 | R2,G2,B2 | ...]
    # 我们必须恢复为 channel-major（原 ply 写法）:
    #   R1..R15, G1..G15, B1..B15

    # 先展开 (N,45)
    f_rest_flat = f_rest.reshape(N, 15*3)

    # 逆置换索引：由 freq→channel
    # freq-major 索引为 idx = [i + 15*j for i in range(15) for j in range(3)]
    # 我们需要其逆操作 inv_idx
    idx = np.array([i + 15*j for i in range(15) for j in range(3)])    
    inv_idx = np.argsort(idx)

    # 按原来的 ply 写法恢复顺序
    f_rest_original = f_rest_flat[:, inv_idx]   # (N,45)


    # -----------------------------------------
    # 3. 构建 ply 顶点结构化数组
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

    # 添加 f_rest_0 ~ f_rest_44
    for i in range(45):
        dtype_list.append((f"f_rest_{i}", 'f4'))

    vertex_data = np.zeros(N, dtype=dtype_list)

    # 填充值
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

    vertex_data['f_dc_0'] = f_dc[:, 0]
    vertex_data['f_dc_1'] = f_dc[:, 1]
    vertex_data['f_dc_2'] = f_dc[:, 2]

    for i in range(45):
        vertex_data[f"f_rest_{i}"] = f_rest_original[:, i]

    # -----------------------------------------
    # 4. 写出 PLY
    # -----------------------------------------
    ply_el = PlyElement.describe(vertex_data, 'vertex')

    if binary:
        PlyData([ply_el], text=False).write(ply_path)
    else:
        PlyData([ply_el], text=True).write(ply_path)




