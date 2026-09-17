from typing import Literal
import math

import gsplat
import torch
import torch.nn.functional as F
from torch import Tensor

from gs_panorama_rasterization import panorama_rasterization
from gs_feature_rasterization import feature_rasterization


from gaussians import Gaussians
from gaussians_pbr import Gaussians_PBR
import utils

def gs_render(
        device, gs: Gaussians, 
        view_origins, view_targets, 
        h_ws, Ks, 
        bg_colors, ups=[], 
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB+ED"
    ):
    
    xyz = gs.xyz
    scale = torch.exp(gs.scale)
    rot = gs.rot
    opacity =  torch.sigmoid(gs.opacity)
    color = gs.color

    #! Using Opencv camera coordinate system here
    if len(ups) == 0:
        ups = [[0,-1,0]] * len(view_origins)

    view_matrix = [utils.create_view_matrix(
        position=pos,
        look_at =tar,
        up = up,
        device=device
    ) 
    for pos, tar, up in zip(view_origins, view_targets, ups)
    ]

    view_matrix = torch.stack(view_matrix, dim=0)

    # render
    assert (h_ws == h_ws[0]).all().item(), "Consistent resolution required in h_ws"
    # print("<gs render>: rendering...", end="")
    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,
        quats=rot,
        scales=scale,
        opacities=opacity,
        colors=color,
        viewmats=view_matrix,
        Ks = Ks,
        height=h_ws[0][0],
        width=h_ws[0][1],
        render_mode=render_mode,
        packed=False,
        sh_degree=3,
        backgrounds=bg_colors,
    )
    # print("\r<gs render>: rendering [finished]",)
    
    rgb, d = None, None
    if render_mode in ["RGB", "RGB+D", "RGB+ED"]:
        rgb = render_colors[..., :3]
    if render_mode in ["D", "ED", "RGB+D", "RGB+ED"]:
        d = render_colors[..., -1]
        
    return rgb, d

def gs_unified_render(device, gs: Gaussians_PBR, view_origins, view_targets, h_ws, Ks, bg_colors, ups=[], IOR1=1.0, IOR2=1.5, view_mats=None):
    """
        if view_mats is provieded, then (view_origins, view_targets, ups) are no need
        view_mats will be used for gsplat
    """
    #! Using Opencv camera coordinate system here
    if len(ups) == 0:
        ups = [[0,-1,0]] * len(view_origins)

    if not isinstance(ups, Tensor):
        ups = torch.tensor(ups, dtype=torch.float32, device=device)

    view_matrix = [utils.create_view_matrix(
        position=pos,
        look_at =tar,
        up = up,
        device=device
    ) 
    for pos, tar, up in zip(view_origins, view_targets, ups)
    ]

    view_matrix = torch.stack(view_matrix, dim=0)    
    
    if view_mats is not None:
        view_matrix = view_mats

    C = view_matrix.shape[0]

    means = gs.xyz  # [N, 3]
    quats = gs.rot  # [N, 4]
    scales = torch.exp(gs.scale)  # [N, 3]
    opacities = torch.sigmoid(gs.opacity)  # [N,]
    colors = gs.color  # [N, K, 3]
    
    normals = F.normalize(gs.normal, p=2, dim=-1) # [N, 3]
    roughness = torch.sigmoid(gs.roughness) # [N,]
    
    metallics = torch.sigmoid(gs.metallic)
    metallics: Tensor  =torch.full_like(roughness, 0)# [N,] #! TensorIR数据集金属度都是0
    
    features = torch.cat([normals, roughness.unsqueeze(-1), metallics.unsqueeze(-1)], dim=-1)  # [N, 5]

    # render
    assert (h_ws == h_ws[0]).all().item(), "Consistent resolution required in h_ws"
    # print("<gs feature render>: rendering...", end="")
    renders, alphas, info = feature_rasterization(
        means = means,
        quats = quats,
        scales = scales,
        opacities = opacities,
        colors = colors,
        features=features,

        viewmats = view_matrix,
        Ks = Ks,
        height = h_ws[0][0],
        width = h_ws[0][1],

        packed = False,
        absgrad = True,
        sh_degree = gs.sh_degree,
        render_mode="RGB+ED",
    )
    # print("\r<gs feature render>: rendering [finished]",)
    # renders # [C, H, W, 3+3+1+1+1]， 

    albedo    = renders[..., :3]    # [C, H, W, 3]
    normal    = renders[..., 3:6]   # [C, H, W, 3]
    roughness = renders[..., 6]   # [C, H, W]
    metallics = renders[..., 7]   # [C, H, W]
    depth     = renders[..., -1]    # [C, H, W]
    mask = depth > 1e-4             # [C, H, W]
    mask = alphas.squeeze(-1)       # [C, H, W] #!黑边修复
    # making t_dict
    t_dict = {
        "albedo"    : albedo,
        "roughness" : roughness,
        "metallic"  : metallics,
    }
    
    # === making g_dict ===
    B, H, W = depth.shape
    
    # 将相机空间转换到世界空间 P
    # view_matrix 是 World-to-Camera 矩阵 [B, 4, 4]，求逆得到 Camera-to-World
    c2w = torch.inverse(view_matrix) 
    R_c2w = c2w[:, :3, :3]  # [B, 3, 3]
    T_c2w = c2w[:, :3, 3]   # [B, 3] ⚠️这正是真正的世界坐标系下的相机原点(cam_centers)！
    
    # 1. 提取真正的相机中心 (直接从位姿矩阵提取，无视传入的 view_origins)
    cam_centers_view = T_c2w.view(B, 1, 1, 3)

    # 2. 从深度图反算世界空间位置 P (Unprojection)
    # 生成像素网格 (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = x.float() + 0.5
    v = y.float() + 0.5

    # 提取内参
    fx = Ks[:, 0, 0].view(B, 1, 1)
    fy = Ks[:, 1, 1].view(B, 1, 1)
    cx = Ks[:, 0, 2].view(B, 1, 1)
    cy = Ks[:, 1, 2].view(B, 1, 1)

    # 计算相机空间坐标 (基于 OpenCV 坐标系：+Z向前, +X向右, +Y向下)
    x_cam = (u.view(1, H, W) - cx) * depth / fx
    y_cam = (v.view(1, H, W) - cy) * depth / fy
    z_cam = depth
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, H, W, 3]
    
    # P_world = P_cam @ R_c2w^T + T_c2w
    P = torch.matmul(p_cam, R_c2w.transpose(1, 2)) + cam_centers_view

    # 3. 计算观察向量 V (从表面点指向真实的相机中心)
    V = F.normalize(cam_centers_view - P, dim=-1, eps=1e-6)

    # 4. 重新归一化渲染出的法线
    # 【非常重要】：高斯的 Alpha 混合会导致累加后的法线向量长度缩短，必须重新 Normalize
    n = F.normalize(normal, dim=-1, eps=1e-6)

    # 5. 可选：用 Mask 清理背景像素，防止无穷远处的游离点干扰后续光追计算
    mask_float = mask.float().unsqueeze(-1)
    P = P * mask_float
    n = n * mask_float
    V = V * mask_float

    g_dict = {
        "mask": mask,                           # [C, H, W]  Boolean
        "depth": depth.unsqueeze(-1),           # [C, H, W, 1]
        "position": P,                          # [C, H, W, 3]
        "normal": n,                            # [C, H, W, 3]
        "view": V,                              # [C, H, W, 3]
    }
    # ==========================================================
    # 纯张量数学计算：折射与菲涅尔 (Refract & Fresnel)
    # ==========================================================
    eta1 = IOR1
    eta2 = IOR2

    l, n = -g_dict["view"], g_dict["normal"] 
    
    cos_theta = torch.sum(l * (-n), dim=-1).unsqueeze(-1)  
    i_p = l + n * cos_theta                                
    t_p = (eta1 / eta2) * i_p                              

    t_p_norm = torch.sum(t_p * t_p, dim=-1)                
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(-1) 

    t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(-1).expand_as(n) * (-n) 
    t = t_i + t_p
    t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=-1), min=1e-10)).unsqueeze(-1) 

    cos_theta_t = torch.sum(t * (-n), dim=-1).unsqueeze(-1) 

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
          torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
          torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach() 

    g_dict["refract_dir"] = t                       # (N,H,W,3) 折射光线的世界坐标方向
    g_dict["attenuate"] = attenuate                 # (N,H,W,1) 菲涅尔反射强度系数 (范围0~1)
    g_dict["total_reflect"] = totalReflectMask      # (N,H,W,3) 折射光线的世界坐标方向    

    return t_dict, g_dict


def gs_env_render(device, gs: Gaussians, origin=[0, 0, 0], out_h=1024, out_w=2048, fov_deg=90, bg_color=[1.0,1.0,1.0]):

    """
    用你的高斯渲染器生成球面环境贴图 (equirectangular map)
    参数:
        gs: 高斯模型
        origin: 相机原点 (torch.Tensor)
        out_h, out_w: 输出球面贴图尺寸
        fov_deg: 每个立方体面的视场角 (默认90)
    返回:
        equirect_img: [H, W, 3] torch.Tensor
    """
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin, device=device).float()
    if not isinstance(bg_color, torch.Tensor):
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
    # ---- 1. 定义 cube 六个方向 ----
    directions = {
        '-X': (torch.tensor([1, 0, 0], dtype=torch.float32, device=device),
            torch.tensor([0, -1, 0], dtype=torch.float32, device=device)),
        '+X': (torch.tensor([-1, 0, 0], dtype=torch.float32, device=device),
            torch.tensor([0, -1, 0], dtype=torch.float32, device=device)),
        '+Y': (torch.tensor([0, 1, 0], dtype=torch.float32, device=device),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device)),
        '-Y': (torch.tensor([0, -1, 0], dtype=torch.float32, device=device),
            torch.tensor([0, 0, -1], dtype=torch.float32, device=device)),
        '-Z': (torch.tensor([0, 0, 1], dtype=torch.float32, device=device),
            torch.tensor([0, -1, 0], dtype=torch.float32, device=device)),
        '+Z': (torch.tensor([0, 0, -1], dtype=torch.float32, device=device),
            torch.tensor([0, -1, 0], dtype=torch.float32, device=device)),
    }
    h_ws = torch.full((len(directions), 2), out_h, dtype=torch.float32, device=device) # [6, 2]

    view_origins, view_targets, fovs, ups = [], [], [], [],
    face_names = []
    for name, (dir_vec, up_vec) in directions.items():
        view_origins.append(origin)
        view_targets.append(origin + dir_vec)
        ups.append(up_vec)
        fovs.append(fov_deg)
        face_names.append(name)

    view_origins = torch.stack(view_origins, dim=0)   # [6, 3]
    view_targets = torch.stack(view_targets, dim=0)   # [6, 3]
    ups = torch.stack(ups, dim=0)                     # [6, 3]

    Ks = torch.stack([
        utils.create_camera_intrinsic(h_w, fov, device=device)
        for h_w, fov in zip(h_ws, fovs)
    ], dim=0).float().to(device)

    bg_colors = bg_color.repeat(6, 1)
    # ---- 2. 高斯渲染 ----
    images, ds = gs_render(
        device=device, 
        gs=gs, 
        view_origins=view_origins, 
        view_targets=view_targets, 
        h_ws=h_ws, 
        Ks=Ks, 
        ups=ups,
        bg_colors=bg_colors,
    )
   
    # 期望返回: images.shape = (6, H, W, 3)
    cube_imgs = {name: images[i] for i, name in enumerate(face_names)}

    # ---- 3. Cube -> Equirectangular 转换 ----
    H, W = out_h, out_w

    # theta, phi
    theta = torch.linspace(0, 2*torch.pi, W, device=device).unsqueeze(0).repeat(H,1)
    phi   = torch.linspace(0, torch.pi, H, device=device).unsqueeze(1).repeat(1,W)

    # dirs
    dirs = torch.stack([
        torch.sin(phi) * torch.cos(theta),  # x
        torch.cos(phi),                      # y
        torch.sin(phi) * torch.sin(theta)    # z
    ], dim=-1)

    x, y, z = dirs[...,0], dirs[...,1], dirs[...,2]

    # 防止除零
    eps = 1e-8
    abs_x, abs_y, abs_z = torch.abs(x)+eps, torch.abs(y)+eps, torch.abs(z)+eps

    # 找出每个像素对应的最大轴
    axis_max = torch.stack([abs_x, abs_y, abs_z], dim=-1).argmax(dim=-1)  # 0:X,1:Y,2:Z

    out = torch.zeros((H, W, 3), device=device)

    # mask 对应每个面
    mask_posx = (axis_max==0) & (x>0)
    mask_negx = (axis_max==0) & (x<0)
    mask_posy = (axis_max==1) & (y>0)
    mask_negy = (axis_max==1) & (y<0)
    mask_posz = (axis_max==2) & (z>0)
    mask_negz = (axis_max==2) & (z<0)

    # 计算 u,v
    u = torch.zeros_like(x)
    v = torch.zeros_like(y)

    def sample(face_img, u, v):
        h, w, _ = face_img.shape
        x_idx = (u*(w-1)).clamp(0,w-1).long()
        y_idx = (v*(h-1)).clamp(0,h-1).long()
        return face_img[y_idx, x_idx]  # tensor 索引

    # +X
    u[mask_posx] = z[mask_posx] / abs_x[mask_posx] / 2 + 0.5
    v[mask_posx] = -y[mask_posx] / abs_x[mask_posx] / 2 + 0.5
    out[mask_posx] = sample(cube_imgs['+X'], u[mask_posx], v[mask_posx])

    # -X
    u[mask_negx] = -z[mask_negx] / abs_x[mask_negx] / 2 + 0.5
    v[mask_negx] = -y[mask_negx] / abs_x[mask_negx] / 2 + 0.5
    out[mask_negx] = sample(cube_imgs['-X'], u[mask_negx], v[mask_negx])

    # +Y
    u[mask_posy] = x[mask_posy] / abs_y[mask_posy] / 2 + 0.5
    v[mask_posy] = -z[mask_posy] / abs_y[mask_posy] / 2 + 0.5
    out[mask_posy] = sample(cube_imgs['+Y'], u[mask_posy], v[mask_posy])

    # -Y
    u[mask_negy] = x[mask_negy] / abs_y[mask_negy] / 2 + 0.5
    v[mask_negy] = z[mask_negy] / abs_y[mask_negy] / 2 + 0.5
    out[mask_negy] = sample(cube_imgs['-Y'], u[mask_negy], v[mask_negy])

    # +Z
    u[mask_posz] = -x[mask_posz] / abs_z[mask_posz] / 2 + 0.5
    v[mask_posz] = -y[mask_posz] / abs_z[mask_posz] / 2 + 0.5
    out[mask_posz] = sample(cube_imgs['+Z'], u[mask_posz], v[mask_posz])

    # -Z
    u[mask_negz] = x[mask_negz] / abs_z[mask_negz] / 2 + 0.5
    v[mask_negz] = -y[mask_negz] / abs_z[mask_negz] / 2 + 0.5
    out[mask_negz] = sample(cube_imgs['-Z'], u[mask_negz], v[mask_negz])

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from utils import plt_show
    # images = np.clip(images.cpu().numpy(), 0.0,1.0)
    # plt_show([images])

    # out = np.clip(out.cpu().numpy(), 0.0,1.0)
    # # plt.imshow((out/out.max())**(1/2.2))
    # # plt.imshow((out/out.max()))
    # plt.imshow(out)
    # plt.axis('off')
    # plt.show()
    # exit()
    out = utils.srgb_to_linear(out)
    return out

def gs_env_render2(device, gs: Gaussians, origin=[0, 0, 0], out_h=512, out_w=1024, bg_color=[1.0,1.0,1.0]):
    if not isinstance(bg_color, torch.Tensor):
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Making Camera
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin, dtype=torch.float32, device=device)

    view_dir = torch.tensor([1,0,0], dtype=torch.float32, device=device)
    #! Using Opencv camera coordinate system here
    up = torch.tensor([0,-1,0], dtype=torch.float32, device=device)
    view_matrix = utils.create_view_matrix(
        position=origin,
        look_at =origin+view_dir,
        up = up,
        device=device
    ).unsqueeze(0)

    # GS
    xyz = gs.xyz
    scale = torch.exp(gs.scale)
    rot = gs.rot
    opacity =  torch.sigmoid(gs.opacity)
    color = gs.color

    rgb_panorama = panorama_rasterization(
        means=xyz,
        quats=rot,
        scales=scale,
        opacities=opacity,
        colors=color,
        sh_degree=3,
        
        viewmats=view_matrix,
        height=out_h,
        width=out_w,
        
        render_mode="RGB",
        backgrounds=bg_color,
    ).squeeze(0) # [H, W, 3]
    out = utils.srgb_to_linear(rgb_panorama)
    return out

def add_sun_to_env_map(env_img: torch.Tensor, 
                       sun_direction: torch.Tensor, 
                       intensity: float = 100.0, 
                       radius_deg: float = 2.0,
                       softness: float = 0.5,
                       sun_color: torch.Tensor = None):
    """
    向全景环境图(Equirectangular)中注入一个高能太阳。
    
    Args:
        env_img (Tensor): [H, W, 3] 线性空间的RGB环境图
        sun_direction (Tensor): [3] 指向太阳的方向向量 (会被自动归一化)
        intensity (float): 太阳核心亮度 (建议 50.0 - 500.0)
        radius_deg (float): 太阳圆盘的半径（角度制）
        softness (float): 边缘柔化程度 [0.0 - 1.0]。0.0=硬边缘，1.0=大光晕。
        sun_color (Tensor): [3] 太阳颜色，默认为白色 [1., 1., 1.]
        
    Returns:
        Tensor: [H, W, 3] 添加了太阳的新环境图
    """
    H, W, C = env_img.shape
    device = env_img.device
    
    sun_dir = F.normalize(sun_direction.to(device), dim=0)
    
    if sun_color is None:
        sun_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    else:
        sun_color = sun_color.to(device)

    # 2. 构建全景图的像素方向向量 (World Space Vectors)
    # 这一步与之前的 spherical_uv_to_cartesian 逻辑一致
    # Grid: U [0, 1], V [0, 1]
    u = torch.linspace(0, 1, W, device=device)
    v = torch.linspace(0, 1, H, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij') # [H, W]

    # UV to Spherical (Theta, Phi)
    # 注意：这里假设标准的 Y-Up 坐标系
    # Theta: 0 (Top/Y+) -> PI (Bottom/Y-)
    # Phi: -PI -> +PI (绕Y轴旋转)
    theta = v_grid * math.pi
    phi = (u_grid - 0.5) * 2 * math.pi

    # Spherical to Cartesian (x, y, z)
    # y = cos(theta)
    # x = sin(theta) * cos(phi)
    # z = sin(theta) * sin(phi)
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = torch.cos(theta)
    z = sin_theta * torch.sin(phi)
    
    # 组合成 [H, W, 3] 的方向张量
    pixel_dirs = torch.stack([x, y, z], dim=-1)

    # 3. 计算每个像素与太阳方向的夹角余弦
    # dot(A, B) = cos(angle)
    dot_prod = torch.sum(pixel_dirs * sun_dir.view(1, 1, 3), dim=-1) # [H, W]
    
    # 4. 计算遮罩 (基于角度)
    # 将角度限制在 [0, pi]
    pixel_angles = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    
    # 将角度转换为弧度阈值
    rad_radius = math.radians(radius_deg)
    
    # 5. 生成太阳亮度分布 (Mask)
    # 使用 smoothstep 或简单的线性插值来做边缘柔化
    # mask = 1.0 when angle < radius
    # mask fades out as angle approaches radius * (1 + softness)
    
    rad_outer = rad_radius * (1.0 + softness)
    
    # 简单的线性衰减: (outer - angle) / (outer - inner)
    mask = (rad_outer - pixel_angles) / (rad_outer - rad_radius + 1e-6)
    mask = torch.clamp(mask, 0.0, 1.0)
    
    # 如果想要更像光源的衰减，可以对 mask 进行平方或指数处理
    # mask = mask ** 2 
    
    # 6. 叠加太阳
    # Additive Blending (物理上光是叠加的)
    # Solar Radiance = Mask * Intensity * Color
    sun_radiance = mask.unsqueeze(-1) * intensity * sun_color.view(1, 1, 3)
    
    final_env = env_img + sun_radiance
    
    return final_env

def apply_shadow_smoothing(shadow_mask, kernel_size=3):
    """
    对硬阴影遮罩进行平滑处理 (模拟 PCF)。
    
    Args:
        shadow_mask: (N, H, W) 或 (N, H, W, 1) 的 0/1 张量
        kernel_size: 模糊核大小 (奇数, 如 3, 5, 7)
    
    Returns:
        smoothed_mask: 形状同输入，值在 0.0 ~ 1.0 之间
    """
    import torch.nn.functional as F
    
    # 1. 维度处理: 确保是 (N, C, H, W) 格式以便卷积
    original_shape = shadow_mask.shape
    if shadow_mask.dim() == 3: # (N, H, W)
        input_tensor = shadow_mask.unsqueeze(1) # -> (N, 1, H, W)
    elif shadow_mask.dim() == 4 and shadow_mask.shape[-1] == 1: # (N, H, W, 1)
        input_tensor = shadow_mask.permute(0, 3, 1, 2) # -> (N, 1, H, W)
    else:
        raise ValueError(f"Unsupported shape: {shadow_mask.shape}")

    # 2. 边缘填充 (Padding) 保持尺寸不变
    padding = kernel_size // 2
    
    # 3. 使用 AvgPool2d 进行快速均值模糊 (Stride=1)
    # 这比手动写 conv2d 更快且不用构建权重
    smoothed = F.avg_pool2d(
        input_tensor, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding,
        count_include_pad=False # 边缘不计算 padding 的 0，防止边缘变黑
    )
    
    # 4. 恢复原始维度
    if shadow_mask.dim() == 3:
        return smoothed.squeeze(1)
    else:
        return smoothed.permute(0, 2, 3, 1)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_m'] = torch.ones_like(rays_o[..., 0])

    return results

def refract(l, normal, eta1, eta2):
    # l ... x 3 
    # normal ... x 3
    # eta1 float
    # eta2 float
    #l = -l
    cos_theta = torch.sum(l * (-normal), dim=-1).unsqueeze(-1)
    i_p = l + normal * cos_theta
    t_p = eta1 / eta2 * i_p

    t_p_norm = torch.sum(t_p * t_p, dim=-1)
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(-1)

    t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(-1).expand_as(normal) * (-normal)
    t = t_i + t_p
    t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=-1), min=1e-10)).unsqueeze(-1)

    cos_theta_t = torch.sum(t * (-normal), dim=-1).unsqueeze(-1)

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
            torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
            torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()
    #print("attenuate[1280800]:", attenuate[1280800])

    return t, attenuate, totalReflectMask