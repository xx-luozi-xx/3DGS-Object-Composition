import torch
import torch.nn.functional as F

import env_diff
import env_sp3g as env_sp
import lut2

def IBL_transparent2(env_d, mips, G, T):
    num_mips = len(mips)
    if not hasattr(IBL_transparent2, "_lut"):
        print("[IBL] loading lut...", end='')
        # lut2 是你的模块
        IBL_transparent2._lut = lut2.lut_loader(path="./lut.png", device="cuda")
        print("\r[IBL] loading lut[finished]")
    lut = IBL_transparent2._lut
    
    N, H, W = G["mask"].shape
    mask = G["mask"].unsqueeze(-1)
    Nn = F.normalize(G["normal"], dim=-1)
    V  = F.normalize(G["view"], dim=-1)
    R  = reflect_vec(-V, Nn)              # 自定义函数
    refract_dir = F.normalize(G["refract_dir"], dim=-1)

    albedo = T["albedo"].clamp(0.0, 1.0)
    
    metallic = torch.zeros_like(T["metallic"]).unsqueeze(-1)
    roughness = torch.full_like(T["roughness"], 0.04).unsqueeze(-1)

    cos_theta = torch.sum(Nn * V, dim=-1, keepdim=True).clamp(0.0, 1.0)
    F0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)

    uv_refract = direction_to_uv(refract_dir).clamp(0.0, 1.0)
    grid_refract = uv_refract * 2.0 - 1.0
    
    refract_light_tensor = mips[0].permute(2, 0, 1).contiguous().unsqueeze(0).expand(N, -1, -1, -1)
    refract_light = F.grid_sample(
        refract_light_tensor, grid_refract,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1)

    uv_n = direction_to_uv(Nn).clamp(0.0, 1.0)
    grid_n = uv_n * 2.0 - 1.0
    
    env_d_tensor = env_d.permute(2, 0, 1).contiguous().unsqueeze(0).expand(N, -1, -1, -1)
    env_diffuse = F.grid_sample(
        env_d_tensor, grid_n,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1)

    uv_r = direction_to_uv(R).clamp(0.0, 1.0)
    grid_r = uv_r * 2.0 - 1.0
    
    mip_level = roughness * (num_mips - 1)
    floor_mip = torch.floor(mip_level).long()
    ceil_mip  = torch.ceil(mip_level).long()
    mip_weight = mip_level - floor_mip.float()
    
    specular_color = torch.zeros((N, H, W, 3), device=env_d.device)
    
    for i in range(num_mips):
        is_floor = (floor_mip == i)
        is_ceil  = (ceil_mip == i)
        
        if not (is_floor.any() or is_ceil.any()):
            continue
            
        m_tensor = mips[i].permute(2, 0, 1).contiguous().unsqueeze(0).expand(N, -1, -1, -1)
        sampled_m = F.grid_sample(
            m_tensor, grid_r, 
            mode="bilinear", padding_mode="border", align_corners=True
        ).permute(0, 2, 3, 1)
        
        w = torch.zeros_like(mip_weight)
        w = torch.where(is_floor, 1.0 - mip_weight, w)
        w = torch.where(is_ceil, w + mip_weight, w)
        
        if i == 0: 
            w = torch.where(is_floor & is_ceil, torch.ones_like(w), w)
             
        specular_color += sampled_m * w

    grid_lut = torch.cat([cos_theta * 2.0 - 1.0, roughness * 2.0 - 1.0], dim=-1)
    lut_tensor = lut.contiguous().unsqueeze(0).expand(N, -1, -1, -1) 
    
    brdf_res = F.grid_sample(
        lut_tensor, grid_lut,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1)
    
    env_brdf_scale = brdf_res[..., 0:1] 
    env_brdf_bias  = brdf_res[..., 1:2] 

    Ks = G["attenuate"] 
    Kd = (1.0 - Ks) * (1.0 - metallic)
    
    specular = specular_color * (F0 * env_brdf_scale + env_brdf_bias)
    refract = Kd * refract_light

    color = (specular + refract) * mask

    return color

def IBL(env_d, mips, G, T):
    """
    env_d: #  (H, W, 3) 漫反射分量
    mips: #  list of (H, W, 3) 镜面反射Part A
    G: G-buffer dict
        dict: {
            "mask":         # (N,H,W)
            "depth":        # (N,H,W)
            "position":     # (N,H,W,3)
            "normal":       # (N,H,W,3)
            "view":         # (N,H,W,3)
        }   
    T: Texture dict
        dict: {
            "albedo":           # (N,H,W,3)
            "metallic":         # (N,H,W)
            "roughness":        # (N,H,W)
            "specular_strength":# (N,H,W)
        }   
    """    
    num_mips = len(mips)
    
    # 镜面反射 Part B: BRDF LUT
    # lut 形状为 (3, 512, 512) [C, H, W]
    if not hasattr(IBL, "_lut"):
        print("[IBL] loading lut...", end='')
        # lut2 是你的模块
        IBL._lut = lut2.lut_loader(path="./lut.png", device="cuda")
        print("\r[IBL] loading lut[finished]")
    lut = IBL._lut
    
    
    # 2. 准备基础向量与参数
    N, H, W = G["mask"].shape
    mask = G["mask"].unsqueeze(-1)
    Nn = F.normalize(G["normal"], dim=-1) # (N, H, W, 3)
    V  = F.normalize(G["view"], dim=-1)   # (N, H, W, 3)
    R  = reflect_vec(-V, Nn)              # (N, H, W, 3)
    
    albedo = T["albedo"].clamp(0.0, 1.0)
    metallic = T["metallic"].clamp(0.0, 1.0).unsqueeze(-1)
    roughness = T["roughness"].clamp(0.04, 1.0).unsqueeze(-1)
    
    # cos_theta (N.V) 用于 LUT 采样和 Fresnel
    cos_theta = torch.sum(Nn * V, dim=-1, keepdim=True).clamp(0.0, 1.0)
    
    # F0 计算
    F0_dielectric = 0.04
    F0 = torch.lerp(torch.full_like(albedo, F0_dielectric), albedo, metallic)

    # --------------------------------------------------
    # 3. 漫反射部分 (Diffuse IBL)
    # --------------------------------------------------
    uv_n = direction_to_uv(Nn).clamp(0.0, 1.0)
    grid_n = uv_n * 2.0 - 1.0
    env_d_tensor = env_d.permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
    
    env_diffuse = F.grid_sample(
        env_d_tensor, grid_n,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1) # (N, H, W, 3)

    # --------------------------------------------------
    # 4. 镜面反射 Part A: 采样预滤波贴图 (Prefiltered Env Map)
    # --------------------------------------------------
    uv_r = direction_to_uv(R).clamp(0.0, 1.0)
    grid_r = uv_r * 2.0 - 1.0
    
    # 根据 roughness 计算 mip 层级 (假设 5 级: 0, 1, 2, 3, 4)
    mip_level = roughness * (num_mips - 1)
    
    # 线性插值采样两个相邻的 Mip 层级 (Trilinear filtering 模拟)
    floor_mip = torch.floor(mip_level).long()
    ceil_mip  = torch.ceil(mip_level).long()
    mip_weight = mip_level - floor_mip.float()
    
    # 这里的实现为了效率，假设 mips 分辨率一致或使用 grid_sample 自动处理
    # 采样 floor 层
    specular_color = torch.zeros((N, H, W, 3), device=env_d.device)
    
    # 简单的线性插值采样逻辑
    for i in range(num_mips):
        # 这种方式虽然略慢，但能处理不同层级的采样
        # 实际生产环境通常会将 mips 拼成一个 TextureArray
        m_tensor = mips[i].permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
        sampled_m = F.grid_sample(
            m_tensor, grid_r, 
            mode="bilinear", padding_mode="border", align_corners=True
        ).permute(0, 2, 3, 1)
        
        # 权重计算：只有当 i 是 floor 或 ceil 时才贡献
        w = torch.where(floor_mip == i, 1.0 - mip_weight, torch.zeros_like(mip_weight))
        w = torch.where(ceil_mip == i, w + mip_weight, w)
        # 处理 floor == ceil 的情况
        if i == 0: # 特殊处理起始层
             w = torch.where((floor_mip == i) & (ceil_mip == i), torch.ones_like(w), w)
             
        specular_color += sampled_m * w

    # --------------------------------------------------
    # 5. 镜面反射 Part B: 采样 BRDF LUT
    # --------------------------------------------------
    # LUT 输入: x = N.V (cos_theta), y = Roughness
    # grid_sample 需要 [-1, 1] 空间
    lut_u = cos_theta * 2.0 - 1.0
    lut_v = roughness * 2.0 - 1.0
    grid_lut = torch.cat([lut_u, lut_v], dim=-1) # (N, H, W, 2)
    
    lut_tensor = lut.unsqueeze(0).expand(N, -1, -1, -1) # (N, 3, 512, 512)
    brdf_res = F.grid_sample(
        lut_tensor, grid_lut,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1) # (N, H, W, 3)
    
    env_brdf_scale = brdf_res[..., 0:1] # R 通道
    env_brdf_bias  = brdf_res[..., 1:2] # G 通道

    # --------------------------------------------------
    # 6. 能量守恒与最终合成
    # --------------------------------------------------
    # 计算带有粗糙度修正的 Fresnel (用于决定 Kd)
    def fresnel_schlick_roughness(cos_theta, F0, roughness):
        return F0 + (torch.maximum(1.0 - roughness, F0) - F0) * torch.pow(1.0 - cos_theta, 5.0)
    
    F_ = fresnel_schlick_roughness(cos_theta, F0, roughness)
    
    Ks = F_
    Kd = (1.0 - Ks) * (1.0 - metallic)
    
    # 最终镜面反射合成: Prefiltered * (F0 * scale + bias)
    specular = specular_color * (F0 * env_brdf_scale + env_brdf_bias)
    
    # 最终漫反射
    diffuse = Kd * env_diffuse * albedo
    
    # 合并结果
    color = (diffuse + specular) * mask
    
    return color

def IBL_test3(env_map, G, T):
    """
    env_map: #  (H, W, 3)
    G: G-buffer dict
        dict: {
            "mask":         # (N,H,W)
            "depth":        # (N,H,W)
            "position":     # (N,H,W,3)
            "normal":       # (N,H,W,3)
            "view":         # (N,H,W,3)
        }   
    T: Texture dict
        dict: {
            "albedo":           # (N,H,W,3)
            "metallic":         # (N,H,W)
            "roughness":        # (N,H,W)
            "specular_strength":# (N,H,W)
        }   
    """    
    # 1. 加载预计算资源
    # 漫反射部分
    env_d = env_diff.compute_diffuse_panorama(env_map) # (H, W, 3)
    
    # 镜面反射 Part A: Prefiltered Mips
    # mips 是一个 list, 每个元素为 (H, W, 3)
    mips = env_sp.prefilter_env_map(env_map, num_samples=1024, max_mips=5)
    num_mips = len(mips)
    
    # 镜面反射 Part B: BRDF LUT
    # lut 形状为 (3, 512, 512) [C, H, W]
    lut = lut2.lut_loader(path="./lut.png", device="cuda") 
    
    # 2. 准备基础向量与参数
    N, H, W = G["mask"].shape
    mask = G["mask"].unsqueeze(-1)
    Nn = F.normalize(G["normal"], dim=-1) # (N, H, W, 3)
    V  = F.normalize(G["view"], dim=-1)   # (N, H, W, 3)
    R  = reflect_vec(-V, Nn)              # (N, H, W, 3)
    
    albedo = T["albedo"].clamp(0.0, 1.0)
    metallic = T["metallic"].clamp(0.0, 1.0).unsqueeze(-1)
    roughness = T["roughness"].clamp(0.04, 1.0).unsqueeze(-1)
    
    # cos_theta (N.V) 用于 LUT 采样和 Fresnel
    cos_theta = torch.sum(Nn * V, dim=-1, keepdim=True).clamp(0.0, 1.0)
    
    # F0 计算
    F0_dielectric = 0.04
    F0 = torch.lerp(torch.full_like(albedo, F0_dielectric), albedo, metallic)

    # --------------------------------------------------
    # 3. 漫反射部分 (Diffuse IBL)
    # --------------------------------------------------
    uv_n = direction_to_uv(Nn).clamp(0.0, 1.0)
    grid_n = uv_n * 2.0 - 1.0
    env_d_tensor = env_d.permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
    
    env_diffuse = F.grid_sample(
        env_d_tensor, grid_n,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1) # (N, H, W, 3)

    # --------------------------------------------------
    # 4. 镜面反射 Part A: 采样预滤波贴图 (Prefiltered Env Map)
    # --------------------------------------------------
    uv_r = direction_to_uv(R).clamp(0.0, 1.0)
    grid_r = uv_r * 2.0 - 1.0
    
    # 根据 roughness 计算 mip 层级 (假设 5 级: 0, 1, 2, 3, 4)
    mip_level = roughness * (num_mips - 1)
    # 线性插值采样两个相邻的 Mip 层级 (Trilinear filtering 模拟)
    floor_mip = torch.floor(mip_level).long()
    ceil_mip  = torch.ceil(mip_level).long()
    mip_weight = mip_level - floor_mip.float()
    
    # 这里的实现为了效率，假设 mips 分辨率一致或使用 grid_sample 自动处理
    # 采样 floor 层
    specular_color = torch.zeros((N, H, W, 3), device=env_map.device)
    
    # 简单的线性插值采样逻辑
    for i in range(num_mips):
        # 这种方式虽然略慢，但能处理不同层级的采样
        # 实际生产环境通常会将 mips 拼成一个 TextureArray
        m_tensor = mips[i].permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
        sampled_m = F.grid_sample(
            m_tensor, grid_r, 
            mode="bilinear", padding_mode="border", align_corners=True
        ).permute(0, 2, 3, 1)
        
        # 权重计算：只有当 i 是 floor 或 ceil 时才贡献
        w = torch.where(floor_mip == i, 1.0 - mip_weight, torch.zeros_like(mip_weight))
        w = torch.where(ceil_mip == i, w + mip_weight, w)
        # 处理 floor == ceil 的情况
        if i == 0: # 特殊处理起始层
             w = torch.where((floor_mip == i) & (ceil_mip == i), torch.ones_like(w), w)
             
        specular_color += sampled_m * w

    # --------------------------------------------------
    # 5. 镜面反射 Part B: 采样 BRDF LUT
    # --------------------------------------------------
    # LUT 输入: x = N.V (cos_theta), y = Roughness
    # grid_sample 需要 [-1, 1] 空间
    lut_u = cos_theta * 2.0 - 1.0
    lut_v = roughness * 2.0 - 1.0
    grid_lut = torch.cat([lut_u, lut_v], dim=-1) # (N, H, W, 2)
    
    lut_tensor = lut.unsqueeze(0).expand(N, -1, -1, -1) # (N, 3, 512, 512)
    brdf_res = F.grid_sample(
        lut_tensor, grid_lut,
        mode="bilinear", padding_mode="border", align_corners=True
    ).permute(0, 2, 3, 1) # (N, H, W, 3)
    
    env_brdf_scale = brdf_res[..., 0:1] # R 通道
    env_brdf_bias  = brdf_res[..., 1:2] # G 通道

    # --------------------------------------------------
    # 6. 能量守恒与最终合成
    # --------------------------------------------------
    # 计算带有粗糙度修正的 Fresnel (用于决定 Kd)
    def fresnel_schlick_roughness(cos_theta, F0, roughness):
        return F0 + (torch.maximum(1.0 - roughness, F0) - F0) * torch.pow(1.0 - cos_theta, 5.0)
    
    F_ = fresnel_schlick_roughness(cos_theta, F0, roughness)
    
    Ks = F_
    Kd = (1.0 - Ks) * (1.0 - metallic)
    
    # 最终镜面反射合成: Prefiltered * (F0 * scale + bias)
    specular = specular_color * (F0 * env_brdf_scale + env_brdf_bias)
    
    # 最终漫反射
    diffuse = Kd * env_diffuse * albedo
    
    # 合并结果
    color = (diffuse + specular) * mask
    
    return color

def reflect_vec(I, N):
    """
    Reflect incoming vector I about normal N.
    I, N: (...,3). reflect = I - 2*(I·N)*N
    We use I = -V above typically, but this function is general.
    """
    dot = (I * N).sum(dim=-1, keepdim=True)
    return I - 2.0 * dot * N

def direction_to_uv(dir_xyz):
    """
    dir_xyz: (...,3) unit direction in world space.
    returns (...,2) u,v in [0,1] for equirectangular env map.
    phi = atan2(z, x), theta = acos(y)
    u = phi/(2pi) + 0.5, v = theta/pi
    """
    x, y, z = dir_xyz.unbind(-1)
    
    # ==== 1. 修复 atan2 梯度爆炸 ====
    eps = 1e-6
    zero_mask = (x.abs() < eps) & (z.abs() < eps)
    x = torch.where(zero_mask, x + eps, x)
    
    phi = torch.atan2(z, x)               # [-pi, pi]
    
    # ==== 2. 修复 acos 梯度爆炸 ====
    # 绝不能用 (-1.0, 1.0)，必须留有余地！
    safe_y = y.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(safe_y)            # [0, pi]
    
    u = (phi / (2.0 * torch.pi) + 0.5)
    v = theta / torch.pi
    return torch.stack([u, v], dim=-1)
