import torch
import torch.nn.functional as F
import math

def radical_inverse_vdc(bits: torch.Tensor):
    """ 计算 Van der Corput 序列 """
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.float() * 2.3283064365386963e-10

def hammersley(i: int, N: int, device):
    """ 生成第 i 个 Hammersley 采样点 """
    i_tensor = torch.tensor([i], dtype=torch.int32, device=device)
    u = i_tensor.float() / float(N)
    v = radical_inverse_vdc(i_tensor)
    return torch.stack([u, v], dim=-1)

def spherical_to_uv(dirs):
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    
    phi = torch.atan2(z, x)
    u = (phi / (2.0 * math.pi)) + 0.5
    
    theta = torch.asin(torch.clamp(y, -1.0, 1.0))
    v = 0.5 - (theta / math.pi)
    
    return torch.stack([u, v], dim=-1)

def generate_mips(env_map, max_mips):
    mips = [env_map.permute(2, 0, 1).unsqueeze(0)] # [1, 3, H, W]
    current = mips[0]
    
    for i in range(1, max_mips):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        mips.append(current)
    return mips

def sample_env_lod(mips, uv, lod):
    max_lod = len(mips) - 1.0
    lod = torch.clamp(lod, 0.0, max_lod)
    
    grid = (uv - 0.5) * 2.0
    batch_grid = grid.unsqueeze(0) 
    
    final_color = torch.zeros(uv.shape[0], uv.shape[1], 3, device=uv.device)
    
    for i in range(len(mips)):
        weight = torch.clamp(1.0 - torch.abs(lod - float(i)), 0.0, 1.0)
        
        if weight.sum() > 0:
            # 使用 reflection 模式减少边缘缝隙
            sampled = F.grid_sample(mips[i], batch_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1).squeeze(0) 
            final_color += sampled * weight
            
    return final_color

def gaussian_blur_spatial(img_tensor, kernel_size=3, sigma=1.0):
    """
    对 Equirectangular 图像应用高斯模糊，处理左右循环填充和上下边缘复制。
    img_tensor: [H, W, 3]
    """
    device = img_tensor.device
    channels = img_tensor.shape[-1]
    
    # 1. 生成高斯核
    k = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(- (k**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 扩展为 2D 卷积核 (Separable logic or direct 2D)
    # 这里为了简单直接构建 2D 核
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0) # [K, K]
    
    # 调整形状以适应 conv2d: [Out, In, H, W] -> [3, 1, K, K] (Depthwise convolution)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    
    # 2. 准备图像 [1, 3, H, W]
    x = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # 3. 填充 (Padding)
    pad_size = kernel_size // 2
    
    # 左右: Circular (地球经度是循环的)
    x = F.pad(x, (pad_size, pad_size, 0, 0), mode='circular')
    # 上下: Replicate (两极不能循环，而是延伸)
    x = F.pad(x, (0, 0, pad_size, pad_size), mode='replicate')
    
    # 4. 卷积
    # groups=3 意味着每个通道独立卷积，互不干扰
    out = F.conv2d(x, kernel, groups=channels)
    
    # 5. 还原形状 [H, W, 3]
    return out.squeeze(0).permute(1, 2, 0)

def prefilter_env_map(env_map, num_samples=1024, max_mips=5):
    device = env_map.device
    H_img, W_img, C = env_map.shape
    
    # --- 步骤 0: 生成 Mipmap 链 ---
    total_mips_gen = max_mips + 3 
    env_mips = generate_mips(env_map, total_mips_gen)
    
    omega_p = 4.0 * math.pi / (6.0 * W_img * H_img) 

    # --- 步骤 1: 构建观察空间坐标系 (N_all) ---
    u = torch.linspace(0, 1, W_img, device=device)
    v = torch.linspace(0, 1, H_img, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    
    phi = (u_grid - 0.5) * 2 * math.pi
    theta = (0.5 - v_grid) * math.pi
    
    x = torch.cos(theta) * torch.cos(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.sin(phi)
    N_all = torch.stack([x, y, z], dim=-1)
    N_all = F.normalize(N_all, dim=-1)
    
    results = []
    
    print(f"Starting Prefilter (Samples: {num_samples}, Jitter: ON, LOD: ON, Blur: ON)...")

    for mip in range(max_mips):
        roughness = float(mip) / float(max_mips - 1)
        roughness = max(roughness, 0.01)
        
        print(f"  Processing Level {mip} | Roughness: {roughness:.3f}")
        
        prefiltered_color = torch.zeros_like(N_all)
        total_weight = torch.zeros((H_img, W_img, 1), device=device)
        
        rand_rot = torch.rand((H_img, W_img, 1), device=device) * 2.0 * math.pi
        
        # --- 蒙特卡洛积分循环 ---
        for i in range(num_samples):
            Xi_val = hammersley(i, num_samples, device)
            Xi = Xi_val.expand(H_img, W_img, 2)
            
            a = roughness * roughness
            a2 = a * a
            
            phi_h = (2.0 * math.pi * Xi[..., 0] + rand_rot[..., 0])
            
            cos_theta_h = torch.sqrt((1.0 - Xi[..., 1]) / (1.0 + (a2 - 1.0) * Xi[..., 1]))
            sin_theta_h = torch.sqrt(1.0 - cos_theta_h * cos_theta_h)
            
            H_x = sin_theta_h * torch.cos(phi_h)
            H_y = sin_theta_h * torch.sin(phi_h)
            H_z = cos_theta_h
            
            up = torch.abs(N_all[..., 2]) < 0.999
            up_vec = torch.zeros_like(N_all)
            up_vec[..., 0] = up.float()
            up_vec[..., 2] = (~up).float()
            
            tangent = F.normalize(torch.cross(up_vec, N_all, dim=-1), dim=-1)
            bitangent = torch.cross(N_all, tangent, dim=-1)
            
            H_vec = tangent * H_x.unsqueeze(-1) + bitangent * H_y.unsqueeze(-1) + N_all * H_z.unsqueeze(-1)
            H_vec = F.normalize(H_vec, dim=-1)
            
            VoH = torch.sum(N_all * H_vec, dim=-1, keepdim=True).clamp(0.0, 1.0)
            L_vec = 2.0 * VoH * H_vec - N_all
            L_vec = F.normalize(L_vec, dim=-1)
            
            NoL = torch.sum(N_all * L_vec, dim=-1, keepdim=True).clamp(0.0, 1.0)
            NoH = VoH
            
            mask = (NoL > 0).float()
            
            if mask.sum() > 0:
                denom = (NoH.squeeze(-1) ** 2) * (a2 - 1.0) + 1.0
                D = a2 / (math.pi * denom * denom)
                
                pdf = D / 4.0 + 0.0001
                omega_s = 1.0 / (float(num_samples) * pdf)
                
                # 保持原有的 LOD 计算
                lod = 0.5 * torch.log2(omega_s / omega_p) + 1.0
                
                uv = spherical_to_uv(L_vec)
                env_sample = sample_env_lod(env_mips, uv, lod.unsqueeze(-1))
                
                prefiltered_color += env_sample * NoL * mask
                total_weight += NoL * mask
                
        # 归一化
        prefiltered_color = prefiltered_color / (total_weight + 1e-6)
        
        # --- [NEW] Post-Process: Gaussian Blur ---
        # 只有当 Roughness > 0 (即 mip > 0) 时才进行平滑
        # Mip 0 通常是完美的镜面反射，不应该被模糊
        if mip > 0:
            # 这里的 kernel_size 和 sigma 可以根据需要微调
            # 3x3, sigma=0.8 是一个温和的降噪参数
            prefiltered_color = gaussian_blur_spatial(prefiltered_color, kernel_size=5, sigma=0.8)
            
        results.append(prefiltered_color)
        
    return results

