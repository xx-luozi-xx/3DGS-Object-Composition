import math
import os

import torch
import torchvision.utils as vutils
import torchvision.transforms as T
from PIL import Image

def radical_inverse_vdc(bits):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits.float() * 2.3283064365386963e-10

def hammersley(i, N, device):
    i_tensor = torch.arange(i, i+1, dtype=torch.int32, device=device)
    u = i_tensor.float() / float(N)
    v = radical_inverse_vdc(i_tensor)
    return torch.stack([u, v], dim=-1)

def geometry_schlick_ggx(NdotV, roughness):
    # k = (roughness * roughness) / 2.0
    a = roughness
    k = (a * a) / 2.0

    nom = NdotV
    denom = NdotV * (1.0 - k) + k
    return nom / denom

def geometry_smith(N, V, L, roughness):
    # float NdotV = max(dot(N, V), 0.0);
    # float NdotL = max(dot(N, L), 0.0);
    NdotV = torch.clamp(torch.sum(N * V, dim=-1), 0.0, 1.0)
    NdotL = torch.clamp(torch.sum(N * L, dim=-1), 0.0, 1.0)

    ggx2 = geometry_schlick_ggx(NdotV, roughness)
    ggx1 = geometry_schlick_ggx(NdotL, roughness)

    return ggx1 * ggx2

def integrate_brdf(size=512, num_samples=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating BRDF LUT (LearnOpenGL Match) on {device}...")

    steps = torch.arange(size, dtype=torch.float32, device=device)
    coords = (steps + 0.5) / float(size) # range: [0.5/512, 511.5/512]
    
    u = coords  # NdotV
    v = coords  # Roughness
    
    roughness_grid, nov_grid = torch.meshgrid(v, u, indexing='ij')

    A = torch.zeros_like(nov_grid)
    B = torch.zeros_like(nov_grid)

    V_x = torch.sqrt(torch.clamp(1.0 - nov_grid*nov_grid, min=0.0))
    V = torch.stack([V_x, torch.zeros_like(nov_grid), nov_grid], dim=-1)
    
    N = torch.tensor([0.0, 0.0, 1.0], device=device).expand(size, size, 3)

    for i in range(num_samples):
        Xi = hammersley(i, num_samples, device).view(1, 1, 2).expand(size, size, 2)
        
        # --- Importance Sample GGX ---
        #  float a = roughness * roughness;
        a = roughness_grid * roughness_grid
        
        phi = 2.0 * math.pi * Xi[..., 0]
        cos_theta = torch.sqrt((1.0 - Xi[..., 1]) / (1.0 + (a*a - 1.0) * Xi[..., 1]))
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
        
        H = torch.stack([
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            cos_theta
        ], dim=-1)
        
        VoH = torch.clamp(torch.sum(V * H, dim=-1), 0.0, 1.0)
        L = 2.0 * VoH.unsqueeze(-1) * H - V
        
        NdotL = torch.clamp(L[..., 2], 0.0, 1.0) # L.z
        NdotH = torch.clamp(H[..., 2], 0.0, 1.0) # H.z
        VoH = torch.clamp(torch.sum(V * H, dim=-1), 0.0, 1.0)
        
        # Mask
        mask = (NdotL > 0.0).float()
        
        if mask.sum() > 0:
            G = geometry_smith(N, V, L, roughness_grid)
            
            # G_Vis = (G * VoH) / (NdotH * NdotV)
            G_Vis = (G * VoH) / (NdotH * nov_grid + 1e-6)
            
            Fc = torch.pow(1.0 - VoH, 5.0)
            
            A += (1.0 - Fc) * G_Vis * mask
            B += Fc * G_Vis * mask

    A /= float(num_samples)
    B /= float(num_samples)
    
    lut = torch.stack([A, B, torch.zeros_like(A)], dim=-1)
    return lut

def lut_loader(path="lut.png", device="cuda"):
    """
    返回形状: [3, 512, 512] (RGB), 值域 [0.0, 1.0]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 LUT 文件: {path}，请先运行 integrate_brdf 生成它。")

    img = Image.open(path).convert('RGB')
    
    transform = T.ToTensor()
    lut_tensor = transform(img).to(device)
    
    print(f"LUT Loaded from {path}. Shape: {lut_tensor.shape}, Device: {lut_tensor.device}")
    
    return lut_tensor

if __name__ == "__main__":
    lut = integrate_brdf(size=512, num_samples=1024)
    
    vutils.save_image(lut.permute(2, 0, 1), "brdf_lut_logl.png")
    print("Saved brdf_lut_logl.png. This should now match LearnOpenGL exactly.")
