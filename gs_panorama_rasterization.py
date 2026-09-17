import math
from typing import Dict, Optional, Literal

import torch
from torch import Tensor
import torch.nn.functional as F

import diff_gaussian_rasterization_panorama

from gsplat.cuda._wrapper import (
    spherical_harmonics,
)

def panorama_rasterization(
    means: Tensor,      # [N, 3]    高斯中心
    quats: Tensor,      # [N, 4]    旋转四元数
    scales: Tensor,     # [N, 3]    缩放 (exp后)
    opacities: Tensor,  # [N] 不透明度 (sigmoid后)
    colors: Tensor,     # [N, K, 3] SH 特征 (暂不考虑多外观特征 [C, N, K, 3])
    sh_degree: int,

    viewmats: Tensor,   # [C, 4, 4] World-to-Camera 矩阵 (OpenCV 格式)
    width: int,
    height: int,

    scaling_modifier: float = 1.0,
    backgrounds: Optional[Tensor] = None, # [C, 3] 或 [3]
    render_mode: Literal['RGB', "D"] = 'RGB',
):
    """
    return    
    **render_colors**: The rendered colors. [C, height, width, X]. Or depth [C, height, width, 1]
    """
    device = means.device
    N = means.shape[0]
    C = viewmats.shape[0]   # 相机数量
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert render_mode in ["RGB", "D"], render_mode

    opacities = opacities.unsqueeze(-1)
        
    quats = F.normalize(quats, dim=-1)

    if colors.dim() == 4:
        colors = colors[0] 
    num_sh_bases = (sh_degree + 1) ** 2    
    assert num_sh_bases <= colors.shape[-2], f"Provided SH features ({colors.shape[-2]}) are less than required ({num_sh_bases})"
    shs_view = colors[:, :num_sh_bases, :].contiguous() 
    
    rendered_images = []
    for i in range(C):
        viewmat = viewmats[i] # [4, 4], OpenCV 格式的 W2C
        
        world_view_transform = viewmat.transpose(0, 1).contiguous()
        
        c2w = torch.inverse(viewmat)
        campos = c2w[:3, 3].contiguous()
        
        fov = math.pi / 2
        tan_half_fov = math.tan(fov * 0.5)
        znear, zfar = 0.01, 100.0
        
        proj_matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1.0 / tan_half_fov
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        proj_matrix[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        proj_matrix[3, 2] = 1.0
        proj_matrix = proj_matrix.transpose(0, 1).contiguous() # 同样需转置
        
        full_proj_transform = (world_view_transform @ proj_matrix).contiguous()
        
        if backgrounds is not None:
            bg_color = backgrounds[i] if backgrounds.dim() == 2 else backgrounds
        else:
            bg_color = torch.zeros(3, dtype=torch.float32, device=device)
            
        raster_settings = diff_gaussian_rasterization_panorama.GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tan_half_fov,
            tanfovy=tan_half_fov,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False
        )
        
        rasterizer = diff_gaussian_rasterization_panorama.GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(means, requires_grad=True, device=device)
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        if render_mode == 'RGB':
            dir_pp = means - campos.unsqueeze(0)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            
            sh2rgb = spherical_harmonics(sh_degree, dir_pp_normalized, shs_view)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # [N, 3]
        else:
            depth = torch.norm(means - campos.unsqueeze(0), dim=-1, keepdim=True)
            colors_precomp = depth.expand(-1, 3) # 扩展为 [N, 3]

        rendered_image, radii = rasterizer(
            means3D=means,
            means2D=screenspace_points,
            shs=None, 
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None
        )
        
        if render_mode == 'RGB':
            rendered_images.append(rendered_image)       # [3, H, W]
        else:
            rendered_images.append(rendered_image[:1])   # [1, H, W]

    stacked_images = torch.stack(rendered_images, dim=0)  
    render_colors = stacked_images.permute(0, 2, 3, 1).contiguous()
    return render_colors