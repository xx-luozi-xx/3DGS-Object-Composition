import torch
import torch.nn.functional as F


def build_sat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cumsum(dim=-1).cumsum(dim=-2)

def query_sat(sat: torch.Tensor, y1, x1, y2, x2, H_L, W_L) -> torch.Tensor:
    N_batch, _, H_sat, W_sat = sat.shape

    y1c = y1.clamp(0, H_sat - 1).long()
    x1c = x1.clamp(0, W_sat - 1).long()
    y2c = y2.clamp(0, H_sat - 1).long()
    x2c = x2.clamp(0, W_sat - 1).long()

    H_img, W_img = y1c.shape[-2], y1c.shape[-1]
    b_idx = (
        torch.arange(N_batch, device=sat.device)
        .view(N_batch, 1, 1)
        .expand(-1, H_img, W_img)
    )

    A = sat[b_idx, 0, y2c, x2c]
    y1_sub = (y1c - 1).clamp(min=0)
    x1_sub = (x1c - 1).clamp(min=0)
    B = sat[b_idx, 0, y1_sub, x2c]
    C = sat[b_idx, 0, y2c,    x1_sub]
    D = sat[b_idx, 0, y1_sub, x1_sub]

    res = A - B - C + D

    y_border = (y1c == 0)
    x_border = (x1c == 0)
    res = torch.where(y_border & x_border, A,     res)
    res = torch.where(y_border & ~x_border, A - C, res)
    res = torch.where(~y_border & x_border, A - B, res)

    return res

def get_area(py, px, r, H_L, W_L):
    y1_c = (py - r).clamp(0, H_L - 1)
    y2_c = (py + r).clamp(0, H_L - 1)
    x1_c = (px - r).clamp(0, W_L - 1)
    x2_c = (px + r).clamp(0, W_L - 1)
    return ((y2_c - y1_c + 1) * (x2_c - x1_c + 1)).float().clamp(min=1.0)

def _unproject_opencv(depth_map: torch.Tensor, K: torch.Tensor, viewmat: torch.Tensor):
    N, H, W = depth_map.shape
    device = depth_map.device

    us = torch.arange(W, device=device, dtype=torch.float32)
    vs = torch.arange(H, device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing='ij')

    u = grid_u.unsqueeze(0).expand(N, -1, -1)
    v = grid_v.unsqueeze(0).expand(N, -1, -1)

    fx = K[:, 0, 0].view(N, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)

    Z = depth_map
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    cam_pts = torch.stack([X, Y, Z], dim=-1).view(N, H * W, 3)

    R = viewmat[:, :3, :3]
    t = viewmat[:, :3,  3]

    R_inv = R.transpose(1, 2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    world_pts = cam_pts @ R_inv.transpose(1, 2) + t_inv.unsqueeze(1)
    return world_pts

def _project_opencv(world_pts: torch.Tensor, K: torch.Tensor, viewmat: torch.Tensor):
    N = world_pts.shape[0]

    R = viewmat[:, :3, :3]
    t = viewmat[:, :3,  3]

    cam_pts = world_pts @ R.transpose(1, 2) + t.unsqueeze(1)
    depth = cam_pts[..., 2]

    eps = 1e-6
    Z_safe = depth.clamp(min=eps)
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    px_f = fx * cam_pts[..., 0] / Z_safe + cx
    py_f = fy * cam_pts[..., 1] / Z_safe + cy

    return px_f, py_f, depth

def compute_VSSM(
    device,
    camera_depth_map,
    main_camera,
    shadow_map,
    light_camera,
    light_size=0.1,                 
    blocker_search_radius=6,        
    max_pcf_radius=16,              
    shadow_bias=0.01,               
    normal_bias_scale=0.02,
    light_bleeding_reduction=0.2,   
    camera_normal=None
):
    N, H, W = camera_depth_map.shape

    # ------------------------------------------------------------------
    # 0. shadow_map 格式统一 & Min-Max 重映射
    # ------------------------------------------------------------------
    sm = shadow_map
    if sm.dim() == 3:
        sm = sm.unsqueeze(1)
    elif sm.dim() == 4 and sm.shape[-1] != sm.shape[1]:
        sm = sm.permute(0, 3, 1, 2)
    sm = sm.float()
    _, _, H_L, W_L = sm.shape
    if sm.shape[0] == 1 and N > 1:
        sm = sm.expand(N, -1, -1, -1)

    is_inf_sm = torch.isinf(sm) | torch.isnan(sm)
    valid_mask = ((sm > 0.01) & ~is_inf_sm).float()
    
    valid_z = sm[valid_mask.bool()]
    if valid_z.numel() == 0:
        return torch.ones((N, H, W), device=device)

    z_min = valid_z.min()
    z_max = valid_z.max()
    depth_range = (z_max - z_min).clamp(min=1e-3)

    # 在进行减法和除法前，将 inf 替换为 z_min，从源头抹杀 0 * inf 产生的 NaN 梯度
    safe_sm = torch.where(valid_mask.bool(), sm, z_min.detach())

    # 此时 safe_sm 里面只有干净的有限数字了，不再会报错
    z_norm = ((safe_sm - z_min) / depth_range).clamp(0.0, 1.0) * valid_mask
    z2_norm = (z_norm ** 2) * valid_mask

    sat_w  = build_sat(valid_mask)
    sat_z  = build_sat(z_norm)
    sat_z2 = build_sat(z2_norm)

    # ------------------------------------------------------------------
    # 1. 反投影主相机深度图 → 世界坐标
    # ------------------------------------------------------------------
    main_viewmat = main_camera["viewmat"]
    main_K       = main_camera["K"]

    # 主相机背景也会传 inf 进来，投射到光源空间计算切比雪夫和相似三角形时，也会触发链式法则的 inf 陷阱
    invalid_main_depth = torch.isinf(camera_depth_map) | torch.isnan(camera_depth_map)
    safe_camera_depth = torch.where(invalid_main_depth, torch.ones_like(camera_depth_map), camera_depth_map)

    world_points = _unproject_opencv(
        safe_camera_depth, main_K, main_viewmat
    )

    # ------------------------------------------------------------------
    # 2. 法线偏置
    # ------------------------------------------------------------------
    if camera_normal is not None:
        N_vec = F.normalize(camera_normal.view(N, H * W, 3), p=2, dim=-1)
        lR = light_camera["viewmat"][:, :3, :3]
        lt = light_camera["viewmat"][:, :3,  3]
        light_pos = (-(lR.transpose(1, 2) @ lt.unsqueeze(-1))).squeeze(-1)
        light_pos = light_pos.expand(N, -1).unsqueeze(1)

        L = F.normalize(light_pos - world_points, p=2, dim=-1)
        N_dot_L = torch.sum(N_vec * L, dim=-1, keepdim=True).clamp(min=0.0)
        slope_factor   = 1.0 - N_dot_L
        normal_offset  = N_vec * normal_bias_scale * slope_factor
        world_points   = world_points + normal_offset

    # ------------------------------------------------------------------
    # 3. 投影到光源相机 → 像素坐标 + 深度
    # ------------------------------------------------------------------
    light_viewmat = light_camera["viewmat"].expand(N, -1, -1)
    light_K       = light_camera["K"].expand(N, -1, -1)

    px_f, py_f, current_depth = _project_opencv(
        world_points, light_K, light_viewmat
    )

    px_f = px_f.view(N, H, W)
    py_f = py_f.view(N, H, W)
    current_depth = current_depth.view(N, H, W)

    px = px_f.round().long()
    py = py_f.round().long()

    in_light_frustum = (
        (px >= 0) & (px < W_L) &
        (py >= 0) & (py < H_L) &
        (current_depth > 0)
    )

    # ------------------------------------------------------------------
    # 4. 当前接收面深度重映射
    # ------------------------------------------------------------------
    t_metric_biased = current_depth - shadow_bias
    t_norm_biased = (t_metric_biased - z_min) / depth_range

    # ------------------------------------------------------------------
    # 5. Blocker Search（PCSS 第一步）
    # ------------------------------------------------------------------
    r_b = blocker_search_radius

    sum_w_b  = query_sat(sat_w,  py - r_b, px - r_b, py + r_b, px + r_b, H_L, W_L)
    sum_z_b  = query_sat(sat_z,  py - r_b, px - r_b, py + r_b, px + r_b, H_L, W_L)
    sum_z2_b = query_sat(sat_z2, py - r_b, px - r_b, py + r_b, px + r_b, H_L, W_L)

    valid_count_b = sum_w_b.clamp(min=1e-5)
    mu_b  = sum_z_b  / valid_count_b
    var_b = (sum_z2_b / valid_count_b - mu_b ** 2).clamp(min=1e-8) 

    delta_b = t_norm_biased - mu_b
    p_unoccluded_b = torch.where(
        delta_b > 0,
        var_b / (var_b + delta_b ** 2),
        torch.ones_like(var_b)
    ).clamp(0.0, 1.0)

    if light_bleeding_reduction > 0.0:
        p_unoccluded_b = ((p_unoccluded_b - light_bleeding_reduction) / (1.0 - light_bleeding_reduction)).clamp(min=0.0)

    p_occ          = (1.0 - p_unoccluded_b).clamp(min=1e-5)
    z_blocker_norm = (mu_b - p_unoccluded_b * t_norm_biased) / p_occ

    valid_blocker_mask = (sum_w_b >= 1.0) & (p_occ > 0.05)
    z_blocker_norm = torch.where(valid_blocker_mask, z_blocker_norm, t_norm_biased)

    # ------------------------------------------------------------------
    # 6. 半影半径估算 
    # ------------------------------------------------------------------
    z_blocker_metric = z_blocker_norm * depth_range + z_min
    
    w_penumbra_metric = (
        (t_metric_biased - z_blocker_metric) / z_blocker_metric.clamp(min=1e-3) * light_size
    ).clamp(min=0.0)

    fx = light_K[:, 0, 0].view(N, 1, 1)
    r_pcf_px = (w_penumbra_metric / 2.0 * fx / t_metric_biased.clamp(min=1e-3)).round().long()
    r_pcf_px = r_pcf_px.clamp(min=1, max=max_pcf_radius)

    # ------------------------------------------------------------------
    # 7. VSSM PCF（Chebyshev 上界）
    # ------------------------------------------------------------------
    area_p   = get_area(py, px, r_pcf_px, H_L, W_L)
    sum_w_p  = query_sat(sat_w,  py - r_pcf_px, px - r_pcf_px, py + r_pcf_px, px + r_pcf_px, H_L, W_L)
    sum_z_p  = query_sat(sat_z,  py - r_pcf_px, px - r_pcf_px, py + r_pcf_px, px + r_pcf_px, H_L, W_L)
    sum_z2_p = query_sat(sat_z2, py - r_pcf_px, px - r_pcf_px, py + r_pcf_px, px + r_pcf_px, H_L, W_L)

    valid_count_p = sum_w_p.clamp(min=1e-5)
    mu_p  = sum_z_p  / valid_count_p
    var_p = (sum_z2_p / valid_count_p - mu_p ** 2).clamp(min=1e-8)

    delta_p = t_norm_biased - mu_p
    P_valid = torch.where(
        delta_p > 0,
        var_p / (var_p + delta_p ** 2),
        torch.ones_like(var_p)
    ).clamp(0.0, 1.0)

    if light_bleeding_reduction > 0.0:
        P_valid = ((P_valid - light_bleeding_reduction) / (1.0 - light_bleeding_reduction)).clamp(min=0.0)

    obj_ratio        = (sum_w_p / area_p).clamp(0.0, 1.0)
    obj_occlude_rate = 1.0 - P_valid
    visibility       = 1.0 - (obj_ratio * obj_occlude_rate)

    # 处理主深度图中无穷远的背景，使其不受阴影判定干扰
    visibility = torch.where(invalid_main_depth, torch.ones_like(visibility), visibility)

    # 视锥外强制可见
    visibility = torch.where(in_light_frustum, visibility, torch.ones_like(visibility))

    return visibility  # (N, H, W)
