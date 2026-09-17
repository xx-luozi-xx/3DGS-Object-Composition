import time
from functools import lru_cache
        
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from gaussians import Gaussians
from gaussians_pbr import Gaussians_PBR
import utils
from utils import (
    plt_show,
    axis_angle_to_matrix,
)
from renders import (
    gs_unified_render,
    gs_render,
    gs_env_render2,
    add_sun_to_env_map,
)
import env_diff
import env_sp3g as env_sp
import probes_tool
from IBL import (
    IBL_test3,
    IBL,
    IBL_transparent2 as IBL_transparent
)
from shadow6 import compute_VSSM


class Scene:
    def __init__(self, gs_bg_path, gs_obj_path, device, dir_light=None, transparent = False, shadow_factor = 0.5, obj_IOR = 1.1, light_size = 2):
        self.device         = device
        self.transparent    = transparent
        self.shadow_factor  = shadow_factor
        self.obj_IOR        = obj_IOR
        self.light_size      = light_size

        #* loading gs_obj
        if gs_obj_path != None:
            print("\r<Scene>: loading gs obj...", end="")
            self.gs_obj = Gaussians_PBR.from_ply(gs_obj_path)
            self.gs_obj.to(self.device)
            print("\r<Scene>: loading gs obj [finished]")
        else:
            self.gs_obj = None

        #* loading gs_bg 
        print("\r<Scene>: loading gs background...", end="")
        self.gs_bg = Gaussians.from_ply(gs_bg_path)
        self.gs_bg.to(self.device)
        print("\r<Scene>: loading gs background [finished]")

        if dir_light != None:
            self.set_dir_light(dir_light)

        #* defluat camera_intrinsics, img size and intrinsics
        self.set_camera_intrinsics()

        #* obj deflaut transform is I
        self.obj_transform = torch.eye(4, dtype=torch.float32, device=self.device)
        self.frame_idx = 0

        #* deflaut render background color
        self.set_background(bg_color=[0,0,0])

    def set_IOR(self, IOR):
        self.obj_IOR = IOR

    def set_dir_light(self, dir_light):
        if not isinstance(dir_light, torch.Tensor):
            dir_light = torch.tensor(dir_light, dtype=torch.float32, device=self.device)
        self.dir_light = torch.nn.functional.normalize(dir_light, dim=0)

    def set_background(self, bg_color):
        if not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.bg_color = bg_color

    def set_camera_intrinsics(
            self, 
            h_w: tuple[int, int] = (512, 512), 
            intrinsics: tuple[float, float, float, float] =  (402.2, 402.2, 256.0, 256.0)
    ):
        """
        h_w: (height, width) 图像尺寸
        intrinsics: (fx, fy, cx, cy) 相机内参
        """
        if not isinstance(h_w, torch.Tensor):
            h_w = torch.tensor(h_w, dtype=torch.float32, device=self.device)
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)

        self.h_w = h_w

        self.K = torch.eye(3, device=self.device)
        self.K[0, 0] = intrinsics[0]
        self.K[1, 1] = intrinsics[1]
        self.K[0, 2] = intrinsics[2]
        self.K[1, 2] = intrinsics[3]

    def set_object_transform(self, transform: torch.Tensor):
        if not isinstance(transform, torch.Tensor):
            transform = torch.tensor(transform, dtype=torch.float32, device=self.device)
    
        self.obj_transform = transform

    def object_transform(self, transform):
        """
        transform: # 4,4
        """
        if not isinstance(transform, torch.Tensor):
            transform = torch.tensor(transform, dtype=torch.float32, device=self.device)    
        self.obj_transform =  transform @ self.obj_transform

    def object_transform_translation(self, translation:tuple[float, float, float] = [0.0, 0.0, 0.0]):
        """
        应用平移变换到物体
        translation: [dx,dy,dz]# 3
        """
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32, device=self.device)    
        
        # 创建纯平移矩阵
        translation_matrix = torch.eye(4, device=self.device)
        translation_matrix[:3, 3] = translation
        
        # 应用平移：T_new * T_old (先执行旧变换，再执行新平移)
        self.obj_transform = translation_matrix @ self.obj_transform

    def set_object_transform_translation(self, translation:tuple[float, float, float] = [0.0, 0.0, 0.0]):
        """
        赋值平移变换到物体
        translation: [dx,dy,dz]# 3
        """
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32, device=self.device)    
        
        # 创建纯平移矩阵
        translation_matrix = torch.eye(4, device=self.device)
        translation_matrix[:3, 3] = translation
        
        self.obj_transform = translation_matrix

    def object_transform_rotation(
        self,
        axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        degree: float = 0.0,
    ):
        """
        应用旋转变换到物体
        axis: 旋转轴 (x, y, z)
        degree: 旋转角度（度）
        """
        if not isinstance(axis, torch.Tensor):
            axis = torch.tensor(axis, dtype=torch.float32, device=self.device)
        if not isinstance(degree, torch.Tensor):
            degree = torch.tensor(degree, dtype=torch.float32, device=self.device)
        
        # 归一化旋转轴
        axis = torch.nn.functional.normalize(axis, dim=0)
        
        # 角度转弧度
        angle_rad = degree * torch.pi / 180.0
        
        # 创建轴角表示并转换为旋转矩阵
        axis_angle = axis * angle_rad
        R_3x3 = axis_angle_to_matrix(axis_angle.unsqueeze(0))[0]  # (3, 3)
        
        # 扩展到4x4齐次坐标矩阵
        rotation_matrix_4x4 = torch.eye(4, device=self.device)
        rotation_matrix_4x4[:3, :3] = R_3x3
        
        # 应用旋转：R_new * T_old
        self.obj_transform = rotation_matrix_4x4 @ self.obj_transform

    def object_scale(self, scale):
        if self.gs_obj is not None:
            self.gs_obj.apply_uniform_scale(scale)


    def render(self, eyes, targets, ups=None):
        """
        设置物体变换：
        eyes: 相机原点 # N,3 list
        targets: 视线目标点 # N,3 list
        ups:相机up向量, 默认[0,1,0], opengl风格 # N,3 list
        """
        if not isinstance(eyes, torch.Tensor):
            eyes = torch.tensor(eyes, dtype=torch.float32, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        assert (eyes.shape == targets.shape)
        N = eyes.shape[0]
        H = int(self.h_w[0])
        W = int(self.h_w[1])
        h_ws = self.h_w.repeat(N, 1)
        Ks = self.K.unsqueeze(0).repeat(N, 1, 1)
        bg_colors = self.bg_color.repeat(N, 1)

        if ups is None:
            ups = [[0.0,1.0,0.0]]*N


        #* 对物体变换(obj->world)等价于对相机逆变换(world->obj)
        inv = torch.linalg.inv(self.obj_transform) #todo 可优化为预计算 ro lazy tag
        # 转换为齐次坐标 (N, 4)
        eyes_homo = torch.cat([eyes, torch.ones(eyes.shape[0], 1, device=self.device)], dim=-1)
        targets_homo = torch.cat([targets, torch.ones(targets.shape[0], 1, device=self.device)], dim=-1)

        if ups is None:
            ups_world = torch.tensor([[0, 1, 0]] * N, dtype=torch.float32, device=self.device)
        else:
            ups_world = torch.tensor(ups, dtype=torch.float32, device=self.device)
        
        ups_homo = torch.cat([ups_world, torch.zeros(N, 1, device=self.device)], dim=-1)
        ups_obj_homo = (inv @ ups_homo.T).T
        ups_obj = ups_obj_homo[:, :3]

        eyes_obj_homo = (inv @ eyes_homo.T).T
        targets_obj_homo = (inv @ targets_homo.T).T

        eyes_obj = eyes_obj_homo[:, :3]
        targets_obj = targets_obj_homo[:, :3]

        #* 求物体中心用于渲染环境贴图以及shadow mapping
        bbox = None
        if self.gs_obj is not None:
            bbox_min = self.gs_obj.xyz.min(dim=0)[0]
            bbox_max = self.gs_obj.xyz.max(dim=0)[0]
            bbox = torch.stack([bbox_min, bbox_max], dim=1)
        else:
            raise ValueError("gs_obj is None.")

        bbox_center = bbox.mean(dim=1)
        bbox_center_homo = torch.cat([bbox_center, torch.tensor([1.0], dtype=torch.float32, device=bbox_center.device)])
        bbox_center_homo = self.obj_transform @ bbox_center_homo
        bbox_center = bbox_center_homo[:3]

        #* 生成主相机服务阴影
        main_viewmats = [
            utils.create_view_matrix(
                position=pos,
                look_at=tar,
                up=up,
                device=self.device
            )
            for pos, tar, up in zip(eyes_obj, targets_obj, ups_obj * (-1))
        ]
        main_viewmats = torch.stack(main_viewmats, dim=0)

        main_cameras = {
            "viewmat": main_viewmats,
            "K":       Ks,
            "height":  H,
            "width":   W,
        }

        if self.gs_obj is not None:
            obj_mesh_T, obj_mesh_G = gs_unified_render(
                device = self.device,
                gs = self.gs_obj,
                view_origins = eyes_obj,
                view_targets = targets_obj,
                h_ws = h_ws,
                Ks = Ks,
                IOR2 = self.obj_IOR,
                bg_colors = [0,0,0],
                ups = ups_obj*(-1)
            )
        else:
            raise ValueError("gs_obj is None. ")

        # !将局部坐标系下的 G-buffer 几何属性全部转回世界坐标系
        mask_float = obj_mesh_G["mask"].float().unsqueeze(-1)

        R_l2w = self.obj_transform[:3, :3]
        T_l2w = self.obj_transform[:3,  3]

        P_local = obj_mesh_G["position"]
        P_world = P_local @ R_l2w.T + T_l2w.view(1, 1, 1, 3)
        obj_mesh_G["position"] = P_world * mask_float

        N_local = obj_mesh_G["normal"]
        N_world = torch.nn.functional.normalize(N_local @ inv[:3, :3], p=2, dim=-1)
        obj_mesh_G["normal"] = N_world * mask_float

        V_local = obj_mesh_G["view"]
        V_world = torch.nn.functional.normalize(V_local @ R_l2w.T, p=2, dim=-1)
        obj_mesh_G["view"] = V_world * mask_float

        if "refract_dir" in obj_mesh_G:
            Ref_local = obj_mesh_G["refract_dir"]
            Ref_world = torch.nn.functional.normalize(Ref_local @ R_l2w.T, p=2, dim=-1)
            obj_mesh_G["refract_dir"] = Ref_world * mask_float

        obj_mesh_G["depth"][obj_mesh_G["depth"] < 1e-4] = float('inf')

        #* 渲染背景高斯
        bg_gs_Cs, bg_gs_ds = gs_render(
            device = self.device,
            gs = self.gs_bg,
            view_origins = eyes,
            view_targets = targets,
            h_ws = h_ws,
            Ks = Ks,
            bg_colors = bg_colors,
            render_mode = "RGB+ED",
            ups=[[-val for val in up] for up in ups]
        )
        bg_gs_ds[bg_gs_ds < 1e-4] = float('inf')

        #* IBL
        if hasattr(self, "probes_pos"):
            pos_tensor = torch.as_tensor(self.probes_pos, device=self.device, dtype=torch.float32)
            dists_sq = torch.sum((pos_tensor - bbox_center) ** 2, dim=-1)
            K = 1
            topk_dists_sq, topk_indices = torch.topk(dists_sq, K, largest=False)
            idx = int(topk_indices[0].item())

            _, interp_env_d, interp_mips = self.get_probe(idx)
            if self.transparent:
                obj_CLs = IBL_transparent(interp_env_d, interp_mips, obj_mesh_G, obj_mesh_T)
            else:
                obj_CLs = IBL(interp_env_d, interp_mips, obj_mesh_G, obj_mesh_T)
        else:
            env_img = gs_env_render2(
                device=self.device,
                gs=self.gs_bg,
                origin=bbox_center,
                bg_color = bg_colors[0],
            )

            if not hasattr(self, "dir_light"):
                sun_direction = torch.tensor([0,1,0], dtype=torch.float32, device=self.device)
            else:
                sun_direction = self.dir_light
            env_img = add_sun_to_env_map(
                env_img=env_img,
                sun_direction=sun_direction,
                intensity = 200,
                radius_deg= 3.0,
                softness = 1.0,
            )

            obj_CLs = IBL_test3(env_img, obj_mesh_G, obj_mesh_T)

        obj_CLs = obj_CLs**(1.0/2.2)

        #* z-buffer融合物体和背景
        obj_ds = obj_mesh_G["depth"][..., 0]
        z_pass = (obj_ds < bg_gs_ds).unsqueeze(-1)

        obj_alpha = obj_mesh_G["mask"].float().unsqueeze(-1)  # 确保 float，防止 bool 减法报错
        
        valid_alpha = torch.where(z_pass, obj_alpha, torch.zeros_like(obj_alpha))
        valid_obj_CLs = torch.where(z_pass, obj_CLs, torch.zeros_like(obj_CLs))
        img_CLs = valid_obj_CLs + bg_gs_Cs * (1.0 - valid_alpha)

        img_ds = torch.where(obj_ds < bg_gs_ds, obj_ds, bg_gs_ds)

        # return bg_gs_Cs

        if self.transparent:
            return img_CLs.clamp(0, 1)

        #* shadow
        if hasattr(self, "dir_light"):
            # 1. 计算局部空间半径与包围盒中心
            lengths = bbox[:, 1] - bbox[:, 0]
            local_radius = (torch.norm(lengths) / 2.0).item()
            cover_radius = local_radius
            local_bbox_center = bbox.mean(dim=1)

            # 2. 世界光方向 -> 局部坐标系
            dir_light_homo = torch.cat([self.dir_light, torch.tensor([0.0], device=self.device)])
            local_light_dir = (inv @ dir_light_homo.T).T[:3]
            local_light_dir = torch.nn.functional.normalize(local_light_dir, dim=0)

            # 3. 光源相机位置与内参（局部空间）
            light_dist = cover_radius * 100
            obj_light_point = local_bbox_center + local_light_dir * light_dist

            shadow_res = 1024
            focal_light = shadow_res * light_dist / (2.0 * cover_radius)
            light_K = torch.tensor([
                [focal_light, 0.0,         shadow_res / 2.0],
                [0.0,         focal_light, shadow_res / 2.0],
                [0.0,         0.0,         1.0             ],
            ], dtype=torch.float32, device=self.device)

            light_dir_local = torch.nn.functional.normalize(
                local_bbox_center - obj_light_point, p=2, dim=0
            )
            y_axis = torch.tensor([0.0, -1.0, 0.0], device=self.device)
            if torch.abs(torch.dot(light_dir_local, y_axis)) > 0.99:
                y_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            light_up = y_axis

            light_h_ws = torch.tensor([[shadow_res, shadow_res]], dtype=torch.float32, device=self.device)
            light_Ks = light_K.unsqueeze(0)  # (1,3,3)

            # ----------------------------------------------------------------
            #* Shadow Map 生成：按物体类型分支，统一输出 light_space_deeps (1, shadow_res, shadow_res)
            #  - 三种分支全部工作在局部坐标系下，光源位置/目标均为局部坐标，与 VSSM 完全一致
            # ----------------------------------------------------------------
            if self.gs_obj is not None:
                # --- 高斯物体：沿用原有 gs_render ---
                _, light_space_deeps = gs_render(
                    device=self.device,
                    gs=self.gs_obj,
                    view_origins=obj_light_point.unsqueeze(0),
                    view_targets=local_bbox_center.unsqueeze(0),
                    h_ws=light_h_ws,
                    Ks=light_Ks,
                    bg_colors=torch.zeros((1, 3), device=self.device),
                    ups=[light_up.tolist()],
                    render_mode="RGB+ED",
                )
            else:
                raise ValueError("Both gs_obj is None.")

            # 统一后处理：将无效深度（-1 或 < 1e-4）置为 inf，与高斯路径语义完全一致
            light_space_deeps = light_space_deeps.float()
            light_space_deeps[light_space_deeps < 1e-4] = float('inf')
            # light_space_deeps: (1, shadow_res, shadow_res)

            # ----------------------------------------------------------------
            #* VSSM：以下逻辑对三种物体类型完全通用，无需任何修改
            # ----------------------------------------------------------------
            light_viewmat = utils.create_view_matrix(
                position=obj_light_point,
                look_at=local_bbox_center,
                up=light_up,
                device=self.device,
            )

            light_cameras = {
                "viewmat": light_viewmat.unsqueeze(0),
                "K":       light_Ks,
                "height":  shadow_res,
                "width":   shadow_res,
            }

            visibility = compute_VSSM(
                device=self.device,
                camera_depth_map=img_ds,
                main_camera=main_cameras,
                shadow_map=light_space_deeps,
                light_camera=light_cameras,
                blocker_search_radius=20,
                light_size=self.light_size,
                max_pcf_radius=32,
                shadow_bias=0.01,
                normal_bias_scale=0.0,
                camera_normal=obj_mesh_G["normal"]
            ).unsqueeze(-1)  # (N,H,W) -> (N,H,W,1)

            a = self.shadow_factor
            a = torch.where(valid_alpha > 0.5, a * 0.5, a)
            img_CLs = img_CLs * (a * visibility + 1 - a)

        return img_CLs.clamp(0, 1)

    def make_light_probes(self, pos, save_path=None):
        '''
        pos: #N, 3
        '''
        N = len(pos)
        bg_colors = self.bg_color.repeat(N, 1)

        #* 高斯全景贴图
        env_imgs = [gs_env_render2(
            device=self.device,
            gs=self.gs_bg,
            origin=p,
            bg_color = bg_colors[0],
            # out_h = 320,
            # out_w = 320*2
        )
        for p in pos]

        #* 平行光补偿
        if not hasattr(self, "dir_light"):
            sun_direction = torch.tensor([0,1,0], dtype=torch.float32, device=self.device)
        else:
            sun_direction = self.dir_light
        
        env_imgs = [add_sun_to_env_map(
            env_img=env_img,                                 
            sun_direction=sun_direction,
            intensity = 200,
            radius_deg= 3.0,
            softness = 1.0,
        )
        for env_img in env_imgs]


        #* 漫反射分量
        env_ds = [env_diff.compute_diffuse_panorama(env_img) 
                  for env_img in env_imgs] # N list of (H, W, 3)

        #* 镜面反射Part A
        mipss = [env_sp.prefilter_env_map(env_img, num_samples=4096, max_mips=5)
                 for env_img in env_imgs] # N list of max_mips list of (H, W, 3)

        if save_path is not None:
            probes_tool.save_all_probes(
                pos_list=pos,
                env_ds=env_ds,
                mipss = mipss,
                save_dir=save_path
            )

        return (pos, env_ds, mipss)
    


    def load_light_probes(self, probes_dir):
        """
        self.probes = (pos, env_ds, mipss) #list, list, list(list)
        """
        self.probes_dir = probes_dir
        self.probes_pos =  probes_tool.load_all_probes_pos(
            device=self.device, 
            save_dir=probes_dir
        )
        
    @lru_cache(maxsize=4) 
    def get_probe(self, probe_idx):
        """
        返回:
        pos     : torch.Tensor (3,)
        env_d   : torch.Tensor (H, W, 3)   漫反射分量
        mips    : list of torch.Tensor(H, W, 3)      各级 mip 
        """
        return probes_tool.load_probe(probe_idx, self.probes_dir, device=self.device)
    
def depth_smooth(depth):

    d = depth[0].clone()  # (H, W)
    mask = torch.isinf(d)
    d[mask] = d[~mask].max()

    # 转换为 (B, C, H, W) 供卷积使用 -> (1, 1, H, W)
    d_t = d.unsqueeze(0).unsqueeze(0)
    H, W = d_t.shape[2], d_t.shape[3]

    # ==========================================
    # 1. 纯 PyTorch 双边滤波 (Bilateral Filter)
    # ==========================================
    kernel_size = 15
    radius = kernel_size // 2
    sigma_color = 0.5
    sigma_space = 15.0

    # 生成空间距离权重 (1, 225, 1)
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=d.device),
        torch.arange(-radius, radius + 1, device=d.device), 
        indexing='ij'
    )
    spatial_dist_sq = (x**2 + y**2).float()
    spatial_weights = torch.exp(-spatial_dist_sq / (2 * sigma_space**2)).view(1, -1, 1)

    # 处理边缘填充 (replicate 避免边缘黑边)
    padded_d = F.pad(d_t, (radius, radius, radius, radius), mode='replicate')

    # 将图像展开提取局部窗口: (1, 225, H*W)
    unfolded_d = F.unfold(padded_d, kernel_size=kernel_size) 
    d_flat = d_t.view(1, 1, -1)  # 展平目标图: (1, 1, H*W)

    # 计算色彩距离权重: (1, 225, H*W)
    color_dist_sq = (unfolded_d - d_flat)**2
    color_weights = torch.exp(-color_dist_sq / (2 * sigma_color**2))

    # 空间权重 * 色彩权重
    weights = spatial_weights * color_weights
    weights_sum = weights.sum(dim=1, keepdim=True)

    # 加权平均求和
    d_smooth = (unfolded_d * weights).sum(dim=1, keepdim=True) / (weights_sum + 1e-8)
    d_smooth = d_smooth.view(1, 1, H, W)  # 还原为图像形状

    # ==========================================
    # 2. 纯 PyTorch 高斯滤波 (Gaussian Blur)
    # ==========================================
    # 使用 torchvision 内置的高效 GPU 滤波
    d_smooth = TF.gaussian_blur(d_smooth, kernel_size=[15, 15], sigma=[10.0, 10.0])

    # ==========================================
    # 3. 还原 Mask 并覆盖
    # ==========================================
    d_smooth = d_smooth.squeeze() # 还原为 (H, W)
    d_smooth[mask] = float('inf')
    return d_smooth.unsqueeze(0) # 还原为 (1, H, W)



def make_light_probes_from_env(env_imgs, pos, num_samples=4096, save_path=None):
    '''
    pos: #N, 3
    '''
    N = len(pos)

    #* 漫反射分量
    env_ds = [env_diff.compute_diffuse_panorama(env_img) 
                for env_img in env_imgs] # N list of (H, W, 3)

    #* 镜面反射Part A
    mipss = [env_sp.prefilter_env_map(env_img, num_samples=num_samples, max_mips=5)
                for env_img in env_imgs] # N list of max_mips list of (H, W, 3)

    if save_path is not None:
        probes_tool.save_all_probes(
            pos_list=pos,
            env_ds=env_ds,
            mipss = mipss,
            save_dir=save_path
        )

    return (pos, env_ds, mipss)
