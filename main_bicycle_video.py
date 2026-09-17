import json
import math
import os.path
import time

import cv2
import torch
import numpy as np

from scene import Scene


@torch.no_grad()
def render_camera_orbit_video():
    NUM_CIRCLES        = 2       # 绕行圈数
    ANIMATION_SECONDS  = 10.0   # 动画总时长（秒）
    USE_COSINE_EASING  = True   # True=缓入缓出, False=匀速绕行

    ORBIT_RADIUS       = 2.0    # 轨道半径 r，对应交互代码里的 r=2
    ORBIT_HEIGHT       = 1.4    # 相机高于物体的 Y 偏移，对应代码里 1.4 + obj_position[1]
    THETA_START        = 0.0    # 起始角度（弧度），0 = 正前方

    OUTPUT_FPS         = 30
    VIDEO_CODEC        = 'mp4v'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Video Render] 使用设备: {device}")

    gs_model_path = "./model/hotdog.ply"
    gs_bg_path    = "./model/bicycle_d_1.ply"
    data_path     = "./data/bicycle"

    camera_path = os.path.join(data_path, "camera.json")
    with open(camera_path, "r") as f:
        camera = json.load(f)

    h_w = [camera["image_size"]["height"], camera["image_size"]["width"]]
    intrinsics = [
        camera["intrinsics"]["fx"],
        camera["intrinsics"]["fy"],
        camera["intrinsics"]["cx"],
        camera["intrinsics"]["cy"]
    ]

    scene = Scene(
        gs_bg_path    = gs_bg_path,
        gs_obj_path   = gs_model_path,
        device        = device,
        transparent   = True,          # 关键：开启透明渲染
        shadow_factor = 0.25,
    )
    scene.set_dir_light(dir_light=[0.3, 1, 0.3])
    scene.set_camera_intrinsics(h_w=h_w, intrinsics=intrinsics)
    scene.gs_bg.add_background_gaussians(bg_color=[0, 0, 0])  # 黑色背景更好展示透明效果

    obj_scale      = 0.5
    obj_position   = [0.5, -0.3, 0.1]
    obj_rotation_x = 270
    obj_rotation_y = 0

    scene.object_scale(obj_scale)
    scene.object_transform_rotation([1, 0, 0], obj_rotation_x)
    scene.object_transform_rotation([0, 1, 0], obj_rotation_y)
    scene.object_transform_translation(obj_position)

    probes_dir = os.path.join(data_path, "probes")
    pos = [obj_position]
    with torch.no_grad():
        for i in range(len(pos)):
            print(f"making probe #{i}", end='')
            _, env_ds, mipss = scene.make_light_probes(pos[i:i+1])
            import probes_tool
            probes_tool.save_probe(
                probe_idx=i,
                pos = pos[i],
                env_d = env_ds[0],
                mips = mipss[0],
                save_dir = probes_dir
            )
            del env_ds
            del mipss
            torch.cuda.empty_cache()

            print(f"\rmaking probe #{i}[saved]")   
    scene.load_light_probes(probes_dir)

    total_frames       = int(ANIMATION_SECONDS * OUTPUT_FPS)
    total_rotation_rad = 2.0 * math.pi * NUM_CIRCLES

    theta_sequence = []
    for f in range(total_frames):
        t = f / float(total_frames - 1) if total_frames > 1 else 0.0

        if USE_COSINE_EASING:
            progress = (1.0 - math.cos(math.pi * t)) / 2.0
        else:
            progress = t

        theta = THETA_START + total_rotation_rad * progress
        theta_sequence.append(theta)

    cam_up = [[0.0, 1.0, 0.0]]

    print(f"[Video Render] 相机公转动画配置:")
    print(f"  总帧数:    {total_frames}  ({ANIMATION_SECONDS}s @ {OUTPUT_FPS}fps)")
    print(f"  绕行圈数:  {NUM_CIRCLES} 圈")
    print(f"  轨道半径:  {ORBIT_RADIUS}")
    print(f"  轨道高度:  obj_position[1] + {ORBIT_HEIGHT} = {obj_position[1] + ORBIT_HEIGHT:.2f}")
    print(f"  余弦缓动:  {USE_COSINE_EASING}")

    video_dir  = os.path.join(data_path, "video")
    os.makedirs(video_dir, exist_ok=True)
    easing_tag = "easing" if USE_COSINE_EASING else "linear"
    video_name = f"hotdog_transparent_orbit_{NUM_CIRCLES}circles_{easing_tag}_{OUTPUT_FPS}fps.mp4"
    video_path = os.path.join(video_dir, video_name)

    H, W   = h_w[0], h_w[1]
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (W, H))

    if not writer.isOpened():
        raise RuntimeError(f"[Video Render] 无法创建视频文件: {video_path}")
    print(f"[Video Render] 视频将保存至: {video_path}")

    print("[Video Render] 开始渲染相机公转动画...")
    render_start = time.time()

    for frame_idx, theta in enumerate(theta_sequence):

        eye = [[
            ORBIT_RADIUS * math.sin(theta) + obj_position[0],
            ORBIT_HEIGHT + obj_position[1],
            ORBIT_RADIUS * math.cos(theta) + obj_position[2],
        ]]
        target = [[
            obj_position[0],
            obj_position[1],
            obj_position[2],
        ]]

        imgs = scene.render(eyes=eye, targets=target, ups=cam_up)  # (1, H, W, 3)

        img_np = imgs[0].cpu().numpy()
        if img_np.dtype in (np.float32, np.float64):
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        writer.write(img_bgr)

        if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
            elapsed   = time.time() - render_start
            remaining = elapsed / (frame_idx + 1) * (total_frames - frame_idx - 1)
            circles_done = (theta - THETA_START) / (2.0 * math.pi)
            print(
                f"  [{frame_idx + 1:4d}/{total_frames}]  "
                f"theta={math.degrees(theta):7.2f} deg  "
                f"已绕行 {circles_done:.2f}/{NUM_CIRCLES} 圈 | "
                f"已用时 {elapsed:6.1f}s  预计剩余 {remaining:6.1f}s"
            )

    writer.release()
    total_time = time.time() - render_start
    print(f"\n[Video Render] 渲染完成！")
    print(f"  总帧数:   {total_frames}")
    print(f"  总耗时:   {total_time:.2f} 秒")
    print(f"  渲染速度: {total_frames / total_time:.2f} FPS")
    print(f"  视频路径: {video_path}")


if __name__ == "__main__":
    render_camera_orbit_video()
