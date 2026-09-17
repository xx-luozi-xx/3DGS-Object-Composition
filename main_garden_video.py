import json
import math
import os.path
import time

import cv2
import torch
import numpy as np

from scene import Scene


@torch.no_grad()
def render_floating_rotation_video():

    FLOAT_AMPLITUDE    = 0.12    # 上下浮动的幅度 (单位: 世界坐标)，调大=浮得更高
    FLOAT_CYCLES       = 5       # 浮动次数，5次后自然回到原始高度
    ROTATION_SPEED_DPS = 72.0    # 每秒旋转多少度 (72 dps = 5秒转一圈)
    ANIMATION_SECONDS  = 10.0    # 动画总时长 (10秒 = 5次完整浮动周期)

    OUTPUT_FPS         = 30
    VIDEO_CODEC        = 'mp4v'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Video Render] 使用设备: {device}")

    gs_model_path = "./model/armadillo_3.ply"
    gs_bg_path    = "./model/garden_d_1.ply"
    data_path     = "./data/garden"

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
        shadow_factor = 0.25,
    )
    scene.set_dir_light(dir_light=[0.3, 1, 0.3])
    scene.set_camera_intrinsics(h_w=h_w, intrinsics=intrinsics)
    scene.gs_bg.add_background_gaussians(bg_color=[1, 1, 1])

    obj_scale    = 0.5
    obj_position = [2.0, -1, -0.3]   # 基准位置，浮动以此为中心
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

    fixed_eye    = [[1.7119575284344233, 0.3999999999999999, -4.637711787350052]]
    fixed_target = [[1.0, -1.0, -0.6]]

    total_frames = int(ANIMATION_SECONDS * OUTPUT_FPS)

    deg_per_frame = ROTATION_SPEED_DPS / OUTPUT_FPS

    float_y_offsets = [
        FLOAT_AMPLITUDE * math.sin(2.0 * math.pi * FLOAT_CYCLES * f / total_frames)
        for f in range(total_frames + 1)  # +1 保证最后一帧也能计算 delta
    ]

    print(f"[Video Render] 动画配置:")
    print(f"  总帧数:      {total_frames} ({ANIMATION_SECONDS}秒 @ {OUTPUT_FPS}fps)")
    print(f"  浮动幅度:    ±{FLOAT_AMPLITUDE} 世界单位")
    print(f"  浮动周期:    {FLOAT_CYCLES} 次 (第{total_frames}帧精确回原位)")
    print(f"  旋转速度:    {ROTATION_SPEED_DPS} 度/秒")
    print(f"  总旋转角度:  {ROTATION_SPEED_DPS * ANIMATION_SECONDS:.1f} 度")

    video_dir  = os.path.join(data_path, "video")
    os.makedirs(video_dir, exist_ok=True)
    video_name = f"armadillo_float_rotate_{OUTPUT_FPS}fps.mp4"
    video_path = os.path.join(video_dir, video_name)

    H, W      = h_w[0], h_w[1]
    fourcc    = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer    = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (W, H))

    if not writer.isOpened():
        raise RuntimeError(f"[Video Render] 无法创建视频文件: {video_path}")
    print(f"[Video Render] 视频将保存至: {video_path}")

    print("[Video Render] 开始渲染悬浮旋转动画...")
    render_start      = time.time()
    prev_y_offset     = 0.0   # 上一帧的 Y 浮动偏移量

    for frame_idx in range(total_frames):

        current_y_offset = float_y_offsets[frame_idx]
        delta_y          = current_y_offset - prev_y_offset

        scene.object_transform_translation([
            -obj_position[0],
            -obj_position[1] - prev_y_offset,   # 同时撤销上一帧的浮动偏移
            -obj_position[2]
        ])

        scene.object_transform_rotation([0, 1, 0], deg_per_frame)

        scene.object_transform_translation([
            obj_position[0],
            obj_position[1] + current_y_offset,  # 本帧的浮动高度
            obj_position[2]
        ])

        prev_y_offset = current_y_offset

        imgs = scene.render(eyes=fixed_eye, targets=fixed_target)  # (1, H, W, 3)

        img_np = imgs[0].cpu().numpy()
        if img_np.dtype in (np.float32, np.float64):
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        writer.write(img_bgr)

        if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
            elapsed   = time.time() - render_start
            remaining = elapsed / (frame_idx + 1) * (total_frames - frame_idx - 1)
            cycles_done = FLOAT_CYCLES * frame_idx / total_frames
            print(
                f"  [{frame_idx + 1:4d}/{total_frames}]  "
                f"Y偏移: {current_y_offset:+.3f}  "
                f"浮动: {cycles_done:.1f}/{FLOAT_CYCLES} 圈 | "
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
    render_floating_rotation_video()
