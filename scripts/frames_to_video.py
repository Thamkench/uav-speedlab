# scripts/frames_to_video.py
# -*- coding: utf-8 -*-
"""
Frames to Video Converter (VisDrone-friendly)
帧序列转视频的小工具（兼容 VisDrone 等数据集）

Usage / 用法示例：
    python scripts/frames_to_video.py \
        --seq-dir data/visdrone_frames/uav0000126_00001_v \
        --out-video data/videos/uav0000126_00001_v.mp4 \
        --fps 30

- 自动按文件名排序读取帧（*.jpg / *.png 等）
- 使用第一帧尺寸作为输出视频大小
- 可用于将 VisDrone 的 frame 序列合成为 mp4，方便后续做 YOLO 跟踪/测速
"""

import os
import glob
import argparse
import cv2


def frames_to_video(seq_dir: str,
                    out_video: str,
                    fps: float = 30.0,
                    pattern: str = "*.jpg") -> None:
    """
    Convert an image sequence into a video.
    将图像帧序列合成为视频文件。

    Args:
        seq_dir (str): Directory containing frame images.
                       帧图像所在文件夹路径。
        out_video (str): Output video path (.mp4 recommended).
                         输出视频路径（推荐 .mp4）。
        fps (float): Frames per second for the output video.
                     输出视频的帧率（单位：帧/秒）。
        pattern (str): Glob pattern for frame files, e.g. "*.jpg".
                       帧文件的通配符模式，例如 "*.jpg"。
    """
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    # Collect frame paths, sorted by filename
    # 收集并按文件名排序帧路径
    frame_paths = sorted(glob.glob(os.path.join(seq_dir, pattern)))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in {seq_dir} with pattern {pattern}"
        )

    print(f"[INFO] Found {len(frame_paths)} frames in {seq_dir}")

    # Read first frame to determine resolution
    # 读取第一帧以确定视频分辨率
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    print(f"[INFO] Frame size: {w}x{h}")

    # Ensure output directory exists
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    # Initialize video writer
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_video}")

    # Write all frames
    # 逐帧写入视频
    for i, path in enumerate(frame_paths, start=1):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read frame, skip: {path}")
            continue

        # If size is inconsistent, resize to (w, h)
        # 如遇尺寸不一致，统一缩放到第一帧大小
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        writer.write(img)

        if i % 50 == 0 or i == len(frame_paths):
            print(f"[INFO] Written {i}/{len(frame_paths)} frames")

    writer.release()
    print(f"[OK] Saved video: {out_video}  (fps={fps}, frames={len(frame_paths)})")


def parse_args():
    """
    Parse command-line arguments.
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert an image sequence (e.g. VisDrone frames) into a video.\n"
            "将帧序列（如 VisDrone 数据集）合成为一个视频文件。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--seq-dir",
        type=str,
        required=True,
        help="Directory containing frame images "
             "(e.g. data/visdrone_frames/uav0000126_00001_v). "
             "帧图像所在目录，例如 data/visdrone_frames/uav0000126_00001_v。"
    )

    parser.add_argument(
        "--out-video",
        type=str,
        required=True,
        help="Output video path, e.g. data/videos/demo.mp4. "
             "输出视频路径，例如 data/videos/demo.mp4。"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second of the output video. "
             "输出视频帧率（单位：帧/秒）。"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Glob pattern for frame images, e.g. '*.jpg' or '*.png'. "
             "帧图像的通配符模式，例如 '*.jpg' 或 '*.png'。"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames_to_video(
        seq_dir=args.seq_dir,
        out_video=args.out_video,
        fps=args.fps,
        pattern=args.pattern
    )
