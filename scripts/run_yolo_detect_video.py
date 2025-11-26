# scripts/run_yolo_detect_video.py
# -*- coding: utf-8 -*-
"""
YOLOv11-L video inference demo
YOLOv11-L 视频检测可视化脚本

- Input:  a video file (e.g. VisDrone-converted mp4)
- Output: a new video with detection boxes and labels overlaid

- 输入：  一个视频文件（例如由 VisDrone 帧合成的 mp4）
- 输出：  叠加检测框和类别标签的可视化视频

Usage / 用法示例：

    python scripts/run_yolo_detect_video.py \
        --video data/videos/uav0000126_00001_v.mp4 \
        --weights weights/yolo11l.pt \
        --out data/videos/uav0000126_00001_v_det.mp4 \
        --conf 0.3 \
        --imgsz 960 \
        --device 0
"""

import os
import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    """
    Parse command-line arguments.
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv11 video inference and save an annotated video.\n"
            "对输入视频运行 YOLOv11 检测，并保存带可视化框的视频。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video, e.g. data/videos/demo.mp4. "
             "输入视频路径，例如 data/videos/demo.mp4。"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="./weights/yolo11l.pt",
        help="Path to YOLOv11 weights (.pt). "
             "YOLOv11 权重文件路径（.pt）。"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Path to save annotated video. If empty, auto-generate "
             "<video_name>_det.mp4 in the same directory. "
             "输出视频路径，留空则自动在同目录下生成 <原名>_det.mp4。"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detections. "
             "检测置信度阈值。"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (shorter side). "
             "推理输入尺寸（短边）。"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run on: GPU id like '0' or 'cpu'. "
             "推理设备：GPU 编号如 '0' 或 'cpu'。"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    weight_path = args.weights

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found / 视频不存在: {video_path}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weights not found / 权重不存在: {weight_path}")

    # --------- Determine output path / 确定输出路径 ---------
    if args.out:
        out_path = args.out
    else:
        vdir, vname = os.path.split(video_path)
        stem, _ = os.path.splitext(vname)
        out_path = os.path.join(vdir, f"{stem}_det.mp4")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[INFO] Input video : {video_path}")
    print(f"[INFO] Weights     : {weight_path}")
    print(f"[INFO] Output video: {out_path}")

    # --------- Open video to get FPS & size / 打开视频以获取帧率和尺寸 ---------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video / 无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video FPS  : {fps}")
    print(f"[INFO] Resolution : {width}x{height}")
    print(f"[INFO] Total frames (approx): {total_frames}")

    cap.release()  # We will let Ultralytics read frames / 后面让 Ultralytics 自己读

    # --------- Initialize VideoWriter / 初始化视频写入器 ---------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter / 无法创建输出视频: {out_path}")

    # --------- Load YOLO model / 加载 YOLO 模型 ---------
    print("[INFO] Loading YOLO model ... / 正在加载 YOLO 模型 ...")
    model = YOLO(weight_path)

    # --------- Run prediction in streaming mode / 以流模式运行推理 ---------
    print("[INFO] Running inference ... / 开始对视频逐帧推理 ...")

    # stream=True: 返回一个逐帧的 generator，每一帧是一个 Results 对象
    # stream=True: returns generator of Results for each frame
    results = model.predict(
        source=video_path,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False
    )

    frame_idx = 0
    for r in results:
        # r.orig_img: original BGR image
        # r.plot():   BGR image with boxes & labels drawn
        # r.orig_img 是原图（BGR），r.plot() 是叠加了检测框和标签的图像
        annotated = r.plot()  # numpy array, BGR

        # 保证尺寸与 VideoWriter 一致（理论上 r.plot() 就是原尺寸）
        if annotated.shape[1] != width or annotated.shape[0] != height:
            annotated = cv2.resize(annotated, (width, height))

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames ... / 已处理 {frame_idx} 帧 ...")

    writer.release()
    print(f"[OK] Saved annotated video to / 已保存检测可视化视频: {out_path}")


if __name__ == "__main__":
    main()
