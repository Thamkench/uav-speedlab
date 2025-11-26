# scripts/run_speed.py
# -*- coding: utf-8 -*-
"""
CLI entry for speed estimation pipeline.
测速流水线命令行入口。
"""

import os
import argparse
from src.pipeline.speed_pipeline import SpeedPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO-based speed estimation pipeline on a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path. 输入视频路径。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/speed_config.yaml",
        help="Config file path. 配置文件路径。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output video path; if empty, use <name>_speed.mp4. "
             "输出视频路径，留空则自动生成 <原名>_speed.mp4。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vpath = args.video
    assert os.path.isfile(vpath), f"Video not found: {vpath}"

    if args.out:
        out_path = args.out
    else:
        d, n = os.path.split(vpath)
        stem, _ = os.path.splitext(n)
        out_path = os.path.join(d, f"{stem}_speed.mp4")

    pipe = SpeedPipeline(args.config)
    pipe.run(vpath, out_path)
