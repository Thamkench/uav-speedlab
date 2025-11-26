# src/pipeline/speed_pipeline.py
# -*- coding: utf-8 -*-
"""
YOLO 跟踪 + 背景 Homography 补偿 + 静止/运动判定 + 简易速度估计

"""

from __future__ import annotations
from typing import Dict, Any, Set

import os
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.loader import load_speed_config, resolve_dynamic_ids
from src.vis.draw import draw_label


# ===== 颜色（BGR）=====
# 运动：霓虹青蓝（更亮、更通透）
COLOR_MOVING    = (255, 210,  60)   # cyan-ish, high luminance

# 静止：霓虹橙黄（更亮）
COLOR_STATIC    = (  0, 220, 255)   # vivid orange/yellow

# 未判定：浅灰蓝（更轻）
COLOR_NO_RESULT = (220, 220, 220)


def make_bg_mask(
    shape,
    boxes: np.ndarray,
    clses: np.ndarray,
    vehicle_ids: Set[int],
    expand_ratio: float,
):
    """把动态物体周围抠掉，仅保留背景，用于求 H。"""
    h, w = shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    if boxes is None or len(boxes) == 0:
        return mask

    for (x1, y1, x2, y2), cid in zip(boxes, clses):
        if int(cid) not in vehicle_ids:
            continue
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * expand_ratio, bh * expand_ratio
        xa, ya = max(0, int(x1 - dx)), max(0, int(y1 - dy))
        xb, yb = min(w - 1, int(x2 + dx)), min(h - 1, int(y2 + dy))
        cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)
    return mask


def erode_bbox(b, ratio: float, W: int, H: int):
    """把框向内收缩一点，避免边缘噪声。"""
    x1, y1, x2, y2 = map(int, b)
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * ratio), int(h * ratio)
    xa, ya = max(0, x1 + dx), max(0, y1 + dy)
    xb, yb = min(W - 1, x2 - dx), min(H - 1, y2 - dy)
    if xb <= xa or yb <= ya:
        return None
    return xa, ya, xb, yb


class SpeedPipeline:
    def __init__(self, cfg_path: str) -> None:
        self.cfg = load_speed_config(cfg_path)

        model_cfg = self.cfg["model"]
        track_cfg = self.cfg["tracking"]
        speed_cfg = self.cfg.get("speed", {}) or {}
        static_cfg = self.cfg.get("static_detection", {}) or {}
        homo_cfg = self.cfg.get("homography", {}) or {}
        dyn_cfg = self.cfg.get("dynamic_classes", {}) or {}

        # --- YOLO & 跟踪配置 ---
        self.model_path: str = model_cfg["weights"]
        self.imgsz: int = model_cfg.get("imgsz", 960)
        self.device: str | int = model_cfg.get("device", 0)

        self.tracker_cfg: str = track_cfg["tracker_cfg"]
        self.conf: float = track_cfg.get("conf", 0.3)

        self.model = YOLO(self.model_path)
        self.vehicle_ids: Set[int] = resolve_dynamic_ids(
            self.cfg, self.model_path
        )

        # --- Homography 参数（ORB + RANSAC）---
        self.nfeatures = homo_cfg.get("nfeatures", 2000)
        self.ratio_thresh = homo_cfg.get("ratio_thresh", 0.8)
        self.reproj_thresh = homo_cfg.get("reproj_thresh", 3.0)
        self.max_iters = homo_cfg.get("max_iters", 2000)
        self.confidence = homo_cfg.get("confidence", 0.995)
        self.delta_frames = homo_cfg.get("delta_frames", 5)

        # --- 静止判定阈值（几何 + 亮度 + 迟滞）---
        self.D_STATIC_PX = static_cfg.get("d_static_px", 2.5)
        self.D_MOVING_PX = static_cfg.get("d_moving_px", 5.0)
        self.R_STATIC_MEAN = static_cfg.get("r_static_mean", 12.0)
        self.R_MOVING_MEAN = static_cfg.get("r_moving_mean", 25.0)
        self.K_STATIC = static_cfg.get("k_static", 6)
        self.K_MOVING = static_cfg.get("k_moving", 2)

        # --- 速度估计参数（与旧脚本一致，新增“按类别车长”支持）---
        # 兼容旧配置：若没有 default_length_m，则退回 car_length_m
        self.CAR_LEN_M = speed_cfg.get("car_length_m", 5.0)
        self.DEFAULT_LEN_M = speed_cfg.get("default_length_m", self.CAR_LEN_M)
        self.CLASS_LEN_M = speed_cfg.get("class_length_m", {})

        self.EMA_ALPHA_MPP = speed_cfg.get("ema_alpha_mpp", 0.30)
        self.EMA_ALPHA_SPEED = speed_cfg.get("ema_alpha_speed", 0.60)
        self.MIN_MPP = speed_cfg.get("min_mpp", 0.005)
        self.MAX_MPP = speed_cfg.get("max_mpp", 0.5)
        self.MAX_MPS_CLAMP = speed_cfg.get("max_speed_mps", 60.0)
        self.SHOW_MOVING_ONLY = speed_cfg.get("show_moving_only", True)
        self.MIN_LONG_EDGE = speed_cfg.get("min_long_edge_px", 1)

        # --- 动态类 bbox 外扩比例，用于抠掉前景 ---
        self.expand_ratio: float = dyn_cfg.get("expand_ratio", 0.03)

    # ------------------------------------------------------------------
    # 工具：根据类别名称返回近似车长（米），优先用 class_length_m，退回 default_length_m。
    # ------------------------------------------------------------------
    def _get_vehicle_length_m(self, cls_name: str) -> float:
        """
        根据类别名称返回近似车长（米）。
        优先使用配置中的 speed.class_length_m[cls_name]，
        若未配置则退回 speed.default_length_m（或 car_length_m）。
        """
        length_map = self.CLASS_LEN_M if isinstance(self.CLASS_LEN_M, dict) else {}
        if cls_name in length_map:
            try:
                return float(length_map[cls_name])
            except Exception:
                pass
        # fallback: 默认长度
        try:
            return float(self.DEFAULT_LEN_M)
        except Exception:
            return float(self.CAR_LEN_M)

    def run(self, video_path: str, out_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vw = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )

        # YOLO 跟踪流
        stream = self.model.track(
            source=video_path,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker_cfg,
            stream=True,
            verbose=False,
            persist=True,
        )

        # ==== 状态机 & 速度缓存（完全沿用你旧脚本）====
        static_cnt: Dict[int, int] = {}
        moving_cnt: Dict[int, int] = {}
        state: Dict[int, str] = {}  # 'static' / 'moving' / 'unknown'
        mpp_ema: Dict[int, float] = {}
        speed_ema: Dict[int, float] = {}  # m/s

        last_boxes = None
        last_gray = None
        last_idx = None
        last_H = None

        orb = cv2.ORB_create(nfeatures=self.nfeatures)

        global_idx = -1
        for r in stream:
            global_idx += 1
            frame = r.orig_img  # BGR
            vis = frame.copy()

            # 当前模型类别名称映射（id → name）
            names = getattr(r, "names", None) or getattr(self.model, "names", {})

            # 当前帧检测/跟踪结果
            if r.boxes is not None and len(r.boxes) > 0:
                bb = r.boxes
                boxes = bb.xyxy.cpu().numpy()
                clses = bb.cls.cpu().numpy().astype(int)
                ids = (
                    bb.id.cpu().numpy()
                    if bb.id is not None
                    else np.full(len(bb), -1)
                ).astype(int)
            else:
                boxes = np.zeros((0, 4), dtype=float)
                clses = np.zeros((0,), dtype=int)
                ids = np.zeros((0,), dtype=int)

            gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ================= Homography + 状态/速度更新 =================
            need_pair = (last_idx is None) or (
                (global_idx - last_idx) >= self.delta_frames
            )

            if need_pair and last_gray is not None and last_boxes is not None:
                # 背景 mask（上一关键帧 vs 当前帧）
                mask_prev = make_bg_mask(
                    (H, W, 3),
                    last_boxes["boxes"],
                    last_boxes["clses"],
                    self.vehicle_ids,
                    self.expand_ratio,
                )
                mask_now = make_bg_mask(
                    (H, W, 3),
                    boxes,
                    clses,
                    self.vehicle_ids,
                    self.expand_ratio,
                )

                # ORB 特征
                kp1, des1 = orb.detectAndCompute(last_gray, mask_prev)
                kp2, des2 = orb.detectAndCompute(gray_now, mask_now)

                Hmat = None
                H_ok = False
                if (
                    des1 is not None
                    and des2 is not None
                    and len(kp1) >= 20
                    and len(kp2) >= 20
                ):
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    knn = bf.knnMatch(des1, des2, k=2)
                    good = []
                    for m, n in knn:
                        if m.distance < self.ratio_thresh * n.distance:
                            good.append(m)
                    if len(good) >= 30:
                        pts_prev = np.float32(
                            [kp1[m.queryIdx].pt for m in good]
                        )
                        pts_now = np.float32(
                            [kp2[m.trainIdx].pt for m in good]
                        )
                        Hmat, inl = cv2.findHomography(
                            pts_prev,
                            pts_now,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=self.reproj_thresh,
                            maxIters=self.max_iters,
                            confidence=self.confidence,
                        )
                        if (
                            Hmat is not None
                            and inl is not None
                            and inl.sum() >= 0.5 * len(good)
                        ):
                            H_ok = True
                            last_H = Hmat  # 记录一个“备用 H”

                if not H_ok and last_H is None:
                    # 本次失败且没有备用 H：只更新关键帧缓存，跳过判定
                    last_gray = gray_now
                    last_boxes = {
                        "boxes": boxes,
                        "clses": clses,
                        "ids": ids,
                    }
                    last_idx = global_idx
                else:
                    if not H_ok:
                        Hmat = last_H  # 回退用旧 H

                    # 亮度残差图（把 prev warp 到 now）
                    gray_prev_warp = cv2.warpPerspective(last_gray, Hmat, (W, H))
                    R = cv2.absdiff(
                        cv2.GaussianBlur(gray_now, (5, 5), 0),
                        cv2.GaussianBlur(gray_prev_warp, (5, 5), 0),
                    )

                    # 上一关键帧的车辆中心
                    prev_centers: Dict[int, tuple[float, float]] = {}
                    for (b, tid, c) in zip(
                        last_boxes["boxes"],
                        last_boxes["ids"],
                        last_boxes["clses"],
                    ):
                        if int(c) not in self.vehicle_ids:
                            continue
                        cx = 0.5 * (b[0] + b[2])
                        cy = 0.5 * (b[1] + b[3])
                        prev_centers[int(tid)] = (cx, cy)

                    dt = max(
                        1e-6, float(global_idx - last_idx) / float(fps)
                    )  # 关键帧间时间

                    # 当前帧：对同时出现的 ID 做判定 + 速度
                    for (b, tid, c) in zip(boxes, ids, clses):
                        tid = int(tid)
                        if int(c) not in self.vehicle_ids:
                            continue
                        if tid not in prev_centers:
                            continue  # 新出现/刚恢复

                        # 几何位移（相机补偿后）
                        cx_now = 0.5 * (b[0] + b[2])
                        cy_now = 0.5 * (b[1] + b[3])
                        prev_pt = np.array(
                            [
                                [
                                    prev_centers[tid][0],
                                    prev_centers[tid][1],
                                    1.0,
                                ]
                            ],
                            dtype=np.float32,
                        ).T
                        proj = Hmat @ prev_pt
                        proj /= proj[2] + 1e-9
                        cx_proj, cy_proj = float(proj[0]), float(proj[1])
                        d = float(
                            np.hypot(cx_now - cx_proj, cy_now - cy_proj)
                        )  # px

                        # 亮度残差
                        er = erode_bbox(b, 0.15, W, H)
                        r_mean = 999.0
                        if er is not None:
                            xa, ya, xb, yb = er
                            roi = R[ya:yb, xa:xb]
                            if roi.size > 0:
                                r_mean = float(roi.mean())

                        # ======= 静/动状态机（沿用旧脚本）=======
                        static_cnt.setdefault(tid, 0)
                        moving_cnt.setdefault(tid, 0)
                        state.setdefault(tid, "unknown")

                        is_static_ev = (d <= self.D_STATIC_PX) and (
                            r_mean <= self.R_STATIC_MEAN
                        )
                        is_moving_ev = (d >= self.D_MOVING_PX) or (
                            r_mean >= self.R_MOVING_MEAN
                        )

                        if is_static_ev:
                            static_cnt[tid] += 1
                            moving_cnt[tid] = 0
                        elif is_moving_ev:
                            moving_cnt[tid] += 1
                            static_cnt[tid] = 0
                        else:
                            static_cnt[tid] = max(0, static_cnt[tid] - 1)
                            moving_cnt[tid] = max(0, moving_cnt[tid] - 1)

                        if static_cnt[tid] >= self.K_STATIC:
                            state[tid] = "static"
                        elif moving_cnt[tid] >= self.K_MOVING:
                            state[tid] = "moving"

                        # ======= 简易速度（车长→mpp→v，支持按类别车长）=======
                        w_box = float(b[2] - b[0])
                        h_box = float(b[3] - b[1])
                        long_edge = max(w_box, h_box)
                        if long_edge <= max(1e-6, self.MIN_LONG_EDGE):
                            continue

                        # 按类别名称取车长（米），car/van/truck/bus 可不同
                        cname = names.get(int(c), str(int(c)))
                        length_m = self._get_vehicle_length_m(cname)

                        mpp_now = length_m / long_edge
                        mpp_now = float(
                            np.clip(mpp_now, self.MIN_MPP, self.MAX_MPP)
                        )
                        if tid in mpp_ema:
                            mpp_ema[tid] = (
                                self.EMA_ALPHA_MPP * mpp_now
                                + (1 - self.EMA_ALPHA_MPP) * mpp_ema[tid]
                            )
                        else:
                            mpp_ema[tid] = mpp_now

                        # 只对非静止车辆更新速度
                        if (not self.SHOW_MOVING_ONLY) or (
                            state.get(tid) != "static"
                        ):
                            v_mps = (d * mpp_ema[tid]) / dt
                            if v_mps <= self.MAX_MPS_CLAMP:
                                if tid in speed_ema:
                                    speed_ema[tid] = (
                                        self.EMA_ALPHA_SPEED * v_mps
                                        + (1 - self.EMA_ALPHA_SPEED)
                                        * speed_ema[tid]
                                    )
                                else:
                                    speed_ema[tid] = v_mps

                    # 更新关键帧缓存
                    last_gray = gray_now
                    last_boxes = {
                        "boxes": boxes,
                        "clses": clses,
                        "ids": ids,
                    }
                    last_idx = global_idx

            # 初始化第一帧关键帧
            if last_idx is None:
                last_gray = gray_now
                last_boxes = {"boxes": boxes, "clses": clses, "ids": ids}
                last_idx = global_idx

            # ================= 统一画框 + 速度 + STATIC =================
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clses):
                tid = int(tid)
                cname = names.get(int(c), str(int(c)))

                # 只对需要测速的类别画（car/bus/truck 等）
                if int(c) not in self.vehicle_ids:
                    continue

                st = state.get(tid, "unknown")
                scnt = static_cnt.get(tid, 0)
                has_speed = tid in speed_ema  # 是否已经算出速度

                # ===== 状态 1：静止 =====
                if st == "static" and scnt >= self.K_STATIC:
                    color = COLOR_STATIC
                    thickness = 2
                    label = "STATIC"  # 只显示 STATIC，不显示 id/类别

                # ===== 状态 2：有速度（认为在动）=====
                elif has_speed:
                    color = COLOR_MOVING
                    thickness = 2
                    kmh = float(speed_ema[tid] * 3.6)
                    label = f"{kmh:4.1f} km/h"  # 只显示速度

                # ===== 状态 3：还没出结果 / 正在等待 =====
                else:
                    color = COLOR_NO_RESULT
                    thickness = 1
                    label = f"{cname} ID {tid}"  # 灰色显示类别+ID

                # 画框 + label
                cv2.rectangle(
                    vis,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                draw_label(vis, int(x1), int(y1), label, color)

            vw.write(vis)

        vw.release()
        print(f"[OK] Saved speed-annotated video: {out_path}")
