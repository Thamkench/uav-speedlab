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
from src.motion.weighted_speed_smoother import WeightedSpeedParams, WeightedSpeedSmoother


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

        # --- Weighted speed smoothing (confidence-weighted) / 置信度加权速度平滑 ---
        self.ws_params = WeightedSpeedParams.from_cfg(self.cfg)
        self.ws_smoother = WeightedSpeedSmoother(self.ws_params) if self.ws_params.enabled else None

        # --- Debug dump config / 调试导出配置（默认关闭，不影响性能）---
        ws_cfg = self.cfg.get("weighted_speed", {}) or {}
        self.ws_debug_cfg = ws_cfg.get("debug", {}) or {}
        self.ws_debug_on = bool(self.ws_debug_cfg.get("enabled", False))
        self.ws_debug_every = int(self.ws_debug_cfg.get("dump_every", 1))

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
        import os
        import cv2
        import numpy as np
        from typing import Dict

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        # YOLO tracking stream
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

        # ==== State machine & caches (keep old behavior) ====
        static_cnt: Dict[int, int] = {}
        moving_cnt: Dict[int, int] = {}
        state: Dict[int, str] = {}  # 'static' / 'moving' / 'unknown'
        mpp_ema: Dict[int, float] = {}
        speed_ema: Dict[int, float] = {}  # OLD display speed state (m/s)  <-- keep unchanged for overlay

        # ==== NEW: WS parallel speed state (m/s) for closed-loop comparison ====
        speed_ema_ws: Dict[int, float] = {}  # NEW ws-logic closed-loop state (m/s), NOT used for overlay

        last_boxes = None
        last_gray = None
        last_idx = None
        last_H = None

        orb = cv2.ORB_create(nfeatures=self.nfeatures)

        # --- Debug rows ---
        debug_rows = []
        dbg_cnt = 0

        global_idx = -1
        # --- Homography quality cache (for whomo) ---
        homo_inlier_ratio = None
        homo_inlier_count = None

        for r in stream:
            global_idx += 1
            frame = r.orig_img  # BGR
            vis = frame.copy()

            names = getattr(r, "names", None) or getattr(self.model, "names", {})

            # current boxes
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

            # ================= Homography + state/speed update =================
            need_pair = (last_idx is None) or ((global_idx - last_idx) >= self.delta_frames)

            if need_pair and last_gray is not None and last_boxes is not None:
                # background masks
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

                # ORB features
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
                        pts_prev = np.float32([kp1[m.queryIdx].pt for m in good])
                        pts_now = np.float32([kp2[m.trainIdx].pt for m in good])
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
                            last_H = Hmat
                            inl_cnt = int(inl.sum())
                            good_cnt = max(len(good), 1)
                            homo_inlier_ratio = inl_cnt / good_cnt
                            homo_inlier_count = inl_cnt

                if not H_ok and last_H is None:
                    # no H available: refresh keyframe cache only
                    last_gray = gray_now
                    last_boxes = {"boxes": boxes, "clses": clses, "ids": ids}
                    last_idx = global_idx
                else:
                    if not H_ok:
                        Hmat = last_H
                        homo_inlier_ratio = 0.0
                        homo_inlier_count = 0

                    gray_prev_warp = cv2.warpPerspective(last_gray, Hmat, (W, H))
                    R = cv2.absdiff(
                        cv2.GaussianBlur(gray_now, (5, 5), 0),
                        cv2.GaussianBlur(gray_prev_warp, (5, 5), 0),
                    )

                    # centers on last keyframe
                    prev_centers: Dict[int, tuple[float, float]] = {}
                    for (b, tid, c) in zip(last_boxes["boxes"], last_boxes["ids"], last_boxes["clses"]):
                        if int(c) not in self.vehicle_ids:
                            continue
                        cx = 0.5 * (b[0] + b[2])
                        cy = 0.5 * (b[1] + b[3])
                        prev_centers[int(tid)] = (cx, cy)

                    dt = max(1e-6, float(global_idx - last_idx) / float(fps))

                    # update for ids present in both frames
                    for (b, tid, c) in zip(boxes, ids, clses):
                        tid = int(tid)
                        if int(c) not in self.vehicle_ids:
                            continue
                        if tid not in prev_centers:
                            continue

                        # compensated displacement
                        cx_now = 0.5 * (b[0] + b[2])
                        cy_now = 0.5 * (b[1] + b[3])
                        prev_pt = np.array([[prev_centers[tid][0], prev_centers[tid][1], 1.0]], dtype=np.float32).T
                        proj = Hmat @ prev_pt
                        proj /= proj[2] + 1e-9
                        cx_proj, cy_proj = float(proj[0]), float(proj[1])
                        d = float(np.hypot(cx_now - cx_proj, cy_now - cy_proj))

                        # photometric residual
                        er = erode_bbox(b, 0.15, W, H)
                        r_mean = 999.0
                        if er is not None:
                            xa, ya, xb, yb = er
                            roi = R[ya:yb, xa:xb]
                            if roi.size > 0:
                                r_mean = float(roi.mean())

                        # ======= static/moving FSM (old) =======
                        static_cnt.setdefault(tid, 0)
                        moving_cnt.setdefault(tid, 0)
                        state.setdefault(tid, "unknown")

                        is_static_ev = (d <= self.D_STATIC_PX) and (r_mean <= self.R_STATIC_MEAN)
                        is_moving_ev = (d >= self.D_MOVING_PX) or (r_mean >= self.R_MOVING_MEAN)

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

                        # ======= mpp =======
                        w_box = float(b[2] - b[0])
                        h_box = float(b[3] - b[1])
                        long_edge = max(w_box, h_box)
                        if long_edge <= max(1e-6, self.MIN_LONG_EDGE):
                            continue

                        cname = names.get(int(c), str(int(c)))
                        length_m = self._get_vehicle_length_m(cname)

                        mpp_now = float(np.clip(length_m / long_edge, self.MIN_MPP, self.MAX_MPP))
                        if tid in mpp_ema:
                            mpp_ema[tid] = self.EMA_ALPHA_MPP * mpp_now + (1 - self.EMA_ALPHA_MPP) * mpp_ema[tid]
                        else:
                            mpp_ema[tid] = mpp_now

                        # ======= speed update (only non-static if SHOW_MOVING_ONLY) =======
                        if (not self.SHOW_MOVING_ONLY) or (state.get(tid) != "static"):
                            v_mps = (d * mpp_ema[tid]) / dt
                            if v_mps <= self.MAX_MPS_CLAMP:
                                # ---------- raw / old / ema (km/h) ----------
                                v_raw_kmh = float(v_mps * 3.6)

                                # OLD displayed state (m/s)
                                v_old_mps = float(speed_ema.get(tid, v_mps))
                                v_old_kmh = float(v_old_mps * 3.6)

                                # old EMA prediction (for logging only)
                                if tid in speed_ema:
                                    v_ema_mps = self.EMA_ALPHA_SPEED * v_mps + (1 - self.EMA_ALPHA_SPEED) * speed_ema[
                                        tid]
                                else:
                                    v_ema_mps = v_mps
                                v_ema_kmh = float(v_ema_mps * 3.6)

                                # ---------- WS components & WS "measurement" ----------
                                v_ws_kmh = None
                                v_ws_mps_meas = None
                                v_ws_ema_kmh = None

                                wb = we = wd = wh = W_raw = W_final = None
                                W_use = None

                                if self.ws_smoother is not None:
                                    cx = 0.5 * (b[0] + b[2])
                                    cy = 0.5 * (b[1] + b[3])
                                    wb, we, wd, wh, W_raw, W_final = self.ws_smoother.compute_components(
                                        tid=tid,
                                        bbox_w=w_box,
                                        bbox_h=h_box,
                                        cx=cx, cy=cy,
                                        imgW=W, imgH=H,
                                        v_meas_kmh=v_raw_kmh,
                                        v_smooth_kmh=v_old_kmh,  # keep reference as OLD smooth for fair comparison
                                        inlier_ratio=homo_inlier_ratio,
                                        inlier_count=homo_inlier_count,
                                    )

                                    alpha0 = float(self.EMA_ALPHA_SPEED)  # cap
                                    W_use = min(float(W_final), alpha0)

                                    # WS "measurement" (km/h) using capped weight
                                    v_ws_kmh = float(W_use * v_raw_kmh + (1.0 - W_use) * v_old_kmh)
                                    v_ws_mps_meas = float(v_ws_kmh / 3.6)

                                # ---------- Update OLD logic speed_ema (MUST remain unchanged for video overlay) ----------
                                if tid in speed_ema:
                                    speed_ema[tid] = self.EMA_ALPHA_SPEED * v_mps + (1 - self.EMA_ALPHA_SPEED) * \
                                                     speed_ema[tid]
                                else:
                                    speed_ema[tid] = v_mps

                                # ---------- Update NEW WS closed-loop state (for true comparison; NOT used in overlay) ----------
                                if self.ws_smoother is not None and v_ws_mps_meas is not None:
                                    if tid in speed_ema_ws:
                                        speed_ema_ws[tid] = self.EMA_ALPHA_SPEED * v_ws_mps_meas + (
                                                    1 - self.EMA_ALPHA_SPEED) * speed_ema_ws[tid]
                                    else:
                                        speed_ema_ws[tid] = v_ws_mps_meas
                                    v_ws_ema_kmh = float(speed_ema_ws[tid] * 3.6)

                                # ---------- Debug row ----------
                                if self.ws_debug_on:
                                    dbg_cnt += 1
                                    if self.ws_debug_every <= 1 or (dbg_cnt % self.ws_debug_every == 0):
                                        debug_rows.append({
                                            "frame": global_idx,
                                            "dt": dt,
                                            "class_id": int(c),
                                            "class_name": cname,
                                            "tid": int(tid),
                                            "state": state.get(tid, "unknown"),
                                            "d_px": float(d),
                                            "long_edge_px": float(long_edge),
                                            "mpp": float(mpp_ema.get(tid, 0.0)),

                                            "v_raw_kmh": v_raw_kmh,
                                            "v_old_kmh": v_old_kmh,
                                            "v_ema_kmh": v_ema_kmh,

                                            # ws instantaneous (measurement-like)
                                            "v_ws_kmh": v_ws_kmh,

                                            # ws closed-loop state (THIS is the real comparison curve)
                                            "v_ws_ema_kmh": v_ws_ema_kmh,

                                            "wbbox": wb,
                                            "wedge": we,
                                            "wdv": wd,
                                            "whomo": wh,
                                            "W_raw": W_raw,
                                            "W_final": W_final,
                                            "W_use": W_use,
                                        })

                    # update keyframe cache
                    last_gray = gray_now
                    last_boxes = {"boxes": boxes, "clses": clses, "ids": ids}
                    last_idx = global_idx

            # init first keyframe
            if last_idx is None:
                last_gray = gray_now
                last_boxes = {"boxes": boxes, "clses": clses, "ids": ids}
                last_idx = global_idx

            # ================= overlay (KEEP OLD VIDEO DISPLAY UNCHANGED) =================
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clses):
                tid = int(tid)
                cname = names.get(int(c), str(int(c)))

                if int(c) not in self.vehicle_ids:
                    continue

                st = state.get(tid, "unknown")
                scnt = static_cnt.get(tid, 0)
                has_speed = tid in speed_ema  # OLD has-speed

                if st == "static" and scnt >= self.K_STATIC:
                    color = COLOR_STATIC
                    thickness = 2
                    label = "STATIC"
                elif has_speed:
                    color = COLOR_MOVING
                    thickness = 2
                    kmh = float(speed_ema[tid] * 3.6)  # OLD display (unchanged)
                    label = f"{kmh:4.1f} km/h"
                else:
                    color = COLOR_NO_RESULT
                    thickness = 1
                    label = f"{cname} ID {tid}"

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

        # ---------- Dump debug logs ----------
        if self.ws_debug_on and len(debug_rows) > 0:
            import pandas as pd
            out_dir = str(self.ws_debug_cfg.get("out_dir", "runs/debug"))
            out_name = str(self.ws_debug_cfg.get("out_name", "speed_debug.csv"))
            export_excel = bool(self.ws_debug_cfg.get("export_excel", False))

            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, out_name)

            df = pd.DataFrame(debug_rows)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            if export_excel:
                xlsx_path = csv_path[:-4] + ".xlsx" if csv_path.lower().endswith(".csv") else (csv_path + ".xlsx")
                df.to_excel(xlsx_path, index=False)

            print(f"[DEBUG] Saved speed debug logs: {csv_path}")

        print(f"[OK] Saved speed-annotated video: {out_path}")

