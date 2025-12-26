# src/pipeline/speed_pipeline.py
# -*- coding: utf-8 -*-
"""
YOLO tracking + background homography compensation + static/moving FSM + speed estimation

- Baseline: fixed EMA (alpha = speed.ema_alpha_speed)
- Optional: confidence-weighted smoothing (weighted_speed.enabled)
- Optional: diagnostics export (diagnostics.export.enabled)
"""

from __future__ import annotations
from typing import Dict, Set, Optional, Tuple

import os
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.loader import load_speed_config, resolve_dynamic_ids
from src.vis.draw import draw_label
from src.motion.weighted_speed_smoother import WeightedSpeedParams, WeightedSpeedSmoother


# ===== Colors (BGR) =====
COLOR_MOVING = (255, 210, 60)
COLOR_STATIC = (0, 220, 255)
COLOR_NO_RESULT = (220, 220, 220)


def make_bg_mask(
    shape,
    boxes: np.ndarray,
    clses: np.ndarray,
    vehicle_ids: Set[int],
    expand_ratio: float,
) -> np.ndarray:
    """Mask out dynamic objects, keep background only for homography estimation."""
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


def erode_bbox(b: np.ndarray, ratio: float, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    """Shrink bbox to avoid edge noise for photometric residual."""
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

        model_cfg = self.cfg.get("model", {}) or {}
        track_cfg = self.cfg.get("tracking", {}) or {}
        speed_cfg = self.cfg.get("speed", {}) or {}
        static_cfg = self.cfg.get("static_detection", {}) or {}
        dyn_cfg = self.cfg.get("dynamic_classes", {}) or {}

        # --- YOLO & tracking ---
        self.model_path: str = model_cfg["weights"]
        self.imgsz: int = int(model_cfg.get("imgsz", 960))
        self.device: str | int = model_cfg.get("device", 0)

        self.tracker_cfg: str = track_cfg["tracker_cfg"]
        self.conf: float = float(track_cfg.get("conf", 0.3))

        self.model = YOLO(self.model_path)
        self.vehicle_ids: Set[int] = resolve_dynamic_ids(self.cfg, self.model_path)

        # --- Homography params (STRICT: only read from `homography`) ---
        homo_cfg = self.cfg.get("homography", {}) or {}
        self.nfeatures = int(homo_cfg.get("nfeatures", 2000))
        self.ratio_thresh = float(homo_cfg.get("ratio_thresh", 0.8))
        self.reproj_thresh = float(homo_cfg.get("reproj_thresh", 3.0))
        self.max_iters = int(homo_cfg.get("max_iters", 2000))
        self.confidence = float(homo_cfg.get("confidence", 0.995))
        self.delta_frames = int(homo_cfg.get("delta_frames", 5))

        # --- Static detection thresholds ---
        self.D_STATIC_PX = float(static_cfg.get("d_static_px", 2.5))
        self.D_MOVING_PX = float(static_cfg.get("d_moving_px", 5.0))
        self.R_STATIC_MEAN = float(static_cfg.get("r_static_mean", 12.0))
        self.R_MOVING_MEAN = float(static_cfg.get("r_moving_mean", 25.0))
        self.K_STATIC = int(static_cfg.get("k_static", 6))
        self.K_MOVING = int(static_cfg.get("k_moving", 2))

        # --- Speed params ---
        self.CAR_LEN_M = float(speed_cfg.get("car_length_m", 5.0))  # legacy fallback
        self.DEFAULT_LEN_M = float(speed_cfg.get("default_length_m", self.CAR_LEN_M))
        self.CLASS_LEN_M = speed_cfg.get("class_length_m", {}) or {}

        self.EMA_ALPHA_MPP = float(speed_cfg.get("ema_alpha_mpp", 0.30))
        self.EMA_ALPHA_SPEED = float(speed_cfg.get("ema_alpha_speed", 0.60))
        self.MIN_MPP = float(speed_cfg.get("min_mpp", 0.005))
        self.MAX_MPP = float(speed_cfg.get("max_mpp", 0.5))
        self.MAX_MPS_CLAMP = float(speed_cfg.get("max_speed_mps", 60.0))
        self.SHOW_MOVING_ONLY = bool(speed_cfg.get("show_moving_only", True))
        self.MIN_LONG_EDGE = float(speed_cfg.get("min_long_edge_px", 1))

        self.expand_ratio: float = float(dyn_cfg.get("expand_ratio", 0.03))

        # --- Weighted speed smoothing ---
        self.ws_params = WeightedSpeedParams.from_cfg(self.cfg)
        self.ws_smoother = WeightedSpeedSmoother(self.ws_params) if self.ws_params.enabled else None

        # --- Diagnostics export (new config) ---
        diag_cfg = (self.cfg.get("diagnostics", {}) or {}).get("export", {}) or {}
        self.diag_export_on = bool(diag_cfg.get("enabled", False))
        self.diag_out_dir = str(diag_cfg.get("out_dir", "outputs/diagnostics"))
        self.diag_prefix = str(diag_cfg.get("file_prefix", "speed_debug"))  # 建议你在 config 里设成 speed_debug
        self.diag_format = str(diag_cfg.get("format", "xlsx")).lower().strip()
        if self.diag_format not in ("csv", "xlsx"):
            self.diag_format = "xlsx"

    def _get_vehicle_length_m(self, cls_name: str) -> float:
        length_map = self.CLASS_LEN_M if isinstance(self.CLASS_LEN_M, dict) else {}
        if cls_name in length_map:
            try:
                return float(length_map[cls_name])
            except Exception:
                pass
        return float(self.DEFAULT_LEN_M)

    def run(self, video_path: str, out_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

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

        # ==== FSM & caches ====
        static_cnt: Dict[int, int] = {}
        moving_cnt: Dict[int, int] = {}
        state: Dict[int, str] = {}

        mpp_ema: Dict[int, float] = {}

        # Baseline fixed-EMA state (m/s) -> corresponds to v_ema_kmh column
        speed_ema_fixed: Dict[int, float] = {}

        # Weighted closed-loop state (m/s) -> corresponds to v_ws_ema_kmh column
        speed_ema_ws: Dict[int, float] = {}

        # Active state for overlay (m/s): fixed or weighted depending on enabled
        speed_active: Dict[int, float] = {}

        # diagnostics rows (keep the OLD column set)
        debug_rows = []

        last_boxes = None
        last_gray = None
        last_idx = None
        last_H = None

        orb = cv2.ORB_create(nfeatures=self.nfeatures)

        global_idx = -1
        homo_inlier_ratio: Optional[float] = None
        homo_inlier_count: Optional[int] = None

        for r in stream:
            global_idx += 1
            frame = r.orig_img
            vis = frame.copy()

            names = getattr(r, "names", None) or getattr(self.model, "names", {})

            if r.boxes is not None and len(r.boxes) > 0:
                bb = r.boxes
                boxes = bb.xyxy.cpu().numpy()
                clses = bb.cls.cpu().numpy().astype(int)
                ids = (bb.id.cpu().numpy() if bb.id is not None else np.full(len(bb), -1)).astype(int)
            else:
                boxes = np.zeros((0, 4), dtype=float)
                clses = np.zeros((0,), dtype=int)
                ids = np.zeros((0,), dtype=int)

            gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ================= update step every delta_frames =================
            need_pair = (last_idx is None) or ((global_idx - last_idx) >= self.delta_frames)

            if need_pair and last_gray is not None and last_boxes is not None:
                mask_prev = make_bg_mask((H, W, 3), last_boxes["boxes"], last_boxes["clses"], self.vehicle_ids, self.expand_ratio)
                mask_now = make_bg_mask((H, W, 3), boxes, clses, self.vehicle_ids, self.expand_ratio)

                kp1, des1 = orb.detectAndCompute(last_gray, mask_prev)
                kp2, des2 = orb.detectAndCompute(gray_now, mask_now)

                Hmat = None
                H_ok = False

                if des1 is not None and des2 is not None and len(kp1) >= 20 and len(kp2) >= 20:
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
                            pts_prev, pts_now,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=self.reproj_thresh,
                            maxIters=self.max_iters,
                            confidence=self.confidence,
                        )

                        if Hmat is not None and inl is not None:
                            inl_cnt = int(inl.sum())
                            good_cnt = max(len(good), 1)
                            if inl_cnt >= int(0.5 * len(good)):
                                H_ok = True
                                last_H = Hmat
                                homo_inlier_ratio = float(inl_cnt / good_cnt)
                                homo_inlier_count = int(inl_cnt)

                if not H_ok and last_H is None:
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

                    # centers at last keyframe
                    prev_centers: Dict[int, Tuple[float, float]] = {}
                    for (b_prev, tid_prev, c_prev) in zip(last_boxes["boxes"], last_boxes["ids"], last_boxes["clses"]):
                        if int(c_prev) not in self.vehicle_ids:
                            continue
                        cx_prev = 0.5 * (b_prev[0] + b_prev[2])
                        cy_prev = 0.5 * (b_prev[1] + b_prev[3])
                        prev_centers[int(tid_prev)] = (float(cx_prev), float(cy_prev))

                    dt = max(1e-6, float(global_idx - last_idx) / float(fps))

                    for (b, tid, c) in zip(boxes, ids, clses):
                        tid = int(tid)
                        cid = int(c)
                        if cid not in self.vehicle_ids:
                            continue
                        if tid not in prev_centers:
                            continue

                        # compensated displacement
                        cx_now = 0.5 * (b[0] + b[2])
                        cy_now = 0.5 * (b[1] + b[3])

                        prev_pt = np.array([[prev_centers[tid][0], prev_centers[tid][1], 1.0]], dtype=np.float32).T
                        proj = Hmat @ prev_pt
                        proj /= (proj[2] + 1e-9)
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

                        # FSM
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

                        # mpp
                        w_box = float(b[2] - b[0])
                        h_box = float(b[3] - b[1])
                        long_edge = float(max(w_box, h_box))
                        if long_edge <= max(1e-6, self.MIN_LONG_EDGE):
                            continue

                        cname = names.get(cid, str(cid))
                        length_m = self._get_vehicle_length_m(str(cname))

                        mpp_now = float(np.clip(length_m / long_edge, self.MIN_MPP, self.MAX_MPP))
                        if tid in mpp_ema:
                            mpp_ema[tid] = self.EMA_ALPHA_MPP * mpp_now + (1.0 - self.EMA_ALPHA_MPP) * mpp_ema[tid]
                        else:
                            mpp_ema[tid] = mpp_now

                        # speed update
                        if (not self.SHOW_MOVING_ONLY) or (state.get(tid) != "static"):
                            v_mps = (d * mpp_ema[tid]) / dt
                            if v_mps <= self.MAX_MPS_CLAMP:
                                v_raw_kmh = float(v_mps * 3.6)

                                # v_old_kmh: previous ACTIVE smooth (this matches your old export semantics)
                                v_old_mps = float(speed_active.get(tid, v_mps))
                                v_old_kmh = float(v_old_mps * 3.6)

                                # baseline fixed EMA next (v_ema_kmh)
                                if tid in speed_ema_fixed:
                                    speed_ema_fixed[tid] = self.EMA_ALPHA_SPEED * v_mps + (1.0 - self.EMA_ALPHA_SPEED) * speed_ema_fixed[tid]
                                else:
                                    speed_ema_fixed[tid] = v_mps
                                v_ema_kmh = float(speed_ema_fixed[tid] * 3.6)

                                # weighted instantaneous update (v_ws_kmh) and closed-loop (v_ws_ema_kmh)
                                v_ws_kmh = None
                                v_ws_ema_kmh = None
                                wb = we = wd = wh = W_raw = W_final = W_use = None

                                if self.ws_smoother is not None:
                                    cx = 0.5 * (b[0] + b[2])
                                    cy = 0.5 * (b[1] + b[3])

                                    wb, we, wd, wh, W_raw, W_final = self.ws_smoother.compute_components(
                                        tid=tid,
                                        bbox_w=w_box, bbox_h=h_box,
                                        cx=cx, cy=cy,
                                        imgW=W, imgH=H,
                                        v_meas_kmh=v_raw_kmh,
                                        v_smooth_kmh=v_old_kmh,  # reference = previous active
                                        inlier_ratio=homo_inlier_ratio,
                                        inlier_count=homo_inlier_count,
                                    )

                                    # cap by baseline alpha
                                    alpha_cap = float(self.EMA_ALPHA_SPEED)
                                    W_use = float(min(max(float(W_final), 0.0), alpha_cap))

                                    # instantaneous "measurement-like" weighted update
                                    v_ws_kmh = float(W_use * v_raw_kmh + (1.0 - W_use) * v_old_kmh)
                                    v_ws_mps_meas = float(v_ws_kmh / 3.6)

                                    # closed-loop weighted state (to compare curves stably)
                                    if tid in speed_ema_ws:
                                        speed_ema_ws[tid] = self.EMA_ALPHA_SPEED * v_ws_mps_meas + (1.0 - self.EMA_ALPHA_SPEED) * speed_ema_ws[tid]
                                    else:
                                        speed_ema_ws[tid] = v_ws_mps_meas
                                    v_ws_ema_kmh = float(speed_ema_ws[tid] * 3.6)

                                # choose ACTIVE for overlay: fixed or weighted
                                if self.ws_smoother is not None and v_ws_ema_kmh is not None:
                                    speed_active[tid] = float(speed_ema_ws[tid])  # weighted active
                                else:
                                    speed_active[tid] = float(speed_ema_fixed[tid])  # fixed active

                                # export row (keep EXACT old column names)
                                if self.diag_export_on:
                                    debug_rows.append({
                                        "frame": int(global_idx),
                                        "dt": float(dt),
                                        "class_id": int(cid),
                                        "class_name": str(cname),
                                        "tid": int(tid),
                                        "state": str(state.get(tid, "unknown")),
                                        "d_px": float(d),
                                        "long_edge_px": float(long_edge),
                                        "mpp": float(mpp_ema.get(tid, 0.0)),

                                        "v_raw_kmh": float(v_raw_kmh),
                                        "v_old_kmh": float(v_old_kmh),
                                        "v_ema_kmh": float(v_ema_kmh),
                                        "v_ws_kmh": v_ws_kmh,
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

            # ================= overlay =================
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clses):
                tid = int(tid)
                cid = int(c)
                cname = names.get(cid, str(cid))
                if cid not in self.vehicle_ids:
                    continue

                st = state.get(tid, "unknown")
                scnt = static_cnt.get(tid, 0)
                has_speed = tid in speed_active

                if st == "static" and scnt >= self.K_STATIC:
                    color = COLOR_STATIC
                    thickness = 2
                    label = "STATIC"
                elif has_speed:
                    color = COLOR_MOVING
                    thickness = 2
                    kmh = float(speed_active[tid] * 3.6)
                    label = f"{kmh:4.1f} km/h"
                else:
                    color = COLOR_NO_RESULT
                    thickness = 1
                    label = f"{cname} ID {tid}"

                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
                draw_label(vis, int(x1), int(y1), label, color)

            vw.write(vis)

        vw.release()

        # ---------- Export diagnostics ----------
        if self.diag_export_on and len(debug_rows) > 0:
            import pandas as pd
            os.makedirs(self.diag_out_dir, exist_ok=True)
            out_base = os.path.join(self.diag_out_dir, self.diag_prefix)

            df = pd.DataFrame(debug_rows)

            if self.diag_format == "csv":
                out_file = out_base + ".csv"
                df.to_csv(out_file, index=False, encoding="utf-8-sig")
            else:
                out_file = out_base + ".xlsx"
                df.to_excel(out_file, index=False)

            print(f"[OK] Saved diagnostics: {out_file}")

        print(f"[OK] Saved speed-annotated video: {out_path}")
