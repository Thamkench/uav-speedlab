# src/motion/speed_estimator.py
# -*- coding: utf-8 -*-
"""
Speed estimation using meters-per-pixel and EMA smoothing.
基于像素-米比例与 EMA 的速度估计。
"""

from typing import Dict, Optional
import numpy as np


class SpeedEstimator:
    """
    Maintain per-track id meters-per-pixel and smoothed speed.
    针对每个轨迹维护 mpp 与平滑后的速度。
    """

    def __init__(
        self,
        car_length_m: float = 5.0,
        ema_alpha_mpp: float = 0.3,
        ema_alpha_speed: float = 0.6,
        min_mpp: float = 0.01,
        max_mpp: float = 0.3,
        max_speed_mps: float = 60.0,
        min_long_edge_px: int = 15,
    ) -> None:
        self.car_length_m = car_length_m
        self.ema_alpha_mpp = ema_alpha_mpp
        self.ema_alpha_speed = ema_alpha_speed
        self.min_mpp = min_mpp
        self.max_mpp = max_mpp
        self.max_speed_mps = max_speed_mps
        self.min_long_edge_px = min_long_edge_px

        self.mpp_ema: Dict[int, float] = {}
        self.speed_ema: Dict[int, float] = {}

    def update(
        self,
        tid: int,
        long_edge_px: float,
        d_px: float,
        dt: float,
    ) -> Optional[float]:
        """
        Update speed estimation for tid.

        Args:
            tid: Track id.
            long_edge_px: Bounding box long edge in pixels.
            d_px: Displacement between frames (already camera-compensated).
            dt: Time delta between frames (seconds).

        Returns:
            Smoothed speed in m/s, or None if cannot be estimated.
        """
        if long_edge_px < self.min_long_edge_px or dt <= 0:
            return None

        mpp_now = self.car_length_m / max(long_edge_px, 1e-6)
        mpp_now = float(np.clip(mpp_now, self.min_mpp, self.max_mpp))

        if tid in self.mpp_ema:
            self.mpp_ema[tid] = (
                self.ema_alpha_mpp * mpp_now
                + (1.0 - self.ema_alpha_mpp) * self.mpp_ema[tid]
            )
        else:
            self.mpp_ema[tid] = mpp_now

        v_mps = (d_px * self.mpp_ema[tid]) / dt
        if v_mps > self.max_speed_mps:
            # Treat as outlier; do not update EMA.
            return self.speed_ema.get(tid, None)

        if tid in self.speed_ema:
            self.speed_ema[tid] = (
                self.ema_alpha_speed * v_mps
                + (1.0 - self.ema_alpha_speed) * self.speed_ema[tid]
            )
        else:
            self.speed_ema[tid] = v_mps

        return self.speed_ema[tid]
