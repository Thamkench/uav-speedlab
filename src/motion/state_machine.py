# src/motion/state_machine.py
# -*- coding: utf-8 -*-
"""
Static / moving state machine with hysteresis.
带迟滞的静止 / 运动 状态机。
"""

from typing import Dict


class StaticMovingStateMachine:
    """
    Maintain per-track id state: 'static' / 'moving' / 'unknown'.
    针对每个 track_id 维护“静止/运动/未知”状态。
    """

    def __init__(
        self,
        d_static_px: float,
        d_moving_px: float,
        r_static_mean: float,
        r_moving_mean: float,
        k_static: int,
        k_moving: int,
    ) -> None:
        self.d_static_px = d_static_px
        self.d_moving_px = d_moving_px
        self.r_static_mean = r_static_mean
        self.r_moving_mean = r_moving_mean
        self.k_static = k_static
        self.k_moving = k_moving

        self.static_cnt: Dict[int, int] = {}
        self.moving_cnt: Dict[int, int] = {}
        self.state: Dict[int, str] = {}

    def update(self, tid: int, d_px: float, r_mean: float) -> str:
        """
        Update state for a given track id.
        更新对应 track id 的状态。

        Args:
            tid: Track id.
            d_px: Compensated geometric displacement in pixels.
                  经相机补偿后的几何位移（像素）。
            r_mean: Mean brightness residual in ROI.
                    ROI 的亮度残差均值。

        Returns:
            Current state: 'static', 'moving', or 'unknown'.
        """
        self.static_cnt.setdefault(tid, 0)
        self.moving_cnt.setdefault(tid, 0)
        self.state.setdefault(tid, "unknown")

        is_static_ev = (d_px <= self.d_static_px) and (r_mean <= self.r_static_mean)
        is_moving_ev = (d_px >= self.d_moving_px) or (r_mean >= self.r_moving_mean)

        if is_static_ev:
            self.static_cnt[tid] += 1
            self.moving_cnt[tid] = 0
        elif is_moving_ev:
            self.moving_cnt[tid] += 1
            self.static_cnt[tid] = 0
        else:
            self.static_cnt[tid] = max(0, self.static_cnt[tid] - 1)
            self.moving_cnt[tid] = max(0, self.moving_cnt[tid] - 1)

        if self.static_cnt[tid] >= self.k_static:
            self.state[tid] = "static"
        elif self.moving_cnt[tid] >= self.k_moving:
            self.state[tid] = "moving"

        return self.state[tid]

    def get(self, tid: int) -> str:
        """Get current state for tid (default 'unknown')."""
        return self.state.get(tid, "unknown")

    def get_static_cnt(self, tid: int) -> int:
        return self.static_cnt.get(tid, 0)
