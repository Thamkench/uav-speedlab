# src/motion/weighted_speed_smoother.py
# -*- coding: utf-8 -*-
"""
Confidence-weighted speed smoothing (scaffold + ready-to-use functions).
置信度加权速度平滑（骨架 + 可直接使用的权重计算）。

Core idea / 核心思想：
    W_raw = w_bbox * w_deltaV * w_edge * w_homo(optional)
    W_smooth = EMA(W_raw)
    W = clamp(W_smooth, W_floor, 1)
    V_smooth_new = W * V_meas + (1 - W) * V_smooth_old

Notes / 说明：
- This module maintains per-track state (s_ref, W_smooth).
  本模块维护每个 track 的状态（bbox 尺度参考 s_ref、总权重平滑 W_smooth）。
- It is lightweight: only a few math ops per track per update.
  开销很小：每个 track 每次更新只做少量数学运算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math
import numpy as np


# -------------------------
# Basic utils / 基础工具
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi] / 将 x 限制到 [lo, hi]"""
    return lo if x < lo else (hi if x > hi else x)


def clamp01(x: float) -> float:
    """Clamp x into [0,1] / 将 x 限制到 [0,1]"""
    return clamp(x, 0.0, 1.0)


# -------------------------
# Parameter container / 参数容器
# -------------------------
@dataclass
class WeightedSpeedParams:
    # master switch / 总开关
    enabled: bool = False

    # bbox weight / bbox 权重
    rho_size: float = 0.93          # EMA for bbox size reference / bbox 尺度参考 EMA
    sigma_bbox: float = 0.25        # sensitivity / 映射敏感度
    w_min_bbox: float = 0.30        # floor / 下限

    # deltaV weight / 速度跳变权重
    p_dv: float = 3.0               # soft gating exponent / 软门控指数
    w_min_dv: float = 0.25          # floor / 下限
    dv_a: float = 10.0              # ΔVmax = a + b*V_smooth (km/h) / 固定项
    dv_b: float = 0.15              # proportional term / 比例项

    # edge weight / 边缘权重
    edge_ratio: float = 0.10        # border band ratio / 边缘带宽比例
    w_min_edge: float = 0.40        # floor / 下限

    # total W smoothing / 总权重平滑
    eta_w: float = 0.90             # EMA factor for total W / 总权重 EMA 系数
    W_floor: float = 0.10           # clamp floor / 总权重下限

    # numeric epsilon / 数值稳定项
    eps: float = 1e-6

    homo_enabled: bool = True
    homo_r0: float = 0.5
    homo_N0: float = 60
    w_min_h: float = 0.30




    @staticmethod
    def from_cfg(cfg: dict) -> "WeightedSpeedParams":
        """
        Build params from full config dict / 从总配置 dict 构建参数。
        Expects cfg["weighted_speed"] exists (optional).
        """
        ws = cfg.get("weighted_speed", {}) or {}
        bbox = ws.get("bbox", {}) or {}
        dv = ws.get("deltaV", {}) or {}
        edge = ws.get("edge", {}) or {}
        wsm = ws.get("weight_smooth", {}) or {}
        homo = ws.get("homo", {}) or {}

        return WeightedSpeedParams(
            enabled=bool(ws.get("enabled", False)),

            rho_size=float(bbox.get("rho_size", 0.93)),
            sigma_bbox=float(bbox.get("sigma_bbox", 0.25)),
            w_min_bbox=float(bbox.get("w_min_bbox", 0.30)),

            p_dv=float(dv.get("p", 3.0)),
            w_min_dv=float(dv.get("w_min_dv", 0.25)),
            dv_a=float(dv.get("dv_a", 10.0)),
            dv_b=float(dv.get("dv_b", 0.15)),

            edge_ratio=float(edge.get("edge_ratio", 0.10)),
            w_min_edge=float(edge.get("w_min_edge", 0.40)),

            eta_w=float(wsm.get("eta_w", 0.90)),
            W_floor=float(wsm.get("W_floor", 0.10)),

            eps=float(ws.get("eps", 1e-6)),

            homo_enabled=bool(homo.get("enabled", True)),
            homo_r0=float(homo.get("r0", 0.50)),
            homo_N0=float(homo.get("N0", 60)),
            w_min_h=float(homo.get("w_min_h", 0.30)),

        )


# -------------------------
# Per-track state / 每个 track 的状态
# -------------------------
@dataclass
class TrackWeightState:
    """
    Per-track internal state / 每个 track 的内部状态
    """
    s_ref: float = 0.0       # EMA reference of bbox size sqrt(area) / bbox 尺度参考（sqrt(area) 的 EMA）
    W_smooth: float = 1.0    # EMA of total weight W / 总权重 W 的 EMA


# -------------------------
# Main smoother / 主类
# -------------------------
class WeightedSpeedSmoother:
    """
    Compute confidence weights and return W for smoothing V_meas.
    计算置信度权重并返回 W，用于对 V_meas 做加权平滑。

    This class is designed to be lightweight and safe-by-default.
    本类设计为轻量、默认安全：enabled=False 时可完全不使用。
    """

    def __init__(self, params: WeightedSpeedParams) -> None:
        self.p = params
        self._st: Dict[int, TrackWeightState] = {}

    def _get_state(self, tid: int) -> TrackWeightState:
        if tid not in self._st:
            self._st[tid] = TrackWeightState()
        return self._st[tid]

    # ------------------------------------------------------------------
    # Weight components / 权重分量
    # ------------------------------------------------------------------
    def w_bbox(self, tid: int, bbox_w: float, bbox_h: float) -> float:
        """
        bbox size consistency weight / bbox 尺度一致性权重
        s = sqrt(area), compare to EMA ref size s_ref.
        """
        # s = sqrt(w*h) / bbox 尺度（sqrt(面积)）
        s = math.sqrt(max(self.p.eps, float(bbox_w) * float(bbox_h)))

        st = self._get_state(tid)

        # update s_ref EMA / 更新 s_ref 的 EMA
        if st.s_ref <= 0.0:
            st.s_ref = s
        else:
            st.s_ref = self.p.rho_size * st.s_ref + (1.0 - self.p.rho_size) * s

        # d = |ln(s/s_ref)| / 相对偏离（对称）
        d = abs(math.log((s + self.p.eps) / (st.s_ref + self.p.eps)))

        # w = w_min + (1-w_min)*exp(-d^2/(2*sigma^2))
        denom = 2.0 * self.p.sigma_bbox * self.p.sigma_bbox + self.p.eps
        w = self.p.w_min_bbox + (1.0 - self.p.w_min_bbox) * math.exp(-(d * d) / denom)
        return clamp01(w)

    def w_edge(self, cx: float, cy: float, W: int, H: int) -> float:
        """
        border weight / 边缘权重
        down-weight near borders based on minimal distance to image edge.
        """
        d_edge = min(cx, float(W) - cx, cy, float(H) - cy)
        d0 = self.p.edge_ratio * float(min(W, H))
        t = 0.0 if d0 <= 0.0 else clamp01(d_edge / (d0 + self.p.eps))
        w = self.p.w_min_edge + (1.0 - self.p.w_min_edge) * t
        return clamp01(w)

    def w_deltav(self, v_meas_kmh: float, v_smooth_kmh: float) -> float:
        """
        speed jump gating weight / 速度跳变门控权重
        dv_max = dv_a + dv_b * V_smooth(km/h)
        """
        dv = abs(float(v_meas_kmh) - float(v_smooth_kmh))
        dv_max = self.p.dv_a + self.p.dv_b * float(v_smooth_kmh)
        ratio = dv / (dv_max + self.p.eps)

        # soft gating: 1 / (1 + (ratio)^p)
        soft = 1.0 / (1.0 + (ratio ** self.p.p_dv))
        w = self.p.w_min_dv + (1.0 - self.p.w_min_dv) * soft
        return clamp01(w)

    def w_homo(self, inlier_ratio: Optional[float] = None, inlier_count: Optional[int] = None) -> float:
        """
        homography quality weight / 单应质量权重
        Uses inlier ratio and inlier count, maps to [w_min_h, 1].
        用内点比例+内点数量估计单应质量，并映射到 [w_min_h, 1]。
        """
        if not self.p.homo_enabled:
            return 1.0
        if inlier_ratio is None or inlier_count is None:
            return 1.0

        r = float(inlier_ratio)
        n = float(inlier_count)

        # ratio quality: (r - r0) / (1 - r0)
        qr = (r - self.p.homo_r0) / max(1.0 - self.p.homo_r0, self.p.eps)
        qr = float(np.clip(qr, 0.0, 1.0))

        # count quality: n / N0
        qn = n / max(self.p.homo_N0, 1.0)
        qn = float(np.clip(qn, 0.0, 1.0))

        q_h = qr * qn  # in [0,1]

        # map to [w_min_h, 1]
        return float(self.p.w_min_h + (1.0 - self.p.w_min_h) * q_h)

    # ------------------------------------------------------------------
    # Total weight W / 总权重 W
    # ------------------------------------------------------------------
    def compute_W(
        self,
        tid: int,
        bbox_w: float,
        bbox_h: float,
        cx: float,
        cy: float,
        imgW: int,
        imgH: int,
        v_meas_kmh: float,
        v_smooth_kmh: float,
        inlier_ratio: Optional[float] = None,
        inlier_count: Optional[int] = None,
    ) -> float:
        """
        Compute final W for tid / 计算 tid 对应的最终权重 W。

        Returns:
            W in [W_floor, 1.0]
        """
        wb = self.w_bbox(tid, bbox_w, bbox_h)
        we = self.w_edge(cx, cy, imgW, imgH)
        wd = self.w_deltav(v_meas_kmh, v_smooth_kmh)
        wh = self.w_homo(inlier_ratio, inlier_count)

        W_raw = wb * we * wd * wh
        W_raw = clamp01(W_raw)

        # EMA on W / 对 W 做 EMA 平滑
        st = self._get_state(tid)
        st.W_smooth = self.p.eta_w * st.W_smooth + (1.0 - self.p.eta_w) * W_raw

        # floor clamp / 地板
        W_final = clamp(st.W_smooth, self.p.W_floor, 1.0)
        return W_final

    def compute_components(
        self,
        tid: int,
        bbox_w: float,
        bbox_h: float,
        cx: float,
        cy: float,
        imgW: int,
        imgH: int,
        v_meas_kmh: float,
        v_smooth_kmh: float,
        inlier_ratio: Optional[float] = None,
        inlier_count: Optional[int] = None,
    ):
        """
        Compute weight components and final W / 计算权重分量与最终 W（用于调试分析）

        Returns / 返回：
            wbbox, wedge, wdv, whomo, W_raw, W_final
        """
        wb = self.w_bbox(tid, bbox_w, bbox_h)
        we = self.w_edge(cx, cy, imgW, imgH)
        wd = self.w_deltav(v_meas_kmh, v_smooth_kmh)
        wh = self.w_homo(inlier_ratio, inlier_count)

        W_raw = wb * we * wd * wh
        W_raw = clamp01(W_raw)

        # EMA on W / 对 W 做 EMA 平滑
        st = self._get_state(tid)
        st.W_smooth = self.p.eta_w * st.W_smooth + (1.0 - self.p.eta_w) * W_raw

        # floor clamp / 地板
        W_final = clamp(st.W_smooth, self.p.W_floor, 1.0)
        return wb, we, wd, wh, W_raw, W_final

    # ------------------------------------------------------------------
    # Optional: apply smoothing / 可选：直接输出平滑速度
    # ------------------------------------------------------------------
    def smooth_speed(
        self,
        tid: int,
        v_meas_kmh: float,
        v_smooth_old_kmh: float,
        bbox_w: float,
        bbox_h: float,
        cx: float,
        cy: float,
        imgW: int,
        imgH: int,
        inlier_ratio: Optional[float] = None,
        inlier_count: Optional[int] = None,
    ) -> float:
        """
        One-stop API: compute W and output v_smooth_new.
        一站式接口：计算 W 并输出 v_smooth_new。

        v_new = W * v_meas + (1-W) * v_old
        """
        W = self.compute_W(
            tid=tid,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            cx=cx,
            cy=cy,
            imgW=imgW,
            imgH=imgH,
            v_meas_kmh=v_meas_kmh,
            v_smooth_kmh=v_smooth_old_kmh,
            inlier_ratio=inlier_ratio,
            inlier_count=inlier_count,
        )
        return W * float(v_meas_kmh) + (1.0 - W) * float(v_smooth_old_kmh)
