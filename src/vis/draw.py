# src/vis/draw.py
# -*- coding: utf-8 -*-
"""
Drawing helpers for boxes, labels and speed text.
绘制框、标签与速度文字的辅助函数。
"""

import cv2
import numpy as np


def draw_label(
    img: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    color,
    font_scale: float = 0.5,
    thickness: int = 1,
    alpha: float = 0.4,
) -> None:
    """
    Draw a filled label box above the bbox.
    在框上方绘制带背景的文字标签。

    Args:
        img: BGR 图像
        x1, y1: 框左上角坐标（用于放 label）
        text: 要显示的完整文字，例如 "car ID 3 35.2 km/h STATIC"
        color: BGR 颜色，用作 label 背景色
    """
    if not text:
        return

    text = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 2

    bx1 = int(max(0, x1))
    by1 = int(max(0, y1 - th - 2 * pad))
    bx2 = int(min(img.shape[1] - 1, x1 + tw + 2 * pad))
    by2 = int(min(img.shape[0] - 1, y1))

    if bx2 <= bx1 or by2 <= by1:
        return

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    img[by1:by2, bx1:bx2] = cv2.addWeighted(
        overlay[by1:by2, bx1:bx2],
        alpha,
        img[by1:by2, bx1:bx2],
        1 - alpha,
        0,
    )

    cv2.putText(
        img,
        text,
        (bx1 + pad, by2 - pad),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
