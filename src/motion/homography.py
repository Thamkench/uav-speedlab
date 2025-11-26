# src/motion/homography.py
# -*- coding: utf-8 -*-
"""
Background homography estimation with ORB.
基于 ORB 的背景 Homography 估计（用于相机运动补偿）。
"""

from typing import Tuple, Optional, Sequence
import numpy as np
import cv2


class BackgroundHomographyEstimator:
    """
    Estimate camera motion using background keypoints only.
    使用背景特征点估计相机运动（只利用静态背景区域）。
    """

    def __init__(
        self,
        nfeatures: int = 2000,
        ratio_thresh: float = 0.8,
        reproj_thresh: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.995,
    ) -> None:
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.ratio_thresh = ratio_thresh
        self.reproj_thresh = reproj_thresh
        self.max_iters = max_iters
        self.confidence = confidence
        self.last_H: Optional[np.ndarray] = None

    @staticmethod
    def make_bg_mask(
        shape: Tuple[int, int],
        boxes: np.ndarray,
        clses: np.ndarray,
        vehicle_ids: Sequence[int],
        expand_ratio: float = 0.03,
    ) -> np.ndarray:
        """
        Create a mask for background: dynamic objects (vehicles) are blacked out.
        生成背景掩膜：将动态目标（车辆等）置为黑色，保留背景。

        Args:
            shape: (H, W) of the frame.
            boxes: Nx4 boxes in xyxy.
            clses: N class ids.
            vehicle_ids: class ids to exclude (mask out).
            expand_ratio: extra margin around boxes.

        Returns:
            mask: uint8 array, 255 for background, 0 for dynamic objects.
        """
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if boxes is None or len(boxes) == 0:
            return mask

        vid_set = set(int(v) for v in vehicle_ids)
        for (x1, y1, x2, y2), cid in zip(boxes, clses):
            if int(cid) not in vid_set:
                continue
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            bw, bh = x2 - x1, y2 - y1
            dx, dy = bw * expand_ratio, bh * expand_ratio
            xa = max(0, int(x1 - dx))
            ya = max(0, int(y1 - dy))
            xb = min(w - 1, int(x2 + dx))
            yb = min(h - 1, int(y2 + dy))
            cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)

        return mask

    def estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_boxes: np.ndarray,
        prev_clses: np.ndarray,
        curr_boxes: np.ndarray,
        curr_clses: np.ndarray,
        vehicle_ids: Sequence[int],
        expand_ratio: float,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Estimate homography H(prev -> curr) using only background regions.
        在背景区域上估计 prev->curr 的单应矩阵 H。

        Returns:
            H (3x3) or None, and a boolean flag indicating success.
        """
        h, w = prev_gray.shape[:2]
        mask_prev = self.make_bg_mask((h, w), prev_boxes, prev_clses,
                                      vehicle_ids, expand_ratio)
        mask_curr = self.make_bg_mask((h, w), curr_boxes, curr_clses,
                                      vehicle_ids, expand_ratio)

        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask_prev)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask_curr)

        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            return self.last_H, False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in knn:
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        if len(good) < 30:
            return self.last_H, False

        pts_prev = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in good])

        H, inliers = cv2.findHomography(
            pts_prev,
            pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        if H is not None and inliers is not None and inliers.sum() >= 0.5 * len(good):
            self.last_H = H
            return H, True

        # fallback: use last homography if available
        return self.last_H, False
