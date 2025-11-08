"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spt_base import *


def apply_camera_effects(img: ImageBGR,
                         tilt_x: float = 0.18,
                         tilt_y: float = 0.10,
                         focal_scale: float = 1.0,
                         k1: float = -0.25,
                         k2: float = 0.05,
                         pad_px: int = 100,
                        ) -> ImageBGR:
    """Apply focal scaling, perspective tilt, and radial lens distortion.

    Args:
        img: Input BGR uint8 image (pre-optical domain).

    Returns:
        np.ndarray: Distorted image simulating optical projection.
    """
    # Padding
    if pad_px > 0:
      img = cv2.copyMakeBorder(img, pad_px, pad_px, pad_px, pad_px,
                               borderType=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
    h, w = img.shape[:2]

    # ===============================================================
    # 1. FOCAL LENGTH (Field of View)
    # ---------------------------------------------------------------
    # focal_scale = 1.0 -> baseline FOV
    # focal_scale > 1 -> telephoto (zoom-in, narrower)
    # focal_scale < 1 -> wide angle (zoom-out, wider)
    # ===============================================================
    if focal_scale != 1.0:
        f = focal_scale
        new_w = int(round(w / f))
        new_h = int(round(h / f))
        cx, cy = w // 2, h // 2
        x1 = max(cx - new_w // 2, 0)
        x2 = min(cx + new_w // 2, w)
        y1 = max(cy - new_h // 2, 0)
        y2 = min(cy + new_h // 2, h)
        cropped = img[y1:y2, x1:x2]
        img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # ===============================================================
    # 2. CAMERA TILT (Perspective)
    # ---------------------------------------------------------------
    # tilt_x, tilt_y are normalized fractions of width/height.
    # 0 = no tilt; 0.3 = strong skew.
    # ===============================================================
    dx = tilt_x * w
    dy = tilt_y * h
    src = np.float32([[0, 0],   [w, 0],           [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx, dy / 2], [w, h], [0, h - dy]])
    H = cv2.getPerspectiveTransform(src, dst)
    persp = cv2.warpPerspective(img, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # ===============================================================
    # 3. RADIAL LENS DISTORTION
    # ---------------------------------------------------------------
    # k1 < 0 -> barrel (wide angle)
    # k1 > 0 -> pincushion (telephoto)
    # k2 refines curvature falloff
    # ===============================================================
    cx, cy = w / 2.0, h / 2.0
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d * x_d + y_d * y_d
    factor = 1.0 + k1 * r2 + k2 * (r2**2)
    factor = np.where(factor == 0.0, 1.0, factor)
    x_u = x_d / factor
    y_u = y_d / factor
    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)

    distorted = cv2.remap(persp, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return distorted


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_camera_effects(
                               img=base_bgr,
                               tilt_x=0.18,
                               tilt_y=0.10,
                               focal_scale=1.0,
                               k1=-0.25,
                               k2=0.05,
                               pad_px=10,
                           )

    default_props = {
        "img":            base_bgr,
        "tilt_x":         0,
        "tilt_y":         0,
        "focal_scale":    1,
        "k1":             0,
        "k2":             0,
        "pad_px":         10,
    }
    demos = {
        "Baseline": rgb_from_bgr(apply_camera_effects(**default_props))
    }

    demo_set = [
        {"tilt_x": 0.05, "tilt_y": 0.01},
        {"tilt_x": 0.10, "tilt_y": 0.01},
        {"tilt_x": 0.15, "tilt_y": 0.01},
        {"tilt_x": 0.05, "tilt_y": 0.05},
        {"tilt_x": 0.05, "tilt_y": 0.10},
        {"tilt_x": 0.30, "tilt_y": 0.30},
        {"focal_scale": 1.0,  "k1": 0.01, "k2": 0.01},
        {"focal_scale": 1.2,  "k1": 0.01, "k2": 0.01},
        {"focal_scale": 1.05, "k1": 0.10, "k2": 0.10},
        {"focal_scale": 1.1,  "k1": 0.01, "k2": 0.01},
        {"focal_scale": 1.05, "k1": -0.2, "k2": -0.1},
    ]
    for custom_props in demo_set:
        title = ["Optics "]
        for key, val in custom_props.items():
            title.append(f"key: '{key}': val '{val}'")
        title = "".join(title)
        print(title)
        demos[title] = rgb_from_bgr(
            apply_camera_effects(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()
