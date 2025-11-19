"""
spt_geometry.py
-----------
"""

from __future__ import annotations

__all__ = ["spt_geometry",]

import os
import sys
import time
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt

from mpl_utils import *


def spt_geometry(img: ImageBGR, tilt_x: float = 0.18, tilt_y: float = 0.10,
                 k1: float = -0.25, k2: float = 0.05) -> ImageBGR:
    """
    Apply synthetic camera projection effects: perspective tilt and radial
    lens distortion.
  
    The algorithm simulates two geometric distortions typical of optical imaging
    systems in a simplified pinhole-camera approximation:
  
    1. **Perspective tilt:**  
       A planar projective transform skews the image about its center using
       normalized offsets `tilt_x` and `tilt_y` (fractions of image width and
       height). This emulates off-axis projection or lens-plane tilt, producing
       perspective convergence similar to a tilted camera sensor.
  
    2. **Radial lens distortion:**  
       Each pixel is remapped according to a radial polynomial of normalized
       radius `r` from the optical center:
  
           r' = r * (1 + k1 * r^2 + k2 * r^4)
  
       where negative `k1` values yield barrel (wide-angle) distortion, positive
       values yield pincushion (telephoto) distortion, and `k2` refines the
       curvature roll-off.  
       The normalization ensures that coefficients drawn from a zero-mean normal
       distribution with sigma = 0.25 produce realistic variation magnitudes.
  
    Args:
        img:    Input image (uint8, BGR order).
        tilt_x: Horizontal perspective skew fraction (0 - no tilt).
        tilt_y: Vertical perspective skew fraction (0 - no tilt).
        k1:     Primary radial distortion coefficient.
        k2:     Secondary radial distortion coefficient.
  
    Returns:
        ImageBGR: Image with perspective and lens distortion applied.
    """
    h, w = img.shape[:2]

    # -------------------------------------------------------------
    # 1. Perspective tilt (off-axis projection)
    # -------------------------------------------------------------
    dx, dy = 0.25 * tilt_x * w, 0.25 * tilt_y * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx, dy / 2], [w, h], [0, h - dy]])
    H = cv2.getPerspectiveTransform(src, dst)
    persp = cv2.warpPerspective(img, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # -------------------------------------------------------------
    # 2. Radial lens distortion
    # -------------------------------------------------------------
    cx, cy = w / 2.0, h / 2.0
    r_norm = max(w, h)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_d, y_d = (xx - cx) / r_norm, (yy - cy) / r_norm
    r2 = x_d * x_d + y_d * y_d
    factor = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    factor = np.where(factor == 0.0, 1.0, factor)  # safety
    x_u, y_u = x_d / factor,  y_d / factor

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
    proc_bgr:  ImageBGR  = spt_geometry(
                               img=base_bgr,
                               tilt_x=0.18,
                               tilt_y=0.10,
                               k1=-0.25,
                               k2=0.05,
                           )

    rng = random.Random(os.getpid() ^ int(time.time()))
    random_props = {
        "img":            bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "tilt_x":         max(-1, min(1, rng.normalvariate(0, 1/3))),
        "tilt_y":         max(-1, min(1, rng.normalvariate(0, 1/3))),
        "k1":             max(-1, min(1, rng.normalvariate(0, 1/3))),
        "k2":             max(-1, min(1, rng.normalvariate(0, 1/3))),
    }
    demos = {
        "BASELINE": base_rgba,
        "RANDOM":   rgb_from_bgr(spt_geometry(**random_props)),
    }

    default_props = {
        "img":            base_bgr,
        "tilt_x":         0,
        "tilt_y":         0,
        "k1":             0,
        "k2":             0,
    }

    demo_set = [
        {"tilt_x":  0.20,  "tilt_y":  0.20},
        {"tilt_x":  0.20,  "tilt_y":  0.40},
        {"tilt_x": -0.80,  "tilt_y":  0.80},
        {"tilt_x":  1.00,  "tilt_y": -1.00},
        {"k1":  0.2,  "k2": 0.05},
        {"k1":  0.5,  "k2": 0.05},
        {"k1":  1.0,  "k2": 0.05},
        {"k1": -0.5,  "k2": 0.05},
        {"k1":  0.1,  "k2": 0.1},
        {"k1":  0.1,  "k2": 0.2},
        {"k1":  0.1,  "k2": 0.5},
        {"k1":  0.1,  "k2": 0.8},
    ]
    for custom_props in demo_set:
        title = ["Optics "]
        for key, val in custom_props.items():
            title.append(f"key: '{key}': val '{val}'")
        title = "".join(title)
        print(title)
        demos[title] = rgb_from_bgr(
            spt_geometry(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()
