"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spt_base import *


def apply_vignette_and_color_shift(img: ImageBGR,
                                   vignette_strength: float = 0.35,
                                   warm_strength: float = 0.10,
                                  ) -> ImageBGR:
    """Apply post-lens vignetting and chromatic imbalance."""
    if vignette_strength <= 0 and warm_strength <= 0:
      return img

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    rx = (xx - cx) / cx
    ry = (yy - cy) / cy
    r = np.sqrt(rx * rx + ry * ry)
    r_norm = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

    vignette_mask = 1.0 - vignette_strength * (r_norm**2)
    warm_mask = 1.0 + warm_strength * (r_norm**1.2)
    cool_mask = 1.0 - 0.5 * warm_strength * (r_norm**1.2)

    img_f = img.astype(np.float32)
    img_f *= vignette_mask[..., None]
    img_f[..., 0] *= cool_mask   # B
    img_f[..., 2] *= warm_mask   # R

    return np.clip(img_f, 0, 255).astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_vignette_and_color_shift(
                               img=base_bgr,
                               vignette_strength=0.35,
                               warm_strength=0.10
                           )

    demos = {}
    default_props = {"img": base_bgr,}
    demo_set = [
        {"vignette_strength": 0.00, "warm_strength": 0.00},
        {"vignette_strength": 0.10, "warm_strength": 0.00},
        {"vignette_strength": 0.20, "warm_strength": 0.00},
        {"vignette_strength": 0.30, "warm_strength": 0.00},
        {"vignette_strength": 0.40, "warm_strength": 0.00},
        {"vignette_strength": 0.10, "warm_strength": 0.10},
        {"vignette_strength": 0.10, "warm_strength": 0.20},
    ]
    for custom_props in demo_set:
        title = (
            f"Postoptica vignette x warmth: {float(custom_props['vignette_strength']):.1f} x "
            f"{float(custom_props['warm_strength']):.1f}"
        )
        print(title)
        demos[title] = rgb_from_bgr(
            apply_vignette_and_color_shift(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()

