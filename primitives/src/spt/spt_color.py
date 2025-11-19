"""
spt_color.py
-----------
"""

from __future__ import annotations

__all__ = ["spt_vignette_and_color",]

import os
import sys
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


def spt_vignette_and_color(img: ImageBGR, vignette_strength: float = 0.35,
                           warm_strength: float = 0.10) -> ImageBGR:
    """
    Apply post-lens vignetting and chromatic channel imbalance.
    
    The algorithm constructs a normalized radial field `r in [0, 1]` measured
    from the optical center and applies two coupled effects:
    
    1. **Vignetting:** A quadratic radial attenuation of luminance simulating
       the optical fall-off toward the image corners. The attenuation mask is
    
           V(r) = 1 - vignette_strength * r^2
    
       where `vignette_strength` controls corner darkening amplitude.
    
    2. **Chromatic warming:** A mild wavelength-dependent gain imbalance that
       increases red-channel intensity and slightly decreases blue toward the
       periphery, approximating lens coatings and sensor color-response drift.
       Channel-specific scaling masks are defined as
    
           warm_mask(r) = 1 + warm_strength * r^1.2   (R channel)
           cool_mask(r) = 1 - 0.5 * warm_strength * r^1.2   (B channel)
    
    These masks are applied multiplicatively to the float32 image in BGR order,
    preserving relative color balance near the center while inducing subtle
    warmth and fall-off toward edges. The result is clipped to [0, 255] and
    quantized back to uint8.
    
    Args:
        img: Input image (uint8, BGR order).
        vignette_strength: Radial brightness fall-off magnitude (0-1).
        warm_strength: Peripheral color warming intensity (0-1).
    
    Returns:
        ImageBGR: Image with vignette and color-shift applied.
    """
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
    
    # --- Radial masks ---
    vignette_mask = 1.0 - vignette_strength * (r_norm ** 2)
    warm_mask = 1.0 + warm_strength * (r_norm ** 1.2)
    cool_mask = 1.0 - 0.5 * warm_strength * (r_norm ** 1.2)
    
    # --- Apply modulation ---
    img_f = img.astype(np.float32)
    img_f *= vignette_mask[..., None]
    img_f[..., 0] *= cool_mask   # B channel
    img_f[..., 2] *= warm_mask   # R channel
    
    return np.clip(img_f, 0, 255).astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    proc_bgr:  ImageBGR  = spt_vignette_and_color(
                               img=base_bgr,
                               vignette_strength=0.35,
                               warm_strength=0.10
                           )

    rng = random.Random(os.getpid() ^ int(time.time()))
    random_props = {
        "img":            bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "vignette_strength":  abs(max(-0.5, min(0.5, 0.1 * rng.normalvariate(0, 1)))),
        "warm_strength":      abs(max(-0.5, min(0.5, 0.1 * rng.normalvariate(0, 1)))),
    }
    demos = {
        "BASELINE": base_rgba,
        "RANDOM":   rgb_from_bgr(spt_vignette_and_color(**random_props)),
    }

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
            spt_vignette_and_color(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()

