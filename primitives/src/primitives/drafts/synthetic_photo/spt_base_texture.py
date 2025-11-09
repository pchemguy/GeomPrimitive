"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import time
import random
import math
import numpy as np
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

from spt_base import *


def apply_texture(img: ImageBGR,
                  texture_strength: float = 0.12,
                  texture_scale: float = 8.0,
                 ) -> ImageBGR:
    """
    Apply multiplicative paper-fiber texture modulation before optical effects.
    
    The algorithm generates a smooth random field `N(x, y)` from normally
    distributed noise filtered with a Gaussian kernel of standard deviation
    `texture_scale`. The field is then normalized to [0, 1] and remapped into a
    multiplicative modulation mask:
    
        T(x, y) = 1 + texture_strength * (N(x, y) - 0.5)
    
    where `texture_strength` controls the amplitude of contrast variation and
    `texture_scale` defines the spatial correlation length (fiber coarseness).
    The resulting texture field `T` is applied to the input image
    multiplicatively per pixel and per channel. Values are clipped to the
    [0, 255] range and returned as uint8.
    
    Args:
        img: Input image in uint8 BGR format.
        texture_strength: Amplitude of multiplicative modulation (typ. 0.05-0.3).
        texture_scale: Gaussian blur sigma controlling feature size in pixels.
    
    Returns:
        ImageBGR: Texture-modulated image, same shape as input.
    """
    if texture_strength <= 0 or texture_scale <= 0:
        return img

    h, w = img.shape[:2]

    # --- Generate correlated noise field ---
    noise_field = np.random.randn(h, w).astype(np.float32)
    noise_field = cv2.GaussianBlur(noise_field, (0, 0), texture_scale)

    # --- Normalize to [0, 1] ---
    nf_min, nf_max = float(noise_field.min()), float(noise_field.max())
    noise_norm = (noise_field - nf_min) / (nf_max - nf_min + 1e-9)

    # --- Construct multiplicative texture mask ---
    texture = 1.0 + texture_strength * (noise_norm - 0.5)

    # --- Apply and return ---
    img_f = img.astype(np.float32)
    out = np.clip(img_f * texture[..., None], 0, 255).astype(np.uint8)
    return out


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    proc_bgr:  ImageBGR  = apply_texture(
                               img=base_bgr,
                               texture_strength=0.12,
                               texture_scale=8.0,
                           )

    rng = random.Random(os.getpid() ^ int(time.time()))
    random_props = {
        "img":            bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "texture_strength":   abs(max(-0.5, min(0.5, 0.1 * rng.normalvariate(0, 1)))),
        "texture_scale":      abs(max(5, min(5, rng.normalvariate(0, 1)))),
    }
    demos = {
        "BASELINE": base_rgba,
        "RANDOM":   rgb_from_bgr(apply_texture(**random_props)),
    }

    default_props = {"img": base_bgr,}
    demo_set = [
        {"texture_strength": 0.1, "texture_scale": 8},
        {"texture_strength": 0.2, "texture_scale": 8},
        {"texture_strength": 0.4, "texture_scale": 8},
        {"texture_strength": 0.8, "texture_scale": 8},
        {"texture_strength": 0.2, "texture_scale": 0.5},
        {"texture_strength": 0.2, "texture_scale": 1},
        {"texture_strength": 0.2, "texture_scale": 2},
        {"texture_strength": 0.2, "texture_scale": 4},
    ]
    for custom_props in demo_set:
        title = (
            f"Texture strength x scale: {float(custom_props['texture_strength']):.1f} x "
            f"{float(custom_props['texture_scale']):.1f}"
        )
        print(title)
        demos[title] = rgb_from_bgr(
            apply_texture(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()

