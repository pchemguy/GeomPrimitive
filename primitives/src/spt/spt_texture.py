"""
spt_texture.py
-----------
"""

from __future__ import annotations

__all__ = ["spt_texture",]

import os
import sys
import time
import random
import math
from typing import Union
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

from mpl_utils import *


def spt_texture(img: ImageBGR,
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


# ======================================================================
#  Box blur (same algorithm as previous)
# ======================================================================

def _box_blur(img: ImageBGRF, radius: int) -> ImageBGRF:
    """Cheap separable box filter using cumulative sums."""
    if radius <= 0:
        return img

    h, w = img.shape
    pad = radius

    # Horizontal blur
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    csum = np.cumsum(tmp, axis=1)
    left  = csum[:, :-2 * pad]
    right = csum[:, 2 * pad :]
    horz = (right - left) / (2 * pad)

    # Vertical blur
    tmp = np.pad(horz, ((pad, pad), (0, 0)), mode="reflect")
    csum = np.cumsum(tmp, axis=0)
    top  = csum[:-2 * pad, :]
    bot  = csum[2 * pad :, :]
    vert = (bot - top) / (2 * pad)
    return vert


# ======================================================================
#  Additive "paper texture" modulation for an existing image
# ======================================================================

def spt_texture_additive(
        img           : Union[ImageBGR, ImageBGRF],
        *,
        max_deviation : float = 0.05,
        n_layers      : int   = 3,
        base_radius   : int   = 1,
        seed          : int   = None,
    ) -> Union[ImageBGR, ImageBGRF]:
    """
    Apply *additive* paper-like brightness texture to an image (OpenCV BGR).

    Employs:
        - Multi-scale multi-radius smooth-noise accumulation
        - Normalization to zero-mean and scaling by max_deviation
        - Cumulative-sum box-blur routine
    Adds generated noise field to the input BGR image.

    Args:
        img:
            Input image in BGR uint8 or float32. Value range 0-255 or 0-1.
        max_deviation:
            Max absolute additive brightness variation (relative 0-1 scale).
            For uint8 images this corresponds to +/-(255 * max_deviation) shifts.
        n_layers:
            Number of noise layers to accumulate (multi-scale).
        base_radius:
            Radius exponent base; actual radii are base_radius * (2**i).
            Example: base_radius=2 -> radii = [2,4,8] for n_layers=3.
        seed:
            RNG seed.

    Returns:
        Image with additive paper modulation, same dtype as input.
    """
    if max_deviation <= 0 or n_layers <= 0:
        return img

    # --- Convert to float in [0,1] -------------------------------------
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)
        img_f = np.clip(img_f, 0.0, 1.0)

    h, w = img_f.shape[:2]
    rng = np.random.default_rng(seed)

    # --- Accumulate multi-scale smooth noise field ----------------------
    acc = np.zeros((h, w), dtype=np.float32)

    for i in range(n_layers):
        radius = base_radius * (2 ** i)   # same pattern: 2,4,8,... if base=2
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        smooth = _box_blur(noise, radius)
        acc += smooth

    # --- Normalize noise field exactly like generate_paper_texture ------
    acc -= acc.mean()
    acc /= (acc.std() or 1.0)
    acc *= float(max_deviation)

    # --- Apply additive brightness modulation ---------------------------
    # broadcast acc -> (H,W,1), add to all channels
    out = img_f + acc[..., None]
    out = np.clip(out, 0.0, 1.0)

    # --- Convert back to original dtype --------------------------------
    if img.dtype == np.uint8:
        return (out * 255.0).astype(np.uint8)
    else:
        return out


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    proc_bgr:  ImageBGR  = spt_texture(
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
        "texture_strength":   abs(max(-2, min(2, rng.normalvariate(0, 0.5)))),
        "texture_scale":      abs(max(-5, min(5, rng.normalvariate(0, 1)))),
    }
    additive_props = {
        "img"           : bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "max_deviation" : 0.05,
        "n_layers"      : 3,
        "base_radius"   : 1,
    }
    demos = {
        "BASELINE": base_rgba,
        "RANDOM"  : rgb_from_bgr(spt_texture(**random_props)),
        "Additive": rgb_from_bgr(spt_texture_additive(**additive_props)),
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
            spt_texture(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()

