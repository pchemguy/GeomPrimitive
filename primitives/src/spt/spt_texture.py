"""
spt_texture.py
-----------

https://chatgpt.com/c/69120de6-5468-832d-8bff-88120cb94daa
"""

from __future__ import annotations

__all__ = [
    "spt_texture",
    "spt_texture_additive",
    "spt_texture_multiplicative",
    "spt_texture_combined",
    "spt_texture_presets",
]

import os
import sys
import time
import random
import math
from typing import Union
import numpy as np
from numba import njit
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


# ======================================================================
#  Multiplicative "paper texture" modulation with true blur - slow
# ======================================================================
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
#  Box blur - Fast approximation of a 2D blur
# ======================================================================
def _box_blur(img: ImageBGRF, radius: int) -> ImageBGRF:
    """
    Cheap separable box filter using cumulative sums.
    
    Approximates a 2D blur with a box kernel of size (2*radius) in each
    direction, but without explicit convolution loops.

    For a box blur, each pixel is replaced by the average of
    neighbors within +/-radius in x and y. Direct convolution is
    O(H*W*radius^2). Using prefix sums reduces the per-pixel cost
    to O(1) (after one O(H*W) prefix pass).

    """
    if radius <= 0:
        return img

    h, w = img.shape
    pad = radius

    # Horizontal blur
    #   Reflect padding:
    #   tmp shape: (H, W + 2*pad)
    #   Reflect padding means edges reflect the interior, like [3,4] -> [4,3,4,3] etc.
    #   This avoids edge darkening/brightening.
    #
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")

    #   Cumulative sum along x:
    #   csum[:, j] = sum of tmp[:, 0..j]
    #   This is the standard trick: you can get the sum over any horizontal
    #   interval [a, b] in O(1) as:
    #   sum[a..b] = csum[b] - csum[a-1]
    #   (basically an integral image along one dimension).
    csum = np.cumsum(tmp, axis=1)

    #   Compute sliding window sums with slicing
    #
    left  = csum[:, :-2 * pad]
    right = csum[:, 2 * pad :]
    horz = (right - left) / (2 * pad)

    #     - csum                is (H, W + 2*pad).
    #     - csum[:, :-2*pad] -> shape (H, W) (all columns except the last 2*pad)
    #     - csum[:, 2*pad:]  -> also shape (H, W) (all columns starting from 2*pad)
    #   
    #   So at column index j (0-based in these slices):
    #     - left[:, j] = csum[:, j]
    #     - right[:, j] = csum[:, j + 2*pad]
    #
    #   Interpretation:
    #   We're looking at a window of width 2*pad between indices (j+1)..(j+2*pad) in tmp.
    #
    #   The average over that window is:
    #     (right - left) / (2 * pad)
    #
    #   So horz is a horizontally blurred version of img (box filter of width 2*pad),
    #   back to shape (H, W).
    #
    #   The "center" alignment is slightly shifted compared to a symmetrical [-pad, +pad]
    #   window, but visually for our use (soft paper noise) that's irrelevant. It's
    #   symmetric enough and cheap.

    # Vertical blur
    tmp = np.pad(horz, ((pad, pad), (0, 0)), mode="reflect")
    csum = np.cumsum(tmp, axis=0)
    top  = csum[:-2 * pad, :]
    bot  = csum[2 * pad :, :]
    vert = (bot - top) / (2 * pad)

    #   Same idea, now vertically:
    #     - horz shape: (H, W)
    #     - Pad vertically: tmp shape (H + 2*pad, W)
    #     - csum over axis=0 (rows)
    #     - top = csum[:-2*pad, :] -> shape (H, W)
    #     - bot = csum[2*pad:, :] -> shape (H, W)
    #   Then:
    #     - vert = (bot - top) / (2 * pad)
    #
    #   For each pixel, this is the average over a vertical window of height 2*pad in
    #   the blurred horizontal image.
    #   Result:
    #     - vert shape (H, W)
    #     - Equivalent to applying a 2D box filter of size (2*pad)x(2*pad) but with:
    #         - Reflect padding
    #         - O(H*W) cost via separable prefix sums
    #   
    #    So _box_blur(img, radius) ~ box_filter(img, kernel_size=2*radius).
    
    return vert


# ======================================================================
#  Numba - Reflect index i into range [0, n-1] like NumPy 'reflect'.
# ======================================================================
@njit
def _reflect_index(i: int, n: int) -> int:
    """Reflect index i into range [0, n-1] like NumPy 'reflect'."""
    if i < 0:
        return -i - 1
    if i >= n:
        return 2*n - i - 1
    return i


# ======================================================================
#  Numba - Box blur - Fast approximation of a 2D blur
# ======================================================================
@njit
def _box_blur_numba(img: np.ndarray, radius: int) -> np.ndarray:
    h, w = img.shape
    out_h = np.zeros((h, w), dtype=np.float32)
    temp  = np.zeros((h, w), dtype=np.float32)

    # Horizontal pass
    for y in range(h):
        for x in range(w):
            s = 0.0
            for k in range(x-radius, x+radius):
                s += img[y, _reflect_index(k, w)]
            temp[y, x] = s / (2 * radius)

    # Vertical pass
    for y in range(h):
        for x in range(w):
            s = 0.0
            for k in range(y-radius, y+radius):
                s += temp[_reflect_index(k, h), x]
            out_h[y, x] = s / (2 * radius)

    return out_h


# ======================================================================
#  Additive "paper texture" modulation for an existing image
# ======================================================================
def spt_texture_additive(
        img           : Union[ImageBGR, ImageBGRF],
        *,
        add_strength  : float = 0.05,
        n_layers      : int   = 3,
        base_radius   : int   = 1,
        seed          : int   = None,
    ) -> Union[ImageBGR, ImageBGRF]:
    """
    Apply *additive* paper-like brightness texture to an image (OpenCV BGR).

    Employs:
        - Multi-scale multi-radius smooth-noise accumulation
        - Normalization to zero-mean and scaling by add_strength
        - Cumulative-sum box-blur routine
    Adds generated noise field to the input BGR image.

    Args:
        img:
            Input image in BGR uint8 or float32. Value range 0-255 or 0-1.
        add_strength:
            Max absolute additive brightness variation (relative 0-1 scale).
            For uint8 images this corresponds to +/-(255 * add_strength) shifts.
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
    if add_strength <= 0 or n_layers <= 0:
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
    acc *= float(add_strength)

    # --- Apply additive brightness modulation ---------------------------
    # broadcast acc -> (H,W,1), add to all channels
    out = img_f + acc[..., None]
    out = np.clip(out, 0.0, 1.0)

    # --- Convert back to original dtype --------------------------------
    if img.dtype == np.uint8:
        return (out * 255.0).astype(np.uint8)
    else:
        return out


# ======================================================================
#  Multiplicative "paper texture" modulation for an existing image
# ======================================================================
def spt_texture_multiplicative(
        img           : Union[ImageBGR, ImageBGRF],
        *,
        mult_strength : float = 0.05,
        n_layers      : int   = 3,
        base_radius   : int   = 1,
        seed          : int   = None,
    ) -> Union[ImageBGR, ImageBGRF]:
    """
    Multiplicative texture modulation using the same multi-scale box-blur noise.

    Multiplier field:
        M(x,y) = 1 + mult_strength * N(x,y)
    where N is zero-mean, std=1 noise after multi-scale smoothing.

    Returns same dtype as input.
    """
    if mult_strength <= 0 or n_layers <= 0:
        return img

    # Convert to float [0,1]
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = np.clip(img.astype(np.float32), 0.0, 1.0)

    h, w = img_f.shape[:2]
    rng = np.random.default_rng(seed)

    # Generate multi-scale smooth noise
    acc = np.zeros((h, w), np.float32)
    for i in range(n_layers):
        noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
        radius = base_radius * (2 ** i)
        smooth = _box_blur(noise, radius)
        acc += smooth

    # Normalize and map to multiplicative mask
    acc -= acc.mean()
    acc /= (acc.std() or 1.0)

    M = 1.0 + mult_strength * acc
    out = img_f * M[..., None]
    out = np.clip(out, 0.0, 1.0)

    # Return original dtype
    if img.dtype == np.uint8:
        return (out * 255).astype(np.uint8)
    else:
        return out


# ======================================================================
#  Combined additive + multiplicative paper texture
# ======================================================================
def spt_texture_combined(
        img           : Union[ImageBGR, ImageBGRF],
        *,
        add_strength  : float = 0.04,
        mult_strength : float = 0.07,
        n_layers      : int   = 3,
        base_radius   : int   = 1,
        seed          : int   = None,
    ) -> [ImageBGR, ImageBGRF]:
    """
    Combined additive + multiplicative paper texture.
    Extremely realistic for scanned notebook pages.

    Combined model:
        M(x,y) = mult_strength * N1
        A(x,y) = add_strength  * N2
        out    = img * (1 + M) + A

    N1, N2 are independent correlated noise fields.
    """
    if (add_strength <= 0 and mult_strength <= 0) or n_layers <= 0:
        return img

    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = np.clip(img.astype(np.float32), 0, 1)

    h, w = img_f.shape[:2]
    rng = np.random.default_rng(seed)

    # Generate two independent multi-scale noise fields
    def smooth_noise(seed_offset: int):
        acc = np.zeros((h, w), np.float32)
        local_rng = np.random.default_rng(seed + seed_offset if seed is not None else None)
        for i in range(n_layers):
            radius = base_radius * (2 ** i)
            noise = local_rng.normal(0, 1, size=(h, w)).astype(np.float32)
            acc += _box_blur_numba(noise, radius)
        acc -= acc.mean()
        acc /= (acc.std() or 1.0)
        return acc

    noise_add  = smooth_noise(1000) if add_strength  > 0 else 0.0
    noise_mult = smooth_noise(2000) if mult_strength > 0 else 0.0

    # Apply combined texture
    out = img_f
    if mult_strength > 0:
        out = out * (1.0 + mult_strength * noise_mult[..., None])
    if add_strength > 0:
        out = out + (add_strength * noise_add[..., None])

    out = np.clip(out, 0.0, 1.0)

    if img.dtype == np.uint8:
        return (out * 255).astype(np.uint8)
    else:
        return out


# ======================================================================
#  Parameter sets for combined (add/mult) texture models
# ======================================================================
def spt_texture_presets(name: str) -> dict:
    """
    Return parameter sets for additive/multiplicative texture models.

    Usage:
        cfg = spt_texture_presets("old_paper")
        img2 = spt_texture_combined(img, **cfg)        
    """
    name = name.lower().strip()

    if name == "old paper":
        return dict(
            add_strength=0.06,
            mult_strength=0.10,
            n_layers=4,
            base_radius=3,
            seed=None,
        )

    if name == "scanned sheet":
        return dict(
            add_strength=0.02,
            mult_strength=0.12,
            n_layers=3,
            base_radius=2,
            seed=None,
        )

    if name == "notebook":
        return dict(
            add_strength=0.045,
            mult_strength=0.05,
            n_layers=3,
            base_radius=2,
            seed=None,
        )

    if name == "blueprint background":
        return dict(
            add_strength=0.02,
            mult_strength=0.03,
            n_layers=4,
            base_radius=4,
            seed=None,
        )

    # default clean paper
    return dict(
        add_strength=0.03,
        mult_strength=0.04,
        n_layers=3,
        base_radius=2,
            seed=None,
    )


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

    demos = {
        "BASELINE": base_rgba,
        "RANDOM"  : rgb_from_bgr(spt_texture(**random_props)),
    }

    default_props = {"img": base_bgr,}
    demo_set = [
        {"texture_strength": 0.1, "texture_scale": 8},
        {"texture_strength": 0.2, "texture_scale": 8},
        #{"texture_strength": 0.4, "texture_scale": 8},
        #{"texture_strength": 0.8, "texture_scale": 8},
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

    additive_props = {
        "img"           : bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "add_strength"  : 0.04,
        "n_layers"      : 3,
        "base_radius"   : 2,
    }
    demos["Additive"] = rgb_from_bgr(spt_texture_additive(**additive_props))
    
    multiplicative_props = {
        "img"           : bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "mult_strength" : 0.07,
        "n_layers"      : 3,
        "base_radius"   : 2,
    }
    demos["Multiplicative"] = rgb_from_bgr(spt_texture_multiplicative(**multiplicative_props))
    
    combined_props = {
        "img"           : bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "add_strength"  : 0.04,
        "mult_strength" : 0.05,
        "n_layers"      : 3,
        "base_radius"   : 2,
    }
    demos["Combined"] = rgb_from_bgr(spt_texture_combined(**combined_props))
    
    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()

