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

def apply_lighting_gradient(img: ImageBGR,
                            top_bright: float = 0.0,
                            bottom_dark: float = 0.0,
                            lighting_mode: str = "linear",
                            gradient_angle: float = 90.0,
                            grad_cx: float = 0.0,
                            grad_cy: float = 0.0,
                            brightness: float = 0.0,
                           ) -> ImageBGR:
    """Apply lighting gradient and global brightness, both symmetric in [-1, 1].
  
    Each parameter uses identical semantics:
        b = -1 -> full black
        b =  0 -> no change
        b = +1  full white

    The gradient linearly interpolates between bottom_dark and top_bright.
    The final global brightness is then applied as a post-bias.

    The algorithm constructs a continuous per-pixel brightness field based on a
    normalized gradient coordinate `u in [0, 1]`, where `u = 0` corresponds to the
    bottom (dark) side and `u = 1` to the top (bright) side of the specified
    lighting direction or radial field. Each endpoint, `bottom_dark` and
    `top_bright`, defines a local brightness bias in the range [-1, 1], interpreted
    symmetrically: negative values darken by scaling the pixel intensity by
    (1 + b), and positive values brighten by linearly reducing the distance to
    white as `I' = I + b * (255 - I)`. The effective local brightness coefficient
    at every pixel is obtained by linear interpolation between these two endpoint
    values, producing a smooth gradient of brightness bias. This field is then
    applied to the input image on a per-pixel basis, and an optional global
    `brightness` adjustment (with identical [-1, 1] semantics) is applied
    afterward to uniformly shift the overall exposure without altering the
    gradient contrast.

    Args:
        img: Input image (uint8, RGB or BGR).
        top_bright: Brightness-style adjustment at gradient top ([-1, 1]).
        bottom_dark: Brightness-style adjustment at gradient bottom ([-1, 1]).
        lighting_mode: "linear" or "radial" gradient pattern.
        gradient_angle: For linear mode, direction in degrees.
        grad_cx, grad_cy: Center offsets (for radial mode).
        brightness: Global brightness adjustment in [-1, 1].
  """
    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    angle_rad = math.radians(gradient_angle)

    # --- Base gradient field u in [0, 1] ---
    if lighting_mode == "linear":
        # "bottom-to-top" convention
        u = (-np.cos(angle_rad) * (x / w) + np.sin(angle_rad) * (y / h))
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
    else:  # radial
        cx = (0.5 + grad_cx) * w
        cy = (0.5 - grad_cy) * h
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        r_max = np.max(
            np.sqrt((corners[:, 0] - cx) ** 2 + (corners[:, 1] - cy) ** 2)
        )
        u = np.clip(r / (r_max + 1e-9), 0.0, 1.0)

    # --- Interpolated local brightness adjustment in [-1, 1] ---
    local_brightness = bottom_dark + (top_bright - bottom_dark) * (1 - u)

    img_f = img.astype(np.float32)

    # --- Apply per-pixel brightness (same rule as global brightness) ---
    out = img_f.copy()
    neg_mask = local_brightness < 0
    pos_mask = local_brightness >= 0

    if np.any(neg_mask):
        b_neg = (1.0 + local_brightness[neg_mask])[..., None]
        out[neg_mask] = img_f[neg_mask] * b_neg

    if np.any(pos_mask):
        b_pos = local_brightness[pos_mask][..., None]
        out[pos_mask] = img_f[pos_mask] + b_pos * (255.0 - img_f[pos_mask])

    # --- Global brightness applied after gradient ---
    b = float(np.clip(brightness, -1.0, 1.0))
    if b > 0.0:
        out = out + b * (255.0 - out)
    elif b < 0.0:
        out = out * (1.0 + b)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    proc_bgr:  ImageBGR  = apply_lighting_gradient(
                               img=base_bgr,
                               top_bright=1.1,
                               bottom_dark=0.9,
                               lighting_mode="linear",
                               gradient_angle=90,
                               grad_cx=0,
                               grad_cy=0,
                           )
    rng = random.Random(os.getpid() ^ int(time.time()))
    delta = 1 + max(-1, min(1, 0.25 * rng.normalvariate(0, 1)))
    random_props = {
        "img":            bgr_from_rgba(render_scene(
                              canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                              plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
                          )),
        "top_bright":     0.5 * delta,
        "bottom_dark":    -0.5 * delta,
        "lighting_mode":  rng.choice(["linear", "radial"]),
        "gradient_angle": rng.randint(-180, 180),
        "grad_cx":        max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1))),
        "grad_cy":        max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1))),
        "brightness":     max(-0.4 * delta, min(0.4 * delta, 0.2 * rng.normalvariate(0, 1))),
    }
    demos = {
        "BASELINE": base_rgba,
        "RANDOM":   rgb_from_bgr(apply_lighting_gradient(**random_props)),
    }

    default_props = {
        "img":            base_bgr,
        "top_bright":     0.0,
        "bottom_dark":    0.0,
        "lighting_mode":  "",
        "gradient_angle": 90,
        "grad_cx":        0,
        "grad_cy":        0,
        "brightness":     0,
    }

    default_props["lighting_mode"] = "linear"    
    demo_set = [
        {"top_bright": +0.0, "bottom_dark": +0.0, "gradient_angle": 90, "brightness": 0.75},
        {"top_bright": +0.1, "bottom_dark": -0.1, "gradient_angle": 90},
        {"top_bright": +0.3, "bottom_dark": -0.7, "gradient_angle": 90},
        {"top_bright": +0.3, "bottom_dark": -0.7, "gradient_angle": 45},
    ]
    for custom_props in demo_set:
        title = (
            f"Linear {int(custom_props['gradient_angle'])}deg x "
            f"{float(custom_props['top_bright']):.1f}x{float(custom_props['bottom_dark']):.1f}"
        )
        print(title)
        demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    default_props["lighting_mode"] = "radial"
    demo_set = [
        {"top_bright": +0.0, "bottom_dark": +0.0, "grad_cx": 0,   "grad_cy": 0, "brightness": -0.75},
        {"top_bright": +0.5, "bottom_dark": -0.5, "grad_cx": 0,   "grad_cy": 0},
        {"top_bright": +0.5, "bottom_dark": -0.5, "grad_cx": 0.5, "grad_cy": 0.5},
        {"top_bright": +0.5, "bottom_dark": -0.5, "grad_cx": 1,   "grad_cy": 1},
    ]
    for custom_props in demo_set:
        title = (
            f"Radial {float(custom_props['grad_cx']):.1f}x{float(custom_props['grad_cy']):.1f} x "
            f"{float(custom_props['top_bright']):.1f}x{float(custom_props['bottom_dark']):.1f}"
        )
        print(title)
        demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()
