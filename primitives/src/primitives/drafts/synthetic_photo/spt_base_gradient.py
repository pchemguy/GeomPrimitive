"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import math
from typing import TypeAlias, Sequence, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spt_base import *

ImageBGR:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) BGR order
ImageRGB:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) RGB order
ImageRGBA: TypeAlias = NDArray[np.uint8]  # (H, W, 4) RGBA order
ImageRGBx: TypeAlias = Union[ImageRGB, ImageRGBA] # Either RGB or RGBA


def apply_lighting_gradient(img: ImageBGR,
                            top_bright: float = 1.1,
                            bottom_dark: float = 0.9,
                            lighting_mode: str = "linear",
                            lighting_strength: float = 5,
                            gradient_angle: float = 90,
                            grad_cx: float = 0,
                            grad_cy: float = 0,
                           ) -> ImageBGR:
    """Apply lighting gradient (linear or radial, normalized)."""
    if (not lighting_mode) or (lighting_strength <= 1e-6):
        return img

    angle_rad = math.radians(gradient_angle)

    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    # --- Base gradient shape ---
    if lighting_mode == "linear":
        # Note: Leading signs determine axis direction.
        # This sign combination sets
        #   - top bright for 90 deg, as expected AND
        #   - from bottom-left to top-right for 45 deg
        #     (standard direction for positive angle in conventional coordinates)
        u = (- np.cos(angle_rad) * (x / w)
             + np.sin(angle_rad) * (y / h))
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
    else:  # Radial
        cx = (0.5 + grad_cx) * w
        cy = (0.5 - grad_cy) * h # Use minus to direct vertical axis bottom to top
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        r_max = np.max(np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2))
        u = np.clip(r / (r_max + 1e-9), 0.0, 1.0)

    # --- Compute lighting map ---
    grad = top_bright + (bottom_dark - top_bright) * u
    # Interpolate between flat (1.0) and full gradient
    lighting = 1.0 + lighting_strength * (grad - 1.0)

    img_f = np.clip(img.astype(np.float32) * lighting[..., None], 0, 255)

    return img_f.astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_lighting_gradient(
                               img=base_bgr,
                               top_bright=1.1,
                               bottom_dark=0.9,
                               lighting_mode="linear",
                               lighting_strength=4,
                               gradient_angle=90,
                               grad_cx=0,
                               grad_cy=0,
                           )
    default_props = {
        "img":               base_bgr,
        "top_bright":        1.1,
        "bottom_dark":       0.9,
        "lighting_mode":     "",
        "lighting_strength": 1,
        "gradient_angle":    90,
        "grad_cx":           0,
        "grad_cy":           0,
    }

    default_props["lighting_mode"] = "linear"
    linear_demos = {
        "Linear 90deg x 0": {"lighting_strength": 0, "gradient_angle": 90},
        "Linear 90deg x 1": {"lighting_strength": 1, "gradient_angle": 90},
        "Linear 90deg x 5": {"lighting_strength": 5, "gradient_angle": 90},
        "Linear 45deg x 5": {"lighting_strength": 5, "gradient_angle": 45},
    }
    for (title, custom_props) in linear_demos.items():
        linear_demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    default_props["lighting_mode"] = "radial"
    radial_demos = {
        "Radial 0x0 x 1":       {"lighting_strength": 1, "grad_cx": 0,   "grad_cy": 0},
        "Radial 0x0 x 5":       {"lighting_strength": 5, "grad_cx": 0,   "grad_cy": 0},
        "Radial 0.5x0.5 x 5":   {"lighting_strength": 5, "grad_cx": 0.5, "grad_cy": 0.5},
        "Radial 1x1 x 5":       {"lighting_strength": 5, "grad_cx": 1,   "grad_cy": 1},
    }
    for (title, custom_props) in radial_demos.items():
        radial_demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    show_RGBx_grid({**linear_demos, **radial_demos}, n_columns=4)


if __name__ == "__main__":
    main()
