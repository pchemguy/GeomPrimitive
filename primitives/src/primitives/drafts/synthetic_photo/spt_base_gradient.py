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


import numpy as np
import math
from typing import Literal

ImageBGR = np.ndarray


def apply_lighting_gradient(
    img: ImageBGR,
    top_bright: float = 1.1,
    bottom_dark: float = 0.9,
    lighting_mode: Literal["linear", "radial"] = "linear",
    gradient_angle: float = 90.0,
    grad_cx: float = 0.0,
    grad_cy: float = 0.0,
    brightness: float = 0.0,
) -> ImageBGR:
  """Apply physically consistent lighting gradient + global brightness bias.

  The gradient is fully defined by (top_bright, bottom_dark).
  The global brightness acts symmetrically:
    - brightness > 0 -> lighten: I' = I + b * (255 - I)
    - brightness < 0 -> darken:  I' = I * (1 + b)

  Args:
      img: Input image (uint8, BGR or RGB).
      top_bright: Multiplicative factor for the bright end (e.g. 1.1).
      bottom_dark: Multiplicative factor for the dark end (e.g. 0.9).
      lighting_mode: "linear" or "radial" gradient.
      gradient_angle: For linear mode, direction of the gradient in degrees.
      grad_cx, grad_cy: Center offsets (fractional) for radial mode.
      brightness: Global brightness in [-1, 1].
                  -1 -> full black, 0 -> neutral, +1 -> full white.

  Returns:
      Adjusted image, uint8, same shape as input.
  """
  h, w = img.shape[:2]
  y, x = np.indices((h, w), dtype=np.float32)
  angle_rad = math.radians(gradient_angle)

  # --- Base gradient field u in [0, 1] ---
  if lighting_mode == "linear":
    # "bottom-to-top" convention
    u = (-np.cos(angle_rad) * (x / w) + np.sin(angle_rad) * (y / h))
    u = (u - u.min()) / (u.max() - u.min() + 1e-9)
  else:  # radial gradient
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

  # --- Gradient map: blend between bottom_dark and top_bright ---
  grad = top_bright + (bottom_dark - top_bright) * u
  grad = np.clip(grad, 0.0, None)

  # --- Apply gradient (per pixel, per channel) ---
  img_f = img.astype(np.float32)
  out = img_f * grad[..., None]

  # --- Global brightness bias ---
  b = float(np.clip(brightness, -1.0, 1.0))
  if b > 0.0:
    # brighten toward white
    out = out + b * (255.0 - out)
  elif b < 0.0:
    # darken toward black
    out = out * (1.0 + b)

  return np.clip(out, 0.0, 255.0).astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_lighting_gradient(
                               img=base_bgr,
                               top_bright=1.1,
                               bottom_dark=0.9,
                               lighting_mode="linear",
                               gradient_angle=90,
                               grad_cx=0,
                               grad_cy=0,
                           )
    demos = {}
    default_props = {
        "img":               base_bgr,
        "top_bright":        1.1,
        "bottom_dark":       0.9,
        "lighting_mode":     "",
        "gradient_angle":    90,
        "grad_cx":           0,
        "grad_cy":           0,
    }

    default_props["lighting_mode"] = "linear"    
    demo_set = [
        {"top_bright": 1.0, "bottom_dark": 1.0, "gradient_angle": 90},
        {"top_bright": 1.1, "bottom_dark": 0.9, "gradient_angle": 90},
        {"top_bright": 1.5, "bottom_dark": 0.5, "gradient_angle": 90},
        {"top_bright": 1.5, "bottom_dark": 0.5, "gradient_angle": 45},
    ]
    for custom_props in demo_set:
        title = f"Linear {int(custom_props['gradient_angle'])}deg x {float(custom_props['top_bright']):.1f}-{float(custom_props['bottom_dark']):.1f}"
        demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )
        print(title)

    default_props["lighting_mode"] = "radial"
    demo_set = [
        {"top_bright": 1.0, "bottom_dark": 1.0, "grad_cx": 0,   "grad_cy": 0},
        {"top_bright": 1.1, "bottom_dark": 0.5, "grad_cx": 0,   "grad_cy": 0},
        {"top_bright": 1.5, "bottom_dark": 0.9, "grad_cx": 0.5, "grad_cy": 0.5},
        {"top_bright": 1.5, "bottom_dark": 0.9, "grad_cx": 1,   "grad_cy": 1},
    ]
    for custom_props in demo_set:
        title = f"Radial {int(custom_props['grad_cx'])}x{int(custom_props['grad_cy'])} x {float(custom_props['top_bright']):.1f}-{float(custom_props['bottom_dark']):.1f}"
        demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()
