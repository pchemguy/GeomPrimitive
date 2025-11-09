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
from spt_base_gradient import apply_lighting_gradient
from spt_base_texture  import apply_texture


def main():
    # ----------------------------------------------------------------------
    # Stage 0. Matplotlob and Background Color
    # ----------------------------------------------------------------------
    rng = random.Random(os.getpid() ^ int(time.time()))
    canvas_bg_idx = rng.randrange(len(PAPER_COLORS))
    plot_bg_idx   = rng.randrange(len(PAPER_COLORS))
    base_rgba     = render_scene(
        canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx
    )
    base_bgr = bgr_from_rgba(base_rgba)

    # ----------------------------------------------------------------------
    # Stage 1. Gradient
    # ----------------------------------------------------------------------
    delta            = 1 + max(-1, min(1, 0.25 * rng.normalvariate(0, 1)))
    top_bright       = 0.5 * delta
    bottom_dark      = -0.5 * delta
    lighting_mode    = rng.choice(["linear", "radial"])
    gradient_angle   = rng.randint(-180, 180)
    grad_cx          = max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1)))
    grad_cy          = max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1)))
    brightness       = max(-0.5 * delta, min(0.5 * delta, rng.normalvariate(0, 0.2)))
                     
    stage1_light     = apply_lighting_gradient(base_bgr, top_bright, bottom_dark,
                                              lighting_mode, gradient_angle,
                                              grad_cx, grad_cy, brightness)

    # ----------------------------------------------------------------------
    # Stage 2. Texture
    # ----------------------------------------------------------------------
    texture_strength = abs(max(-0.5, min(0.5, rng.normalvariate(0, 0.1))))
    texture_scale    = abs(max(5, min(5, rng.normalvariate(0, 1))))

    stage2_texture   = apply_texture(stage1_light, texture_strength, texture_scale)

    # ----------------------------------------------------------------------
    # Stage 3. Noise
    # ----------------------------------------------------------------------
    poisson          = rng.choice([False, True])
    gaussian         = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    sp_amount        = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    speckle_var      = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    blur_sigma       = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))

    stage3_noise     = apply_noise(stage2_texture, poisson, gaussian,
                                   sp_amount, speckle_var, blur_sigma)

    # ----------------------------------------------------------------------
    # Stage 4. Geometry
    # ----------------------------------------------------------------------







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

"""
    img = self.from_rgba(rgba)
    img = self.apply_paper_lighting(img)
    img = self.add_paper_texture(img)
    img = self.apply_sensor_noise(img)
    img = self.apply_camera_effects(img)
    img = self.add_vignette_and_color_shift(img)
"""