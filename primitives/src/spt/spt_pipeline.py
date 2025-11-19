"""
spt_pipeline.py
-----------
"""

from __future__ import annotations

import os
import sys
import time
import random

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
from spt_lighting import spt_lighting
from spt_texture  import spt_texture
from spt_noise    import spt_noise
from spt_geometry import spt_geometry
from spt_color    import spt_vignette_and_color


def main():
    clamped_normal = lambda sigma=1, amp=1: max(-amp, min(amp, rng.normalvariate(0, sigma)))
    # ----------------------------------------------------------------------
    # Stage 0. Matplotlib and Background Color
    # ----------------------------------------------------------------------
    rng = random.Random(os.getpid() ^ int(time.time()))
    canvas_bg_idx     = rng.randrange(len(PAPER_COLORS))
    plot_bg_idx       = rng.randrange(len(PAPER_COLORS))
    base_rgba         = render_scene(canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx)
                      
    stage0_mpl        = bgr_from_rgba(base_rgba)

    # ----------------------------------------------------------------------
    # Stage 1. Lighting
    # ----------------------------------------------------------------------
    delta             = 1 + clamped_normal(0.25)
    top_bright        = 0.5 * delta
    bottom_dark       = -0.5 * delta
    lighting_mode     = rng.choice(["linear", "radial"])
    gradient_angle    = rng.randint(-180, 180)
    grad_cx           = clamped_normal(0.4, 1.5)
    grad_cy           = clamped_normal(0.4, 1.5)
    brightness        = clamped_normal(0.2, 0.5 * delta)
                      
    stage1_lighting   = spt_lighting(stage0_mpl, top_bright, bottom_dark, lighting_mode,
                                     gradient_angle, grad_cx, grad_cy, brightness)

    # ----------------------------------------------------------------------
    # Stage 2. Texture
    # ----------------------------------------------------------------------
    texture_strength  = abs(clamped_normal(0.5, 2))
    texture_scale     = abs(clamped_normal(1.0, 8))
                      
    stage2_texture    = spt_texture(stage1_lighting, texture_strength, texture_scale)

    # ----------------------------------------------------------------------
    # Stage 3. Noise
    # ----------------------------------------------------------------------
    poisson           = rng.choice([False, True])
    gaussian          = abs(clamped_normal(0.2))
    sp_amount         = abs(clamped_normal(0.2))
    speckle_var       = abs(clamped_normal(0.2))
    blur_sigma        = abs(clamped_normal(0.2))
                    
    stage3_noise      = spt_noise(stage2_texture, poisson, gaussian,
                                  sp_amount, speckle_var, blur_sigma)

    # ----------------------------------------------------------------------
    # Stage 4. Geometry
    # ----------------------------------------------------------------------
    tilt_x            = clamped_normal(0.25)
    tilt_y            = clamped_normal(0.25)
    k1                = clamped_normal(0.25)
    k2                = clamped_normal(0.25)
                     
    stage4_geometry   = spt_geometry(stage3_noise, tilt_x, tilt_y, k1, k2)

    # ----------------------------------------------------------------------
    # Stage 5. Color
    # ----------------------------------------------------------------------
    vignette_strength = abs(clamped_normal(0.1, 0.5))
    warm_strength     = abs(clamped_normal(0.1, 0.5))

    stage5_color      = spt_vignette_and_color(stage4_geometry, vignette_strength,
                                               warm_strength)

    demos = {
        "0 - Matplotlib": rgb_from_bgr(stage0_mpl),
        "1 - Lighting":   rgb_from_bgr(stage1_lighting),
        "2 - Texture":    rgb_from_bgr(stage2_texture),
        "3 - Noise":      rgb_from_bgr(stage3_noise),
        "4 - Geometry":   rgb_from_bgr(stage4_geometry),
        "5 - Color":      rgb_from_bgr(stage5_color),
    }

    show_RGBx_grid(demos, n_columns=3)


if __name__ == "__main__":
    main()
