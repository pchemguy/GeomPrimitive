"""
spt_base_gradient.py
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

from spt_base           import *
from spt_base_gradient  import apply_lighting_gradient
from spt_base_texture   import apply_texture
from spt_base_noise     import apply_noise
from spt_base_optics    import apply_camera_effects
from spt_base_postoptic import apply_vignette_and_color_shift


def main():
    # ----------------------------------------------------------------------
    # Stage 0. Matplotlob and Background Color
    # ----------------------------------------------------------------------
    rng = random.Random(os.getpid() ^ int(time.time()))
    canvas_bg_idx     = rng.randrange(len(PAPER_COLORS))
    plot_bg_idx       = rng.randrange(len(PAPER_COLORS))
    base_rgba         = render_scene(canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx)
                      
    stage0_mpl        = bgr_from_rgba(base_rgba)

    # ----------------------------------------------------------------------
    # Stage 1. Lighting
    # ----------------------------------------------------------------------
    delta             = 1 + max(-1, min(1, 0.25 * rng.normalvariate(0, 1)))
    top_bright        = 0.5 * delta
    bottom_dark       = -0.5 * delta
    lighting_mode     = rng.choice(["linear", "radial"])
    gradient_angle    = rng.randint(-180, 180)
    grad_cx           = max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1)))
    grad_cy           = max(-1.5, min(1.5, 0.4 * rng.normalvariate(0, 1)))
    brightness        = max(-0.5 * delta, min(0.5 * delta, rng.normalvariate(0, 0.2)))
                      
    stage1_lighting   = apply_lighting_gradient(stage0_mpl, top_bright, bottom_dark,
                                                lighting_mode, gradient_angle,
                                                grad_cx, grad_cy, brightness)

    # ----------------------------------------------------------------------
    # Stage 2. Texture
    # ----------------------------------------------------------------------
    texture_strength  = abs(max(-0.5, min(0.5, rng.normalvariate(0, 0.1))))
    texture_scale     = abs(max(5, min(5, rng.normalvariate(0, 1))))
                      
    stage2_texture    = apply_texture(stage1_lighting, texture_strength, texture_scale)

    # ----------------------------------------------------------------------
    # Stage 3. Noise
    # ----------------------------------------------------------------------
    poisson           = rng.choice([False, True])
    gaussian          = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    sp_amount         = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    speckle_var       = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
    blur_sigma        = abs(max(-1, min(1, rng.normalvariate(0, 0.2))))
                    
    stage3_noise      = apply_noise(stage2_texture, poisson, gaussian,
                                    sp_amount, speckle_var, blur_sigma)

    # ----------------------------------------------------------------------
    # Stage 4. Geometry
    # ----------------------------------------------------------------------
    tilt_x            = max(-1, min(1, rng.normalvariate(0, 0.25)))
    tilt_y            = max(-1, min(1, rng.normalvariate(0, 0.25)))
    k1                = max(-1, min(1, rng.normalvariate(0, 0.25)))
    k2                = max(-1, min(1, rng.normalvariate(0, 0.25)))
                     
    stage4_geometry   = apply_camera_effects(stage3_noise, tilt_x, tilt_y, k1, k2)

    # ----------------------------------------------------------------------
    # Stage 5. Color
    # ----------------------------------------------------------------------
    vignette_strength = abs(max(-0.5, min(0.5, rng.normalvariate(0, 0.1))))
    warm_strength     = abs(max(-0.5, min(0.5, rng.normalvariate(0, 0.1))))

    stage5_color      = apply_vignette_and_color_shift(stage4_geometry,
                                                       vignette_strength, warm_strength)

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
