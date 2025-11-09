"""
spt.py
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

from rng import RNG, get_rng

from spt_base import (
    # Conversion helpers
    bgr_from_rgba, rgb_from_bgr, show_RGBx_grid, render_scene,
    # Type aliases
    ImageBGR, ImageRGB, ImageRGBA, ImageRGBx,
    # Constants
    PAPER_COLORS,
)
from spt_lighting import spt_lighting
from spt_texture  import spt_texture
from spt_noise    import spt_noise
from spt_geometry import spt_geometry
from spt_color    import spt_vignette_and_color



def clamped_normal(self, sigma=1, amp=1):
    return max(-amp, min(amp, self.rng.normalvariate(0, sigma)))


class SPTPipeline:
    """High-level orchestrator for synthetic photo tool pipeline."""

    rng: RNG = get_rng(thread_safe=True)  # class-level RNG shared by all instances

    def __init__(self):
        pass

    @classmethod
    def reseed(cls, seed: int = None) -> None:
        """Re-seed the internal RNG (for deterministic replay)."""
        cls.rng.seed(seed)

    # ---- Stages ----
    def stage0_scene(self):
        canvas_idx = self.rng.randrange(len(PAPER_COLORS))
        plot_idx   = self.rng.randrange(len(PAPER_COLORS))
        rgba = render_scene(canvas_bg_idx=canvas_idx, plot_bg_idx=plot_idx)
        return bgr_from_rgba(rgba)

    def stage1_lighting(self, img):
        delta = max(0.5, min(1.5, 1 + self.clamped_normal(0.25)))
        return apply_lighting_gradient(
            img,
            top_bright=0.5 * delta,
            bottom_dark=-0.5 * delta,
            lighting_mode=self.rng.choice(["linear", "radial"]),
            gradient_angle=self.rng.randint(-180, 180),
            grad_cx=self.clamped_normal(0.4, 1.5),
            grad_cy=self.clamped_normal(0.4, 1.5),
            brightness=self.clamped_normal(0.2, 0.5 * delta),
        )

    def stage2_texture(self, img):
        return apply_texture(
            img,
            texture_strength=abs(self.clamped_normal(0.5, 2)),
            texture_scale=abs(self.clamped_normal(0.5, 8)),
        )

    def stage3_noise(self, img):
        return apply_noise(
            img,
            poisson=self.rng.choice([False, True]),
            gaussian=abs(self.clamped_normal(0.2)),
            sp_amount=abs(self.clamped_normal(0.2)),
            speckle_var=abs(self.clamped_normal(0.2)),
            blur_sigma=abs(self.clamped_normal(0.2)),
        )

    def stage4_geometry(self, img):
        return apply_camera_effects(
            img,
            tilt_x=self.clamped_normal(0.25),
            tilt_y=self.clamped_normal(0.25),
            k1=self.clamped_normal(0.25),
            k2=self.clamped_normal(0.25),
        )

    def stage5_color(self, img):
        return apply_vignette_and_color_shift(
            img,
            vignette_strength=abs(self.clamped_normal(0.1, 0.5)),
            warm_strength=abs(self.clamped_normal(0.1, 0.5)),
        )

    # ---- Pipeline ----
    def run(self, show=True):
        stages = {}
        stages["0 - Matplotlib"] = img0 = self.stage0_scene()
        stages["1 - Lighting"]   = img1 = self.stage1_lighting(img0)
        stages["2 - Texture"]    = img2 = self.stage2_texture(img1)
        stages["3 - Noise"]      = img3 = self.stage3_noise(img2)
        stages["4 - Geometry"]   = img4 = self.stage4_geometry(img3)
        stages["5 - Color"]      = img5 = self.stage5_color(img4)

        if show:
            show_RGBx_grid({k: rgb_from_bgr(v) for k, v in stages.items()}, n_columns=3)
        return img5








def main():
    pass

if __name__ == "__main__":
    main()
