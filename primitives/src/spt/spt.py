"""
spt.py
-----------
"""

from __future__ import annotations

import os
import sys
import time
import random
import logging

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
from logging_utils import configure_logging

import spt_config
from mpl_utils import (
    # Conversion helpers
    bgr_from_rgba, rgb_from_bgr,
    # Rendering helpers
    show_RGBx_grid, render_scene,
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


class SPTPipeline:
    """High-level orchestrator for synthetic photo tool pipeline."""

    rng: RNG = get_rng(thread_safe=True)  # class-level RNG shared by all instances

    def __init__(self):
        self.pid = os.getpid()
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            log_path = configure_logging(
                level=logging.DEBUG,
                name=self.__class__.__name__,
                run_prefix=f"{self.__class__.__name__}_{self.pid}"
            )            

    @classmethod
    def reseed(cls, seed: int = None) -> None:
        """Re-seed the internal RNG (for deterministic replay)."""
        cls.rng.seed(seed)

    def clamped_normal(self, sigma=1, amp=1):
        return max(-amp, min(amp, self.rng.normalvariate(0, sigma)))

    # ---- Stages ----

    def stage1_lighting(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies lighting gradient"""
        rng: RNG = self.__class__.rng                
        delta = 1 + self.clamped_normal(0.25)
        meta: dict = {
            "top_bright":     kwargs.get("top_bright", 0.5 * delta),
            "bottom_dark":    kwargs.get("bottom_dark", -0.5 * delta),
            "lighting_mode":  kwargs.get("lighting_mode",
                                          self.rng.choice(["linear", "radial"])),
            "gradient_angle": kwargs.get("gradient_angle", self.rng.randint(-180, 180)),
            "grad_cx":        kwargs.get("grad_cx", self.clamped_normal(0.4, 1.5)),
            "grad_cy":        kwargs.get("grad_cy", self.clamped_normal(0.4, 1.5)),
            "brightness":     kwargs.get("brightness",
                                          self.clamped_normal(0.2, 0.5 * delta)),
        }
        return meta, spt_lighting(img, **meta)

    def stage2_texture(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies paper texture"""
        rng: RNG = self.__class__.rng                
        meta: dict = {
            "texture_strength": kwargs.get("texture_strength",
                                            abs(self.clamped_normal(0.5, 2))),
            "texture_scale":    kwargs.get("texture_scale",
                                            abs(self.clamped_normal(1, 8))),
        }
        return meta, spt_texture(img, **meta)

    def stage3_noise(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies noise"""
        rng: RNG = self.__class__.rng                
        meta: dict = {
            "poisson":     kwargs.get("poisson", self.rng.choice([False, True])),
            "gaussian":    kwargs.get("gaussian", abs(self.clamped_normal(0.2))),
            "sp_amount":   kwargs.get("sp_amount", abs(self.clamped_normal(0.2))),
            "speckle_var": kwargs.get("speckle_var", abs(self.clamped_normal(0.2))),
            "blur_sigma":  kwargs.get("blur_sigma", abs(self.clamped_normal(0.2))),
        }
        return meta, spt_noise(img, **meta)

    def stage4_geometry(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies geometric effects."""
        rng: RNG = self.__class__.rng                
        meta: dict = {
            "tilt_x": kwargs.get("tilt_x", self.clamped_normal(0.25)),
            "tilt_y": kwargs.get("tilt_y", self.clamped_normal(0.25)),
            "k1":     kwargs.get("k1",     self.clamped_normal(0.25)),
            "k2":     kwargs.get("k2",     self.clamped_normal(0.25)),
        }
        return meta, spt_geometry(img, **meta)

    def stage5_color(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies color effects."""
        rng: RNG = self.__class__.rng                
        meta: dict = {
            "vignette_strength": kwargs.get("vignette_strength", 
                                             abs(self.clamped_normal(0.1, 0.5))),
            "warm_strength":     kwargs.get("warm_strength",
                                             abs(self.clamped_normal(0.1, 0.5))),
        }
        return meta, spt_vignette_and_color(img, **meta)

    # ---- Pipeline ----
    def run(self, img: ImageBGR, **kwargs):
        meta = {}
        meta["1 - Lighting"], img1 = self.stage1_lighting(img, **kwargs)
        meta["2 - Texture"],  img2 = self.stage2_texture(img1, **kwargs)
        meta["3 - Noise"],    img3 = self.stage3_noise(img2, **kwargs)
        meta["4 - Geometry"], img4 = self.stage4_geometry(img3, **kwargs)
        meta["5 - Color"],    img5 = self.stage5_color(img4, **kwargs)
        
        if not spt_config.BATCH_MODE:
            stages = {
                "0 - Matplotlib": rgb_from_bgr(img),
                "1 - Lighting":   rgb_from_bgr(img1),
                "2 - Texture":    rgb_from_bgr(img2),
                "3 - Noise":      rgb_from_bgr(img3),
                "4 - Geometry":   rgb_from_bgr(img4),
                "5 - Color":      rgb_from_bgr(img5),
            }
            show_RGBx_grid(stages, n_columns=3)
        return img5


def main():
    rng = random.Random(os.getpid() ^ int(time.time()))
    canvas_bg_idx = rng.randrange(len(PAPER_COLORS))
    plot_bg_idx = rng.randrange(len(PAPER_COLORS))
    base_rgba = render_scene(canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx)
    stage0_mpl = bgr_from_rgba(base_rgba)

    pipeline: SPTPipeline = SPTPipeline()
    pipeline.run(stage0_mpl)


if __name__ == "__main__":
    main()
