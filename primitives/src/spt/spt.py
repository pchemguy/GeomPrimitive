"""
spt.py
-----------
"""

from __future__ import annotations

__all__ = ["SPTPipeline",]

import os
import sys
import json
import time
import random
import logging

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt

from utils.rng import RNG, get_rng
from utils.logging_utils import configure_logging

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
from spt_texture  import spt_texture
from spt_lighting import spt_lighting
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
        return max(-amp, min(amp, self.__class__.rng.normalvariate(0, sigma)))

    # ---- Stages ----

    def stage1_texture(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies paper texture"""
        self.logger.debug(f"Running stage 2: Texture.")
        meta: dict = {
            "texture_strength": kwargs.get("texture_strength",
                                            abs(self.clamped_normal(0.5, 2))),
            "texture_scale":    kwargs.get("texture_scale",
                                            abs(self.clamped_normal(1, 8))),
        }
        return meta, spt_texture(img, **meta)

    def stage2_lighting(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies lighting gradient"""
        rng: RNG = self.__class__.rng                
        self.logger.debug(f"Running stage 1: Lighting.")
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

    def stage3_noise(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies noise"""
        rng: RNG = self.__class__.rng                
        self.logger.debug(f"Running stage 3: Noise.")
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
        self.logger.debug(f"Running stage 4: Geometry.")
        meta: dict = {
            "tilt_x": kwargs.get("tilt_x", self.clamped_normal(0.25)),
            "tilt_y": kwargs.get("tilt_y", self.clamped_normal(0.25)),
            "k1":     kwargs.get("k1",     self.clamped_normal(0.25)),
            "k2":     kwargs.get("k2",     self.clamped_normal(0.25)),
        }
        return meta, spt_geometry(img, **meta)

    def stage5_color(self, img: ImageBGR, **kwargs) -> tuple[dict, ImageBGR]:
        """Applies color effects."""
        self.logger.debug(f"Running stage 5: Color.")
        meta: dict = {
            "vignette_strength": kwargs.get("vignette_strength", 
                                             abs(self.clamped_normal(0.1, 0.5))),
            "warm_strength":     kwargs.get("warm_strength",
                                             abs(self.clamped_normal(0.1, 0.5))),
        }
        return meta, spt_vignette_and_color(img, **meta)

    # ---- Pipeline ----
    def run(self, img: ImageBGR = None, **kwargs):
        meta = {}
        runtime = {}
        images = {}
        stages = [
            ("1 - Texture",  self.stage1_texture),
            ("2 - Lighting", self.stage2_lighting),
            ("3 - Noise",    self.stage3_noise),
            ("4 - Geometry", self.stage4_geometry),
            ("5 - Color",    self.stage5_color),
        ]        

        total = 0
        name = "0 - Matplotlib"
        if not img:
            self.logger.warning(f"No Matplotlib RGBA image is provided. Using a dummy generator.")
            t0 = time.perf_counter()
            canvas_bg_idx = self.rng.randrange(len(PAPER_COLORS))
            plot_bg_idx = self.rng.randrange(len(PAPER_COLORS))
            base_rgba = render_scene(canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx)
            stage0_mpl = bgr_from_rgba(base_rgba)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            runtime[name] = round(elapsed_ms, 2)
            total += runtime[name]
            out_img = stage0_mpl
        else:
            out_img = img

        images[name] = rgb_from_bgr(out_img)

        for name, stage_fn in stages:
            t0 = time.perf_counter()
            stage_meta, out_img = stage_fn(out_img, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Record metadata + timing
            images[name] = rgb_from_bgr(out_img)
            runtime[name] = round(elapsed_ms, 3)
            meta[name] = stage_meta
            total += runtime[name]

        runtime["Total"] = round(total, 3)
        
        if not spt_config.BATCH_MODE:
            summary = [""]
            summary.append("=" * 80)
            summary.append(f"{'PIPELINE PERFORMANCE SUMMARY':^40}")
            summary.append("=" * 40)

            key_width = 20

            summary.append(f"{'Performance Summary':-^35}")

            for k, v in runtime.items():
                summary.append(f"  {k:<{key_width}}: {v:10.3f} ms")

            for name, stage_meta in meta.items():
                summary.append(
                    f"\n{name:-^35}")
                for k, v in stage_meta.items():
                    if isinstance(v, int):
                        val = f"{v:>6}    "
                    elif isinstance(v, float):
                        val = f"{v:10.3f}    "
                    else:
                        val = v
                    summary.append(f"  {k:<{key_width}}: {val}")
            summary.append("=" * 80)
            self.logger.debug('\n'.join(summary))
            show_RGBx_grid(images, n_columns=3)
        
        return out_img, meta


def main():
    pipeline: SPTPipeline = SPTPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
