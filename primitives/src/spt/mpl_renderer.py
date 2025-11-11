"""
mpl_renderer.py
-----------
"""

from __future__ import annotations

import os
import sys
import time
import random
import logging
from enum import Enum, auto
from typing import Optional, Union
from collections.abc import Iterable

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath
from matplotlib.transforms import Affine2D

from rng import RNG, get_rng
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

numeric = Union[int, float]
PointXY = tuple[numeric, numeric]
CoordRange = tuple[numeric, numeric]


def clamped_normal(self, sigma=1, amp=1):
    return max(-amp, min(amp, self.rng.normalvariate(0, sigma)))


class MPLRenderer:
    """
    Generator of basic Matplotlib scenes
    
    Prepares a Matplotlib scene composed of geometric primitives:
     - Basic geometric shapes (lines, triangles, rectangles, ellipsoidal arcs.
     - Tabulated smooth functions {(x, y) pairs or (x, y, y') triples.
     - Randomizes scale, orientation, position, kind, style.
     - Uses Matplotlib provided cubic Bezier curves to imitate hand-drawn lines.
     - Introduces random jitter to shapes and angles, imitating non-ideal rendering.
     - Selects a random background from a predefined list.
    """

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

    def render_scene() -> ImageRGBA:
        """Renders Matplotlib scene as RGBA"""
        # Dummy stub
        return self.render_dummy_scene()

    def render_dummy_scene() -> ImageRGBA:
        """Renders dummy Matplotlib scene as RGBA"""
        rng: RNG = self.__class__.rng
        return render_scene(
                   canvas_bg_idx=rng.randrange(len(PAPER_COLORS)),
                   plot_bg_idx=rng.randrange(len(PAPER_COLORS)),
               )
