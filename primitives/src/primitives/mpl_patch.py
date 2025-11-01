"""
mpl_patch.py
-------

https://chatgpt.com/c/6905add2-1ff8-8328-ba21-6c370614dcc4
"""
from __future__ import annotations

import os
import sys
import math
import logging
from enum import Enum
from typing import Any, Dict, Union, Tuple, Optional, List

import numpy as np
import matplotlib as mpl

# Use a non-interactive backend (safe for multiprocessing workers)
# mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.patches import PathPatch
from matplotlib.axes import Axes
from matplotlib._enums import JoinStyle, CapStyle
from matplotlib.path import Path as mplPath

sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from primitives.base import Primitive
from primitives.rng import RNG
from runner.logging_utils import configure_logging
from runner.config import WorkerConfig

PointXY = Tuple[float, float]

# =============================================================================
# Constants
# =============================================================================
LOGGER_NAME = "worker"
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)


# =============================================================================
# Patch implementation
# =============================================================================
class Patch(Primitive):
    __slots__ = ("patches")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patches = {}

    # -------------------------------------------------------------------------
    # Abstract API stubs
    # -------------------------------------------------------------------------
    def make_geometry(self, ax: Optional[Axes] = None, **kwargs) -> Primitive:
        return self

    def draw(self) -> Primitive:
        return
    
    # -------------------------------------------------------------------------
    # Key functionality
    # -------------------------------------------------------------------------
    def add_patch(self,
                  path: mplPath,
                  linewidth: Optional[float] = None,
                  pattern: Optional[str] = None,
                  color: Optional[str] = None,
                  alpha: Optional[float] = None,
                  capstyle: Optional[str] = None,
                  joinstyle: Optional[str] = None,
                  **style_options
                 ) -> "Patch":
        rng: RNG = self.__class__.rng
        if not isinstance(self._ax, Axes):
            raise TypeError(f"ax is not set")

        ax: Axes = self._ax

        # Color and alpha
        color_name_tuple: Union[str, Tuple] = self._get_color(color)
        fill: bool = style_options.get("fill", False)
        if not fill:
            style_options["fill"] = False
            if "facecolor" in style_options:
                 style_options.pop("facecolor")
            if "fc" in style_options:
                 style_options.pop("fc")
            style_options["color"] = self._get_color(color)
        else:
            style_options["facecolor"] = self._get_color(color)
            if not ("edgecolor" in style_options or "ec" in style_options):
                style_options["edgecolor"] = self._get_color(color)

        if alpha is None or not isinstance(alpha, (int, float)):
            alpha = 1.5 - rng.paretovariate(1.0) * 0.5
        style_options["alpha"] = max(0.0, min(float(alpha), 1.0))

        # Cap and join styles
        style_options["capstyle"] = CapStyle._member_map_.get(str(capstyle).lower()) or rng.choice(list(CapStyle))
        style_options["joinstyle"] = JoinStyle._member_map_.get(str(joinstyle).lower()) or rng.choice(list(JoinStyle))

        style_options["linewidth"] = linewidth or rng.choice(DEFAULT_LINEWIDTHS)
        style_options["linestyle"] = self._get_linestyle(pattern, hand_drawn=True)

        patch: PathPatch = PathPatch(path, **style_options)
        self.patches[str(patch)] = patch

        return self

    def draw_pathes(self) -> None:
        if not isinstance(self._ax, Axes):
            raise TypeError(f"ax is not set")
        for patch in self.patches:
            self._ax.add_patch(patch)

    @classmethod
    def cubic_spline_ex(cls, start: PointXY, end: PointXY, n_segments: int = 5,
                        amp: float = 0.04, tightness: float = 0.3) -> mplPath:
        rng: RNG = cls.rng
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
    
        ts = sorted(rng.random() for _ in range(n_segments - 1))
        ts = [0.0] + ts + [1.0]
        points = [(x0 + dx * t, y0 + dy * t) for t in ts]
    
        verts: List[Tuple[float, float]] = []
        codes: List[int] = []
        for i in range(len(points) - 1):
            segment_path: mplPath = cls._cubic_spline_segment(
                points[i], points[i + 1], amp, tightness
            )
            v, c = segment_path.vertices, segment_path.codes
            if i == 0:
                # first segment: keep everything
                verts.extend(v)
                codes.extend(c)
            else:
                # subsequent: drop the MOVETO + first vertex
                verts.extend(v[1:])
                codes.extend(c[1:])
    
        return mplPath(verts, codes)
            

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    @classmethod
    def _cubic_spline_segment(cls, start: PointXY, end: PointXY, amp: float = 0.04,
                              tightness: float = 0.3) -> mplPath:
        rng: RNG = cls.rng
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(f"Running cubic_spline_segment.")

        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)

        logger.debug(
            f"Segment parameters:\n"
            f"  start: {start}\n"
            f"    end: {end}\n"
            f"     x0: {x0}\n"
            f"     y0: {y0}\n"
            f"     x1: {x1}\n"
            f"     y1: {y1}\n"
            f"     dx: {dx}\n"
            f"     dy: {dy}\n"
            f" length: {length}\n"
        )

        if length < 1e-6:
            raise ValueError(f"Points are too close.")
    
        tx = dx / length
        ty = dy / length
        nx = -ty
        ny = tx
    
        o1 = rng.uniform(-1, 1) * amp * length
        o2 = rng.uniform(-1, 1) * amp * length
        a1 = tightness * length
        a2 = tightness * length
    
        p0 = (x0, y0)
        p1 = (x0 + tx * a1 + nx * o1, y0 + ty * a1 + ny * o1)
        p2 = (x1 - tx * a2 + nx * o2, y1 - ty * a2 + ny * o2)
        p3 = (x1, y1)
    
        verts = [p0, p1, p2, p3]
        codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
        return mplPath(verts, codes)

# =============================================================================
# INITIAL CODE
# =============================================================================




def main() -> None:
    if not logging.getLogger("worker").handlers:
        config = WorkerConfig()
        log_path = configure_logging(
            level=config.logger_level,
            name=LOGGER_NAME,
            run_prefix=f"{__name__}_{os.getpid()}"
        )
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"Logging initialized.")

    fig, ax = plt.subplots(figsize=(7, 2.5))
    dummy: Patch = Patch(ax)
    path = Patch.cubic_spline_ex(
        (0, 0), (5, 0), n_segments=7, amp=0.15, tightness=0.35
    )
    dummy.add_patch(path, facecolor="none", lw=3.0, edgecolor="royalblue", capstyle="round")
    
    
    #patch = mpl_patches.PathPatch(
    #    path, facecolor="none", lw=3.0, edgecolor="royalblue", capstyle="round"
    #)
    #ax.add_patch(patch)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.6, 0.6)
    plt.show()


# =============================================================================
# =============================================================================
if __name__ == "__main__":
    main()
