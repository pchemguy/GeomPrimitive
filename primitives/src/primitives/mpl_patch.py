"""
mpl_patch.py
-------

https://chatgpt.com/c/6905add2-1ff8-8328-ba21-6c370614dcc4
https://chatgpt.com/c/69025b9d-b044-8333-8159-aac740d7bf70
----------------------------------------------------------

Geometry Specification:
    Segment = {
        x1: Union[float, int, None],
        y1: Union[float, int, None],
        x2: Union[float, int, None],
        y2: Union[float, int, None],
        orientation: Union[int, str, None]
    }

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

PointXY = Tuple[Union[int, float], Union[int, float]]

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
        self.logger = logging.getLogger(LOGGER_NAME)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------
    # APIs
    # -------------------------------------------------------------------------
    def make_geometry(self,
                      ax: Optional[Axes] = None,
                      linewidth: Optional[float] = None,
                      pattern: Optional[str] = None,
                      color: Optional[str] = None,
                      alpha: Optional[float] = None,
                      capstyle: Optional[str] = None,
                      joinstyle: Optional[str] = None,
                      geometry: Dict = None,
                     ) -> "Patch":
        pass

    def draw(self) -> None:
        if not isinstance(self._ax, Axes):
            raise TypeError(f"ax is not set")
        for patch in self.patches.values():
            self._ax.add_patch(patch)
    
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
                 ) -> PathPatch:
        rng: RNG = self.__class__.rng
        if not isinstance(self._ax, Axes):
            raise ValueError(f"ax is not set")

        ax: Axes = self._ax

        # Color and alpha
        fill: bool = style_options.get("fill", False)
        if not fill:
            style_options["fill"] = False
            edge_color = color or style_options.get("edgecolor") or style_options.get("ec")
            for attr in ("facecolor", "fc", "edgecolor", "ec"):
                if attr in style_options:
                     style_options.pop(attr)
            style_options["color"] = self._get_color(edge_color)
        else:
            style_options["facecolor"] = self._get_color(color or style_options.get("facecolor"))
            if not ("edgecolor" in style_options or "ec" in style_options):
                style_options["edgecolor"] = self._get_color(color)

        if alpha is None or not isinstance(alpha, (int, float)):
            alpha = 1.5 - rng.paretovariate(1.0) * 0.5
        style_options["alpha"] = max(0.1, min(float(alpha), 1.0))

        # Cap and join styles
        style_options["capstyle"] = CapStyle._member_map_.get(str(capstyle).lower()) or rng.choice(list(CapStyle))
        style_options["joinstyle"] = JoinStyle._member_map_.get(str(joinstyle).lower()) or rng.choice(list(JoinStyle))

        style_options["linewidth"] = linewidth or style_options.get("lw") or rng.choice(DEFAULT_LINEWIDTHS)
        if style_options.get("lw"):
            style_options.pop("lw")
        style_options["linestyle"] = self._get_linestyle(pattern, hand_drawn=True)

        self.logger.debug(f"add_patch() - style_options:\n{style_options}")
        
        patch: PathPatch = PathPatch(path, **style_options)
        self.patches[str(patch)] = patch

        return patch

    @classmethod
    def cubic_spline_ex(cls, start: PointXY, end: PointXY, n_segments: int = 5,
                        amp: float = 0.15, tightness: float = 0.3) -> mplPath:
        rng: RNG = cls.rng
        x0, y0 = start
        xn, yn = end
        dx, dy = xn - x0, yn - y0
        stepx, stepy = dx / n_segments, dy / n_segments
    
        JITTER_FACTOR = 0.4
        points = [(x0, y0)]
        for i in range(1, n_segments):
            jitter = rng.uniform(-1, 1) * JITTER_FACTOR
            points.append((x0 + (i + jitter) * stepx, y0 + (i + jitter) * stepy))
        points.append((xn, yn))
    
        verts: List[PointXY] = []
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
    def _cubic_spline_segment(cls, start: PointXY, end: PointXY,
                              amp: float = 0.15, tightness: float = 0.3) -> mplPath:
        rng: RNG = cls.rng
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(f"Running cubic_spline_segment.")

        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0

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
        )
        
        dev1 = max(-1, min(rng.normal(0, 1) / 3, 1)) * amp
        dev2 = max(-1, min(rng.normal(0, 1) / 3, 1)) * amp
    
        p0 = (x0, y0)
        p1 = (x0 + dx * tightness - dy * dev1, y0 + dy * tightness + dx * dev1)
        p2 = (x1 - dx * tightness - dy * dev2, y1 - dy * tightness + dx * dev2)
        p3 = (x1, y1)

    
        verts = [p0, p1, p2, p3]
        codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
        return mplPath(verts, codes)

    @classmethod
    def _get_segment_coords(cls,
                    xboxmin: float,
                    yboxmin: float,
                    xboxmax: float,
                    yboxmax: float,
                    geometry: Optional[Dict] = {},
                   ) -> Tuple[List[float], List[float]]:
        rng: RNG = cls.rng
        if not isinstance(geometry, dict):
            raise TypeError(f"Unsupported geometry type: {type(geometry).__name__}")

        if not geometry or geometry.get("orientation") is None:
            return {
                "p1": (
                    geometry.get("x1") or rng.uniform(xboxmin, xboxmax),
                    geometry.get("y1") or rng.uniform(yboxmin, yboxmax),
                ),
                "p2": (
                    geometry.get("x2") or rng.uniform(xboxmin, xboxmax),
                    geometry.get("y2") or rng.uniform(yboxmin, yboxmax),
                ),
            }

        # Orientation
        angle = self._get_angle(geometry("orientation"), hand_drawn=True)

        x1 = geometry.get("x1") or rng.uniform(xboxmin, 0.75 * xboxmax),
        y1 = geometry.get("y1") or rng.uniform(yboxmin, 0.75 * yboxmax),
        
        if abs(90 - abs(angle)) < 0.1:
            x2, y2 = x1, rng.uniform(y1 * 0.8, yboxmax)
            return {"p1": (x1, y1), "p2": (x2, y2)}

        slope = math.tan(math.radians(angle))
        if angle == 0 or abs(slope) < 1e-6:
            x2, y2 = rng.uniform(x1 * 0.8, xboxmax), y1
        else:
            xmax_adj = min(xboxmax, x1 + (yboxmax - y1) / slope)
            x2 = min(xboxmax, rng.uniform(x1 + 1, xmax_adj))
            y2 = min(yboxmax, y1 + slope * (x2 - x1))

        return {
            "p1": (x1, y1),
            "p2": (np.clip(x2, xboxmin, xboxmax), np.clip(y2, yboxmin, yboxmax)),
        }


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
        (0, 0), (5, 0), n_segments=7, amp=0.15, tightness=0.3
    )
    dummy.add_patch(path, facecolor="none", lw=3.0, edgecolor="royalblue", capstyle="round", alpha=1)
    dummy.draw()
    
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
