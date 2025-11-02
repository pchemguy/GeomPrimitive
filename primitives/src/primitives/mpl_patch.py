"""
mpl_patch.py
-------

https://chatgpt.com/c/6905add2-1ff8-8328-ba21-6c370614dcc4
https://chatgpt.com/c/69025b9d-b044-8333-8159-aac740d7bf70
----------------------------------------------------------
"""
from __future__ import annotations

import os
import sys
import math
import logging
from enum import Enum
from types import NoneType
from typing import Any, Union, Optional, List

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

numeric = Union[int, float]
PointXY = tuple[numeric, numeric]
CoordRange = tuple[numeric, numeric]

# =============================================================================
# Constants
# =============================================================================
LOGGER_NAME = "worker"
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)


# =============================================================================
# Patch implementation
# =============================================================================
class Patch(Primitive):
    __slots__ = ("last_patch",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    # -------------------------------------------------------------------------
    # APIs
    # -------------------------------------------------------------------------
    def make_geometry(self,
                      ax: Optional[Axes] = None,
                      shape_path: mplPath = None,
                      linewidth: Optional[float] = None,
                      pattern: Optional[str] = None,
                      color: Optional[str] = None,
                      alpha: Optional[float] = None,
                      capstyle: Optional[str] = None,
                      joinstyle: Optional[str] = None,
                      **style_options
                     ) -> "Patch":
        rng: RNG = self.__class__.rng
        logger = logging.getLogger(LOGGER_NAME)
        logger.info(f"Running make_geometry().")

        if not ax is None:
            self.ax = ax
        if not isinstance(self.ax, Axes):
            raise TypeError(f"self.ax is not set.")

        logger.debug(f"self.ax.get_xlim(): {self.ax.get_xlim()}; self.ax.get_ylim(): {self.ax.get_ylim()}.")
        ax: Axes = self.ax

        if not isinstance(shape_path, mplPath):
            raise TypeError(f"Unsupported shape_path type: {type(shape_path).__name__}")
        if shape_path.vertices is None:
            raise ValueError(f"shape_path.vertices is empty or not set.")
        if shape_path.codes is None:
            raise ValueError(f"shape_path.codes is empty or not set.")

        # Color and alpha
        fill: bool = style_options.get("fill", False)
        if not fill:
            style_options["fill"] = False
            edge_color = (
                color or style_options.get("edgecolor") or style_options.get("ec")
            )
            for attr in ("facecolor", "fc", "edgecolor", "ec"):
                if attr in style_options: style_options.pop(attr)
            style_options["color"] = self._get_color(edge_color)
        else:
            style_options["facecolor"] = (
                self._get_color(color or style_options.get("facecolor"))
            )
            if not ("edgecolor" in style_options or "ec" in style_options):
                style_options["edgecolor"] = self._get_color(color)

        if alpha is None or not isinstance(alpha, (int, float)):
            alpha = 1.5 - rng.paretovariate(1.0) * 0.5
        style_options["alpha"] = max(0.1, min(float(alpha), 1.0))

        style_options["capstyle"] = (
            CapStyle._member_map_.get(str(capstyle).lower())
            or rng.choice(list(CapStyle))
        )
        style_options["joinstyle"] = (
            JoinStyle._member_map_.get(str(joinstyle).lower())
            or rng.choice(list(JoinStyle))
        )

        style_options["linewidth"] = (
            linewidth or style_options.get("lw") or rng.choice(DEFAULT_LINEWIDTHS)
        )
        if style_options.get("lw"): style_options.pop("lw")

        style_options["linestyle"] = self._get_linestyle(pattern, hand_drawn=True)

        self.logger.debug(f"make_geometry() - style_options:\n{style_options}")

        patch: PathPatch = PathPatch(shape_path, **style_options)
        self.last_patch = patch
        self.patches[id(patch)] = patch
        self._meta[id(patch)] = {
            "path": shape_path,
            "style": style_options,
        }

        return self

    def draw(self) -> None:
        if not isinstance(self.ax, Axes):
            raise TypeError(f"ax is not set")
        for patch in self.patches.values():
            self.ax.add_patch(patch)
    
    # -------------------------------------------------------------------------
    # MPL Path generators
    # -------------------------------------------------------------------------
    def make_path(self, shape_kind: str, **geometry_options) -> mplPath:
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(f"Running make_path.")
        xlim: CoordRange = self.ax.get_xlim()
        ylim: CoordRange = self.ax.get_ylim()
        if shape_kind == "segment":
            geometry: dict[str, PointXY] = self.__class__._get_line_coords(
                xlim, ylim, **geometry_options
            )
            return self.__class__.line_path(**geometry, **geometry_options)
        else:
            raise ValueError(f"Unsupported shape_kind: {shape_kind}")

    @classmethod
    def line_path(cls, start: PointXY, end: PointXY, spline_count: int = 5,
                  amp: float = 0.15, tightness: float = 0.3, **kwargs) -> mplPath:
        rng: RNG = cls.rng
        x0, y0 = start
        xn, yn = end
        dx, dy = xn - x0, yn - y0
        stepx, stepy = dx / spline_count, dy / spline_count
    
        JITTER_FACTOR = 0.4
        points = [(x0, y0)]
        for i in range(1, spline_count):
            jitter = rng.uniform(-1, 1) * JITTER_FACTOR
            points.append((x0 + (i + jitter) * stepx, y0 + (i + jitter) * stepy))
        points.append((xn, yn))
    
        verts: List[PointXY] = []
        codes: List[int] = []
        for i in range(len(points) - 1):
            segment_path: mplPath = cls._random_spline_cubic(
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
    def _random_spline_cubic(cls, start: PointXY, end: PointXY,
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
    def _get_line_coords(cls,
                         xlim: CoordRange,
                         ylim: CoordRange,
                         start: Optional[PointXY] = (None, None),
                         end: Optional[PointXY] = (None, None),
                         orientation: Union[str, int, None] = None,
                         **kwargs) -> dict[str, PointXY]:
        rng: RNG = cls.rng
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(f"Running _get_line_coords.")

        if orientation is None:
            return {
                "start": (
                    start[0] or rng.uniform(*xlim),
                    start[1] or rng.uniform(*ylim),
                ),
                "end": (
                    end[0] or rng.uniform(*xlim),
                    end[1] or rng.uniform(*ylim),
                ),
            }

        # Orientation
        angle = cls._get_angle(orientation, hand_drawn=True)

        xmin, xmax = xlim
        ymin, ymax = ylim

        x1: numeric = start[0] or rng.uniform(xmin, 0.75 * xmax)
        y1: numeric = start[1] or rng.uniform(ymin, 0.75 * ymax)
        
        if abs(90 - abs(angle)) < 0.1:
            x2, y2 = x1, rng.uniform(y1 * 0.8, ymax)
            return {"start": (x1, y1), "end": (x2, y2)}

        slope = math.tan(math.radians(angle))
        if angle == 0 or abs(slope) < 1e-6:
            x2, y2 = rng.uniform(x1 * 0.8, xmax), y1
        else:
            xmax_adj = min(xmax, x1 + (ymax if slope > 0 else ymin - y1) / slope)
            x2 = min(xmax, rng.uniform(x1 + 1, xmax_adj))
            y2 = min(ymax, y1 + slope * (x2 - x1))

        return {
            "start": (x1, y1),
            "end": (float(np.clip(x2, *xlim)), float(np.clip(y2, *ylim))),
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
    #dummy.logger = logger
    shape_path = Patch.line_path(
        (-10, 0), (50, 0), spline_count=7, amp=0.15, tightness=0.3
    )

    dummy.make_geometry(shape_path=shape_path, facecolor="none", lw=3.0, edgecolor="royalblue", capstyle="round", alpha=1)
    
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-10, 50)
    ax.set_ylim(-10, 10)

    shape_path = dummy.make_path(shape_kind="segment")
    dummy.make_geometry(ax=ax, shape_path=shape_path, color="red", lw=3.0, capstyle="round", alpha=1)
    
    dummy.draw()
    plt.show()


# =============================================================================
# =============================================================================
if __name__ == "__main__":
    main()
