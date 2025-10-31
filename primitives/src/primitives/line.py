"""
line.py
------------

Implements the Line class - a stylized or 'hand-drawn' line primitive.

Responsibilities:
  - Generate geometry and style metadata (via make_geometry)
  - Draw the line using Matplotlib
  - Support object reuse (reset() updates in-place)
  - Extendable as a base for circle/ellipse/arc primitives
"""

import os
import sys
import math
import logging
import contextlib
from typing import Any, Dict, Union, Tuple, Optional, List

import numpy as np
import matplotlib as mpl

# Use a non-interactive backend (safe for multiprocessing workers)
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib._enums import JoinStyle, CapStyle

# Import RNG utilities
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rng import RNG
    from base import Primitive
else:
    from .rng import RNG
    from .base import Primitive


# =============================================================================
# Constants
# =============================================================================
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)


# =============================================================================
# Primitive implementation
# =============================================================================
class Line(Primitive):
    """
    Stylized line primitive metadata and rendering class.

    Extends:
        Primitive

    Responsibilities:
      - Generate geometry and style metadata (`make_geometry`)
      - Draw line onto Matplotlib axes (`draw`)
    """

    __slots__ = ()

    # -------------------------------------------------------------------------
    # Geometry and style synthesis
    # -------------------------------------------------------------------------
    def make_geometry(
        self,
        ax: Optional[Axes] = None,
        linewidth: Optional[float] = None,
        pattern: Optional[str] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        orientation: Union[str, int, None] = None,
        capstyle: Optional[str] = None,
        joinstyle: Optional[str] = None,
        hand_drawn: Optional[bool] = None,
    ) -> "Line":
        """
        Generate metadata describing a stylized or 'hand-drawn' line.
        
        This method produces a structured dictionary that fully defines the
        geometry and stylistic parameters of a line segment. It does not draw
        anything-only returns a data description suitable for later rendering.
        
        Args:
            ax (matplotlib.axes.Axes):
                Target Matplotlib Axes object. Used only to extract `xlim` and `ylim`
                for coordinate generation.
            linewidth (float, optional):
                Explicit line width in points. If None, chosen randomly from
                `DEFAULT_LINEWIDTHS`.
            pattern (str, optional):
                - Named style: "solid", "dotted", "dashed", "dashdot",
                  or equivalent shorthand ("-", "--", ":", "-.").
                - Symbolic pattern string: "--__-.", "_.- ", etc.
                  where:
                    - space " " -- off length 1
                    - underscore "_" -- off length 4
                    - dash "-" -- on length 4
                    - dot "." -- on length 1
                  Example: "--__-." -- `(8, 8, 5, 2)`
                - If None, a randomized dash pattern is generated.
            color (str | tuple[float, float, float], optional):
                - CSS4/X11 color name (case-insensitive), e.g. "skyblue", "tomato".
                - Tuple of floats `(r, g, b)` with values in [0, 1].
                - Plural color names (e.g., "blues") are accepted; they are
                  interpreted as random brightness variants of the base name.
                - If None, a random CSS4 color is chosen.
            alpha (float, optional):
                Opacity in [0, 1]. Randomized if None.
            orientation (str | int | None):
                - Named directions: "horizontal", "vertical",
                  "diagonal_primary" (45o), "diagonal_auxiliary" (-45o).
                - Numeric value: explicit angle in degrees.
                - None: random endpoints within current axis limits.
            hand_drawn (bool, optional):
                If True, applies XKCD-style randomness and Gaussian jitter to dash
                lengths and orientation.
        
        Returns:
            dict[str, Any]: Metadata for line rendering, with all parameters resolved.
        
            The dictionary fields are as ax.plot() arguments:
                x, y, linewidth, linestyle, color, alpha, orientation,
                solid_capstyle, solid_joinstyle, dash_capstyle, dash_joinstyle.
            Additionally, hand_drawn field is defined, which controls application
            of XKCD style.
        """
        rng: RNG = self.__class__.rng
        if isinstance(ax, Axes):
            self._ax = ax

        if not isinstance(self._ax, Axes):
            raise TypeError(f"Unsupported ax type: {type(ax).__name__}")

        ax = self._ax

        # Determine hand-drawn style
        if hand_drawn is None:
            hand_drawn = rng.choice([False, True])
        elif not isinstance(hand_drawn, bool):
            raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

        # Color and alpha
        color_tuple = self._get_color(color)
        if alpha is None or not isinstance(alpha, (int, float)):
            alpha = 1.5 - rng.paretovariate(1.0) * 0.5
        alpha_value = max(0.0, min(float(alpha), 1.0))

        # Cap and join styles
        capstyle = CapStyle._member_map_.get(str(capstyle).lower()) or rng.choice(list(CapStyle))
        joinstyle = JoinStyle._member_map_.get(str(joinstyle).lower()) or rng.choice(list(JoinStyle))

        # Orientation
        angle = self._get_angle(orientation, hand_drawn)

        # Coordinates
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x, y = self._get_segment_coords(x_min, y_min, x_max, y_max, angle, hand_drawn)

        # Compose full metadata dict
        self._meta: Dict[str, Any] = {
            "x": x,
            "y": y,
            "linewidth": linewidth or rng.choice(DEFAULT_LINEWIDTHS),
            "linestyle": self._get_linestyle(pattern, hand_drawn),
            "color": color_tuple,
            "alpha": alpha_value,
            "orientation": orientation,
            "hand_drawn": hand_drawn,
            "solid_capstyle": capstyle,
            "solid_joinstyle": joinstyle,
            "dash_capstyle": capstyle,
            "dash_joinstyle": joinstyle,
        }
        return self

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------
    def draw(self) -> None:
        """
        Render this line onto the given Matplotlib axis.
        """
        if not isinstance(self._ax, Axes):
            raise TypeError(f"ax is not set.")

        ax = self._ax

        meta = self.meta
        #self.logger.debug(f"Class {self.__class__}.draw() - meta: {meta}")
        with plt.xkcd() if meta.get("hand_drawn") else contextlib.nullcontext():
            ax.plot(
                meta["x"],
                meta["y"],
                color=meta["color"],
                alpha=meta["alpha"],
                linewidth=meta["linewidth"],
                linestyle=meta["linestyle"],
                solid_capstyle=meta["solid_capstyle"],
                solid_joinstyle=meta["solid_joinstyle"],
                dash_capstyle=meta["dash_capstyle"],
                dash_joinstyle=meta["dash_joinstyle"],
            )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    @classmethod
    def _get_segment_coords(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        angle: Union[float, int, None],
        hand_drawn: bool
    ) -> Tuple[List[float], List[float]]:
        rng: RNG = cls.rng
        if angle is None:
            return (
                [rng.uniform(xmin, xmax), rng.uniform(xmin, xmax)],
                [rng.uniform(ymin, ymax), rng.uniform(ymin, ymax)],
            )

        x1, y1 = rng.uniform(xmin, 0.75 * xmax), rng.uniform(ymin, 0.75 * ymax)
        if abs(angle) == 90:
            x2, y2 = x1, rng.uniform(y1 + 1, ymax)
            return [x1, x2], [y1, y2]

        slope = math.tan(math.radians(angle))
        if angle == 0 or abs(slope) < 1e-6:
            x2, y2 = rng.uniform(x1 + 1, xmax), y1
        else:
            xmax_adj = min(xmax, x1 + (ymax - y1) / slope)
            x2 = min(xmax, rng.uniform(x1 + 1, xmax_adj))
            y2 = min(ymax, y1 + slope * (x2 - x1))

        return [float(np.clip(x1, xmin, xmax)), float(np.clip(x2, xmin, xmax))], [
            float(np.clip(y1, ymin, ymax)),
            float(np.clip(y2, ymin, ymax)),
        ]
