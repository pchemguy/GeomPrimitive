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
from matplotlib import colors
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
MAX_ANGLE_JITTER = 5
MAX_DASH_JITTER = 0.05
MAX_PATTERN_LENGTH = 30
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
CSS4_COLOR_NAMES = list(colors.CSS4_COLORS.keys())


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

        # Coordinates
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x, y = self._get_segment_coords(x_min, y_min, x_max, y_max, orientation, hand_drawn)

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
    # Helper methods (static-style)
    # -------------------------------------------------------------------------
    @classmethod
    def _get_linestyle(cls, pattern: Optional[str], hand_drawn: bool
                      ) -> Union[str, Tuple[int, Tuple[float, ...]]]:
        rng: RNG = cls.rng
        if pattern is None or (isinstance(pattern, str) and not pattern.strip()):
            if not hand_drawn:
                pattern = tuple(rng.randint(1, 5) for _ in range(2 * rng.randint(1, 5)))
                return (0, pattern)

            base_len = float(rng.randint(1, 5))
            pattern = tuple(
                max(0.5, base_len * (1 + max(-6, min(round(rng.normal(0.0, 2.0)), 6)) / 6 * MAX_DASH_JITTER))
                for _ in range(rng.randint(6, MAX_PATTERN_LENGTH))
            )
            return (0, pattern)

        if not isinstance(pattern, str):
            raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")

        pattern_lower = pattern.lower().strip()
        named_styles = {"solid", "-", "dotted", ":", "dashed", "--", "dashdot", "-."}
        if pattern_lower in named_styles:
            return pattern_lower

        return cls._pattern_to_linestyle(pattern)

    @classmethod
    def _pattern_to_linestyle(cls, pattern: str) -> Tuple[int, Tuple[float, ...]]:
        rng: RNG = cls.rng
        mapping = {" ": ("off", 1), "_": ("off", 4), "-": ("on", 4), ".": ("on", 1)}
        pattern = pattern.lstrip()
        if not pattern:
            return (0, tuple(rng.randint(1, 5) for _ in range(2 * rng.randint(1, 5))))

        segments, last_type, last_len = [], None, 0
        for ch in pattern:
            if ch not in mapping:
                raise ValueError(f'Invalid character "{ch}" in pattern "{pattern}"')
            seg_type, seg_len = mapping[ch]
            if seg_type == last_type and segments:
                segments[-1] += seg_len
            else:
                segments.append(seg_len)
                last_type = seg_type
            last_len = seg_len
        if len(segments) % 2 != 0:
            segments.append(last_len)
        return (0, tuple(segments))

    @classmethod
    def _get_color(cls, color: Optional[Any]) -> Union[str, Tuple[float, float, float]]:
        rng: RNG = cls.rng
        if color is None or not str(color).strip():
            return rng.choice(CSS4_COLOR_NAMES)
        if isinstance(color, tuple):
            return color
        color = color.strip().lower()
        if color in CSS4_COLOR_NAMES:
            return color
        base = colors.CSS4_COLORS.get(color[:-1])
        if base is None:
            raise ValueError(f"Invalid color: {color}")
        rgb = mpl.colors.to_rgb(base)
        smax = max(rgb)
        scale = rng.uniform(0.1, 1 / smax if smax > 0 else 1)
        return tuple(np.clip(c * scale, 0, 1) for c in rgb)

    @classmethod
    def _get_segment_coords(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        orientation: Union[str, int, None],
        hand_drawn: bool
    ) -> Tuple[List[float], List[float]]:
        rng: RNG = cls.rng
        if orientation is None:
            return (
                [rng.uniform(xmin, xmax), rng.uniform(xmin, xmax)],
                [rng.uniform(ymin, ymax), rng.uniform(ymin, ymax)],
            )

        if isinstance(orientation, (int, float)):
            angle = ((orientation + 90) % 180) - 90
        elif isinstance(orientation, str):
            o = orientation.strip().lower()
            if o == "horizontal":
                angle = 0
            elif o == "vertical":
                angle = 90
            elif o == "diagonal_primary":
                angle = 45
            elif o == "diagonal_auxiliary":
                angle = -45
            else:
                raise ValueError(f"Invalid orientation: {orientation}")
        else:
            raise TypeError(f"Unsupported orientation type: {type(orientation).__name__}")

        if hand_drawn:
            delta = int(round(rng.normal(0.0, 2.0)))
            delta = max(-MAX_ANGLE_JITTER, min(delta, MAX_ANGLE_JITTER))
            angle += delta
            if angle > 90:
                angle -= 180

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
