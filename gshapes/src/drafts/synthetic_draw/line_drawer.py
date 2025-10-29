"""
line_drawer.py
--------------

Generates and draws stylized or 'hand-drawn' lines on a Matplotlib axis.
Supports random or user-specified line styles, colors, and orientations.
"""

import os
import sys
import random
import math
import contextlib
from typing import Any, Dict, Sequence, Union, Tuple, Optional, List

import numpy as np
import matplotlib as mpl

# Use a non-interactive backend (safe for multiprocessing workers)
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors
from matplotlib._enums import JoinStyle, CapStyle


# -----------------------------------------------------------------------------
# Import thread-safe RNG utilities
# -----------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rng import get_rng, set_global_seed
else:
    from .rng import get_rng, set_global_seed


# -----------------------------------------------------------------------------
# RNG access
# -----------------------------------------------------------------------------
# For most cases, use thread-local RNGs (fast, no locks).
# Multiprocessing workers are already PID-isolated.
#
def _rng() -> "RNG":
    return get_rng(thread_safe=True)


def set_rng_seed(seed: int) -> None:
    """Set deterministic seed for global RNG (for reproducible plots)."""
    set_global_seed(seed)
    np.random.seed(seed)


# =============================================================================
# Public API
# =============================================================================
MAX_ANGLE_JITTER = 5
MAX_DASH_JITTER = 0.05
MAX_PATTERN_LENGTH = 20
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)


def draw_line(ax: Axes,
              linewidth: Optional[float] = None,
              pattern: Optional[str] = None,
              color: Optional[str] = None,
              alpha: Optional[float] = None,
              orientation: Union[str, int, None] = None,
              hand_drawn: Optional[bool] = None
             ) -> str:
    """Draw a single line on a given Matplotlib axis.

    Args:
        ax: Target Matplotlib Axes object.
        linewidth: Optional explicit line width.
        pattern: Pattern string ("--__-.", etc.) or named style.
        color: Optional CSS4 color name or RGB tuple.
        alpha: Opacity in [0, 1].
        orientation: "horizontal", "vertical", "diagonal_primary", or "diagonal_auxiliary".
        hand_drawn: If True, use XKCD mode and randomized jitter.
        seed: Optional RNG seed for deterministic behavior.
    
    Returns:
        TODO: Metadata
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"Unsupported ax type: {type(ax).__name__}")
    
    # Determine hand-drawn style
    if hand_drawn is None:
        hand_drawn = _rng.choice([False, True])
    elif not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    # Select color
    color_tuple = _get_color(color)

    # Normalize alpha
    if alpha is None:
        alpha = _rng.uniform(0.0, 1.0)
    elif isinstance(alpha, (int, float)):
        alpha = max(0.0, min(alpha, 1.0))
    else:
        raise TypeError(f"Unsupported alpha type: {type(alpha).__name__}")

    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x, y = _get_coords(
        float(x_min), float(y_min), float(x_max), float(y_max), orientation, hand_drawn
    )

    with plt.xkcd() if hand_drawn else contextlib.nullcontext():
        ax.plot(
            x,
            y,
            color=color_tuple,
            alpha=alpha,
            linewidth=linewidth or _rng.choice(DEFAULT_LINEWIDTHS),
            linestyle=_get_linestyle(pattern, hand_drawn),
            solid_capstyle=_rng.choice(list(CapStyle)),
            solid_joinstyle=_rng.choice(list(JoinStyle)),
            dash_capstyle=_rng.choice(list(CapStyle)),
            dash_joinstyle=_rng.choice(list(JoinStyle)),
        )


# =============================================================================
# Private helpers
# =============================================================================
def _get_linestyle(pattern: Optional[str] = None, hand_drawn: Optional[bool] = True
                  ) -> Union[str, Tuple[int, Tuple[float, ...]]]:
    """Convert textual or tuple pattern into a Matplotlib dash tuple."""
    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    # Case 1: no pattern provided - generate random dash pattern
    if pattern is None or (isinstance(pattern, str) and not pattern.strip()):
        if not hand_drawn:
            pattern = tuple(_rng.randint(1, 5) for _ in range(2 * _rng.randint(1, 5)))
            return (0, pattern)

        base_len: float = float(_rng.randint(1, 5))
        pattern = tuple(val for _ in range(_rng.randint(10, MAX_PATTERN_LENGTH)) for val in (
                max(0.5, base_len * (1 + max(-6, min(round(_rng.normal(0.0, 2.0)), 6)) / 6 * MAX_DASH_JITTER)),
                max(0.5, base_len * (1 + max(-6, min(round(_rng.normal(0.0, 2.0)), 6)) / 6 * MAX_DASH_JITTER) * 0.5),
            )
        )        
        return (0, pattern)

    if not isinstance(pattern, str):
        raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")

    # Case 2: named styles
    pattern_lower = pattern.lower().strip()
    named_styles = {"solid", "-", "dotted", ":", "dashed", "--", "dashdot", "-."}
    if pattern_lower in named_styles:
        return pattern_lower

    # Case 3: symbolic ("--__-." etc.)
    return _pattern_to_linestyle(pattern)


def _pattern_to_linestyle(pattern: str) -> Tuple[int, Tuple[float, ...]]:
    """Convert a symbolic pattern string into a Matplotlib-compatible linestyle."""
    if not isinstance(pattern, str):
        raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")

    mapping: dict[str, Tuple[str, int]] = {
        " ": ("off", 1),
        "_": ("off", 4),
        "-": ("on", 4),
        ".": ("on", 1),
    }

    pattern = pattern.lstrip()
    if not pattern:
        # Empty - random fallback
        return (0, tuple(_rng.randint(1, 5) for _ in range(2 * _rng.randint(1, 5))))

    segments: List[int] = []
    last_type: Optional[str] = None
    last_length: int = 0

    for ch in pattern:
        if ch not in mapping:
            raise ValueError(f'Invalid character "{ch!r}" in pattern "{pattern}"')
        seg_type, seg_len = mapping[ch]
        if seg_type == last_type and segments:
            segments[-1] += seg_len
        else:
            segments.append(seg_len)
            last_type = seg_type
        last_length = seg_len

    if len(segments) % 2 != 0:
        segments.append(last_length)

    return (0, tuple(segments))


def _get_color(color: Optional[Any]) -> Union[str, Tuple[float, float, float]]:
    """Select or generate a line color."""
    if color is None or not str(color).strip():
        return _rng.choice(list(colors.CSS4_COLORS.keys()))

    if isinstance(color, tuple):
        return color

    if not isinstance(color, str):
        raise TypeError(f"Unsupported color type: {type(color).__name__}")

    color = color.strip().lower()
    if color in colors.CSS4_COLORS.keys():
        return color

    # Accept plural CSS4 color names (simply extra "s" suffix),
    # e.g. "blues" to use randomly scaled base color (without "s")
    base_color = colors.CSS4_COLORS.get(color[:-1])
    if base_color is None:
        raise ValueError(f"Invalid color value: {color}")

    rgb = mpl.colors.to_rgb(base_color)
    scale_max = max(rgb)
    scale = _rng.uniform(0.1, 1 / scale_max if scale_max > 0 else 1)
    return tuple(np.clip(component * scale, 0, 1) for component in rgb)


def _get_coords(xmin: float, ymin: float, xmax: float, ymax: float,
                orientation: Union[str, int, None], hand_drawn: Optional[bool] = True,
               ) -> Tuple[List[float], List[float]]:
    """Generate (x, y) coordinates for a line based on orientation."""
    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    if orientation is None:
        return (
            [_rng.uniform(xmin, xmax), _rng.uniform(xmin, xmax)],
            [_rng.uniform(ymin, ymax), _rng.uniform(ymin, ymax)],
        )

    if isinstance(orientation, (int, float)):
         angle_deg = ((orientation + 90) % 180) - 90
    elif isinstance(orientation, str):
        orientation = orientation.strip().lower()
        if orientation == "horizontal":
            angle_deg = 0
        elif orientation == "vertical":
            angle_deg = 90
        elif orientation == "diagonal_primary":
            angle_deg = 45
        elif orientation == "diagonal_auxiliary":
            angle_deg = -45
        else:
            raise ValueError(f"Invalid orientation value: {orientation}")
    else:
        raise TypeError(f"Unsupported orientation type: {type(orientation).__name__}")

    angle_delta = 0
    if hand_drawn:
        angle_delta = int(round(_rng.normal(0.0, 2.0)))
        angle_delta = max(-MAX_ANGLE_JITTER, min(angle_delta, MAX_ANGLE_JITTER))

    angle_deg += angle_delta
    if angle_deg > 90:
        angle_deg -= 180

    x1 = _rng.uniform(xmin, 0.75 * xmax)
    y1 = _rng.uniform(ymin, 0.75 * ymax)

    if abs(angle_deg) == 90:
        x2 = x1
        y2 = _rng.uniform(y1 + 1, ymax)
        return [x1, x2], [y1, y2]

    slope: float = math.tan(math.radians(angle_deg))

    if angle_deg == 0 or abs(slope) < 1e-6:
        x2 = _rng.uniform(x1 + 1, xmax)
        y2 = y1
    else:
        xmax_adjusted: float = min(xmax, x1 + (ymax - y1) / slope)
        x2 = min(xmax, _rng.uniform(x1 + 1, xmax_adjusted))
        y2 = min(ymax, y1 + slope * (x2 - x1))

    x2 = np.clip(x2, xmin, xmax)
    y2 = np.clip(y2, ymin, ymax)

    return [x1, x2], [y1, y2]
