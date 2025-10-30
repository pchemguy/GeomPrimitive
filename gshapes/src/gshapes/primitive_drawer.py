"""
primitive_drawer.py
-------------------

Provides geometry and style primitives for 'hand-drawn' line-like objects.
All randomness is thread- and process-safe via rng.py.

Includes:
- make_line_meta(): Generate full metadata for a line primitive.
- draw_line(): Render a line using Matplotlib based on generated metadata.
"""

import os
import sys
import math
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


# -----------------------------------------------------------------------------
# Import thread-safe RNG utilities
# -----------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rng import RNG, get_rng
else:
    from .rng import RNG, get_rng


# -----------------------------------------------------------------------------
# RNG access
# -----------------------------------------------------------------------------
# Each thread/process gets its own RNG instance.
def _rng() -> RNG:
    """Return a thread-local RNG instance."""
    return get_rng(thread_safe=True)


# =============================================================================
# Constants
# =============================================================================
MAX_ANGLE_JITTER = 5
MAX_DASH_JITTER = 0.05
MAX_PATTERN_LENGTH = 20
DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
CSS4_COLOR_NAMES = list(colors.CSS4_COLORS.keys())


# =============================================================================
# Public API
# =============================================================================
def make_line_meta(
    ax: Axes,
    linewidth: Optional[float] = None,
    pattern: Optional[str] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    orientation: Union[str, int, None] = None,
    hand_drawn: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate metadata describing a stylized or 'hand-drawn' line.

    Args:
        ax: Target Matplotlib Axes object (used for coordinate limits).
        linewidth: Explicit line width or None for random.
        pattern: Pattern string ("--__-.", etc.) or named style.
        color: CSS4 color name or RGB tuple; None for random.
        alpha: Opacity in [0, 1]; None for random.
        orientation: 'horizontal', 'vertical', 'diagonal_primary', or angle in degrees.
        hand_drawn: If True, apply jittered parameters.

    Returns:
        dict: Metadata for line rendering (geometry, color, style, caps, joins).
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"Unsupported ax type: {type(ax).__name__}")

    rnd = _rng()

    # Determine hand-drawn style
    if hand_drawn is None:
        hand_drawn = rnd.choice([False, True])
    elif not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    # Select color
    color_tuple = _get_color(color)

    # Normalize alpha
    if alpha is None:
        alpha_value = rnd.uniform(0.0, 1.0)
    elif isinstance(alpha, (int, float)):
        alpha_value = max(0.0, min(alpha, 1.0))
    else:
        raise TypeError(f"Unsupported alpha type: {type(alpha).__name__}")

    # Compute coordinates
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x, y = _get_coords(x_min, y_min, x_max, y_max, orientation, hand_drawn)

    # Compose full metadata dictionary
    return {
        "x": x,
        "y": y,
        "linewidth": linewidth or rnd.choice(DEFAULT_LINEWIDTHS),
        "linestyle": _get_linestyle(pattern, hand_drawn),
        "color": color_tuple,
        "alpha": alpha_value,
        "orientation": orientation,
        "hand_drawn": hand_drawn,
        "solid_capstyle": rnd.choice(list(CapStyle)),
        "solid_joinstyle": rnd.choice(list(JoinStyle)),
        "dash_capstyle": rnd.choice(list(CapStyle)),
        "dash_joinstyle": rnd.choice(list(JoinStyle)),
    }


def draw_line(ax: Axes, **kwargs) -> Dict[str, Any]:
    """Draw a line using metadata from make_line_meta()."""
    meta = make_line_meta(ax, **kwargs)

    with plt.xkcd() if meta["hand_drawn"] else contextlib.nullcontext():
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

    return meta


# =============================================================================
# Private helpers
# =============================================================================
def _get_linestyle(
    pattern: Optional[str] = None, hand_drawn: Optional[bool] = True
) -> Union[str, Tuple[int, Tuple[float, ...]]]:
    """Convert textual or tuple pattern into a Matplotlib dash tuple."""
    rnd = _rng()

    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    # Case 1: no pattern provided - generate random dash pattern
    if pattern is None or (isinstance(pattern, str) and not pattern.strip()):
        if not hand_drawn:
            pattern = tuple(rnd.randint(1, 5) for _ in range(2 * rnd.randint(1, 5)))
            return (0, pattern)

        base_len: float = float(rnd.randint(1, 5))
        pattern = tuple(
            val
            for _ in range(rnd.randint(10, MAX_PATTERN_LENGTH))
            for val in (
                max(0.5, base_len * (1 + max(-6, min(round(rnd.normal(0.0, 2.0)), 6)) / 6 * MAX_DASH_JITTER)),
                max(0.5, base_len * (1 + max(-6, min(round(rnd.normal(0.0, 2.0)), 6)) / 6 * MAX_DASH_JITTER) * 0.5),
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
    rnd = _rng()

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
        return (0, tuple(rnd.randint(1, 5) for _ in range(2 * rnd.randint(1, 5))))

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
    rnd = _rng()

    if color is None or not str(color).strip():
        return rnd.choice(CSS4_COLOR_NAMES)

    if isinstance(color, tuple):
        return color

    if not isinstance(color, str):
        raise TypeError(f"Unsupported color type: {type(color).__name__}")

    color = color.strip().lower()
    if color in CSS4_COLOR_NAMES:
        return color

    # Accept plural CSS4 color names (extra 's' suffix), e.g., "blues"
    base_color = colors.CSS4_COLORS.get(color[:-1])
    if base_color is None:
        raise ValueError(f"Invalid color value: {color}")

    rgb = mpl.colors.to_rgb(base_color)
    scale_max = max(rgb)
    scale = rnd.uniform(0.1, 1 / scale_max if scale_max > 0 else 1)
    return tuple(np.clip(component * scale, 0, 1) for component in rgb)


def _get_coords(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    orientation: Union[str, int, None],
    hand_drawn: Optional[bool] = True,
) -> Tuple[List[float], List[float]]:
    """Generate (x, y) coordinates for a line based on orientation."""
    rnd = _rng()

    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    if orientation is None:
        return (
            [rnd.uniform(xmin, xmax), rnd.uniform(xmin, xmax)],
            [rnd.uniform(ymin, ymax), rnd.uniform(ymin, ymax)],
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

    # Hand-drawn jitter
    angle_delta = 0
    if hand_drawn:
        angle_delta = int(round(rnd.normal(0.0, 2.0)))
        angle_delta = max(-MAX_ANGLE_JITTER, min(angle_delta, MAX_ANGLE_JITTER))

    angle_deg += angle_delta
    if angle_deg > 90:
        angle_deg -= 180

    x1 = rnd.uniform(xmin, 0.75 * xmax)
    y1 = rnd.uniform(ymin, 0.75 * ymax)

    if abs(angle_deg) == 90:
        x2 = x1
        y2 = rnd.uniform(y1 + 1, ymax)
        return [x1, x2], [y1, y2]

    slope: float = math.tan(math.radians(angle_deg))

    if angle_deg == 0 or abs(slope) < 1e-6:
        x2 = rnd.uniform(x1 + 1, xmax)
        y2 = y1
    else:
        xmax_adjusted: float = min(xmax, x1 + (ymax - y1) / slope)
        x2 = min(xmax, rnd.uniform(x1 + 1, xmax_adjusted))
        y2 = min(ymax, y1 + slope * (x2 - x1))

    x2 = np.clip(x2, xmin, xmax)
    y2 = np.clip(y2, ymin, ymax)

    return [x1, x2], [y1, y2]
