"""
line_drawer.py
--------------

Generates and draws stylized or 'hand-drawn' lines on a Matplotlib axis.
Supports random or user-specified line styles, colors, and orientations.
"""

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


# =============================================================================
# Public API
# =============================================================================
def draw_line(ax: Axes,
              linewidth: Optional[float] = None,
              pattern: Optional[str] = None,
              color: Optional[str] = None,
              alpha: Optional[float] = None,
              orientation: Optional[Union[str, int]] = None,
              hand_drawn: Optional[bool] = None,
              seed: Optional[int] = None
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
        The created Matplotlib Line2D object.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine hand-drawn style
    if hand_drawn is None:
        hand_drawn = hand_drawn or random.choice([False, True])
    elif not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    color = _get_color(color)

    if alpha is None:
        alpha = random.uniform(0.0, 1.0)
    elif isinstance(alpha, (int, float)):
        alpha = float(max(0.0, min(alpha, 1.0)))
    else:
        raise TypeError(f"Unsupported alpha type: {type(alpha).__name__}")

    _axis_limits: Tuple[float, float] = ax.get_xlim()
    x_min: int = math.floor(_axis_limits[0])
    x_max: int = math.ceil(_axis_limits[1])
    _axis_limits: Tuple[float, float] = ax.get_ylim()
    y_min: int = math.floor(_axis_limits[0])
    y_max: int = math.ceil(_axis_limits[1])

    x, y = _get_coords(float(x_min), float(y_min),
                       float(x_max), float(y_max), orientation, hand_drawn)

    with plt.xkcd() if hand_drawn else contextlib.nullcontext():
        line, = ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth or random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),
            linestyle=_get_linestyle(pattern, hand_drawn),
            solid_capstyle=random.choice(list(CapStyle)),
            solid_joinstyle=random.choice(list(JoinStyle))
        )


# =============================================================================
# Private helpers
# =============================================================================
def _get_linestyle(pattern: Optional[str] = None,
                   hand_drawn: Optional[bool] = True
                  ) -> str | Tuple[int | float, ...]:
    """Convert textual or tuple pattern into a Matplotlib dash tuple.

    Returns:
            A tuple (on_off_sequence) or String for named styles.
    """
    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")

    if pattern is None or (isinstance(pattern, str) and not pattern.strip()):
        if not hand_drawn:
            pattern = tuple(random.randint(1, 5) for _ in range(2 * random.randint(1, 5)))
        else:
            base_len: float = float(random.randint(1, 5))
            pattern = tuple(
                base_len * (1 + max(-6, min(round(random.normalvariate(mu=0.0, sigma=2.0)), 6)) / 6 * 0.05),
                base_len * (1 + max(-6, min(round(random.normalvariate(mu=0.0, sigma=2.0)), 6)) / 6 * 0.05) * 0.5
                for _ in range(random.randint(10, 20))
            )

        return (0, pattern)

    if not isinstance(pattern, str):
        raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")
    
    # pattern is str below this line
    
    pattern_lower = pattern.lower()
    named_styles = {"solid", "-", "dotted", ":", "dashed", "--", "dashdot", "-."}

    
    if pattern_lower.strip() in named_styles:
        return pattern_lower.strip()  # Case 1: recognized named pattern
    else:
        return _pattern_to_linestyle(pattern) # Case 2: symbolic pattern string ("--__." etc.)


def _pattern_to_linestyle(pattern: str) -> Tuple[int, Tuple[int, ...]]:
    """
    Convert a symbolic pattern string into a Matplotlib-compatible linestyle.

    Rules:
        - ' ' : off (gap) of 1 unit
        - '_' : off (gap) of 4 units
        - '-' : on  (dash) of 4 units
        - '.' : on  (dash) of 1 unit
        - Leading spaces are ignored (no offset)
        - If the pattern does not end with an off-segment,
          an additional off-segment equal to the last dash length is appended.

    Random pattern generation:
        - Pattern length = 2N, where N in [1, 5]
        - Each on/off length in [1, 5]

    Args:
        pattern (str): Pattern string containing ' ', '_', '-', and '.' characters.

    Returns:
        Tuple[int, Tuple[int, ...]]: A Matplotlib linestyle tuple `(offset, pattern)`.

    Example:
        >>> pattern_to_linestyle("--__-.  ")
        (0, (8, 8, 5, 2))
        >>> pattern_to_linestyle("_--__-.")
        (0, (8, 8, 5, 5))
        >>> pattern_to_linestyle("")
        (0, (3, 5, 2, 4))
    """
    if not isinstance(pattern, str):
        raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")
    
    # Character mapping: defines whether each symbol is "on" or "off" and its length
    mapping: dict[str, Tuple[str, int]] = {
        " ": ("off", 1),
        "_": ("off", 4),
        "-": ("on", 4),
        ".": ("on", 1),
    }

    # Remove leading spaces, as they do not contribute to the offset
    pattern = pattern.lstrip(" ")
    if not pattern:
        raise ValueError("Pattern is empty after trimming leading spaces.")

    # Sequentially build the pattern by merging consecutive segments of the same type
    segments: List[int] = []
    last_type: str | None = None
    last_length: int = 0

    for ch in pattern:
        if ch not in mapping:
            raise ValueError(f'Invalid character "{ch!r}" in pattern "{pattern}"')
        seg_type, seg_len = mapping[ch]

        # If same type as previous, extend the segment
        if seg_type == last_type and segments:
            segments[-1] += seg_len
        else:
            segments.append(seg_len)
            last_type = seg_type

        last_length = seg_len  # Track the last segment length for trailing adjustment

    # Ensure the pattern alternates between on/off - must have an even number of entries
    if len(segments) % 2 != 0:
        # Append a final "off" segment equal to the last dash length
        segments.append(last_length)

    return (0, tuple(segments))


def _get_color(color: Any | None) -> Union[str, Tuple]:
    """Select a line color."""
    if color is None or not str(color).strip():
        return random.choice(list(colors.CSS4_COLORS.keys()))
    if isinstance(color, tuple):
        return color
    if not isinstance(color, str):
        raise TypeError(f"Unsupported color type: {type(color).__name__}")

    # color is str only below this line.

    _color: Union[Tuple, str, None] = colors.CSS4_COLORS.get(color.strip().lower()[:-1])
    if _color is None:
        raise ValueError(f"Invalid color value: {color}.")
    _color = mpl.colors.to_rgb(_color)
    scale_max: float = max(_color)
    scale_max = 1 / scale_max if scale_max > 0 else 1
    scale = random.uniform(0.1, scale_max)
    _color = tuple(component * scale for component in _color)
    return _color


def _get_coords(xmin: float, ymin: float,
                xmax: float, ymax: float,
                orientation: Union[str, int],
                hand_drawn: Optional[bool] = True
               ) -> Tuple[List[float], List[float]]:
    if not isinstance(hand_drawn, bool):
        raise TypeError(f"Unsupported hand_drawn type: {type(hand_drawn).__name__}")
    if not isinstance(orientation, str):
        raise TypeError(f"Unsupported orientation type: {type(orientation).__name__}")
    orientation = orientation.strip().lower()

    angle_delta: int = 0
    if hand_drawn:
        angle_delta = int(round(random.normalvariate(mu=0.0, sigma=2.0)))
        angle_delta = max(-5, min(angle_delta, 5))

    angle_deg: int
    if orientation == "horizontal":
        angle_deg = 0
    elif orientation == "vertical":
        angle_deg = 90
    elif orientation == "diagonal_primary":
        angle_deg = 45
    elif orientation == "diagonal_auxiliary":
        angle_deg = -45
    else:
        raise ValueError(f"Invalid orientation value: {orientation}.")

    angle_deg += angle_delta
    if angle_deg > 90:
      angle_deg -= 180

    x1: float = random.uniform(xmin, 0.75 * xmax)
    y1: float = random.uniform(ymin, 0.75 * ymax)
    x2: float
    y2: float

    if abs(angle_deg) == 90:
        x2 = x1
        y2 = random.uniform(y1 + 1, ymax)
    elif angle_deg == 0:
        x2 = random.uniform(x1 + 1, xmax)
        y2 = y1
    else:
        slope: float = math.tan(math.radians(angle_deg))
        xmax_adjusted: float = min(xmax, x1 + (ymax - y1) / slope)
        x2 = min(xmax, random.uniform(x1 + 1, xmax_adjusted))
        y2 = min(ymax, y1 + slope * (x2 - x1))

    return [x1, x2], [y1, y2]
