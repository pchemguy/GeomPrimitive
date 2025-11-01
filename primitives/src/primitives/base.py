"""
base.py
-------

Defines the abstract base class for drawable geometric primitives.
"""

from __future__ import annotations

import os
import sys
import copy
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib import colors

# =============================================================================
# Constants
# =============================================================================
MAX_PATTERN_LENGTH = 30
MAX_DASH_JITTER = 0.1
MAX_ANGLE_JITTER = 5
LOGGER_NAME = "worker" if logging.getLogger("worker").handlers else "root"
CSS4_COLOR_NAMES = list(colors.CSS4_COLORS.keys())

# -----------------------------------------------------------------------------
# Import thread-safe RNG utilities
# -----------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rng import RNG, get_rng
else:
    from .rng import RNG, get_rng


class Primitive(ABC):
    """
    Abstract base class for all drawable geometric primitives  (line, circle, arc, etc.).

    Provides:
      - Deterministic RNG compatibility
      - Extensible `.draw(ax)` and `.make_geometry()` methods

    Design:
      Subclasses (e.g. Line, Circle, Ellipse) implement:
        - `make_geometry(ax, **kwargs)` -- self
        - `draw(ax, **kwargs)` -- renders using current metadata

    Attributes:
      - _meta (dict[str, Any]): Metadata describing the primitive's geometry and style.
      - meta: deep-copy getter for safe inspection.
      - json: JSON-encoded serialization of `.meta`.

    Example:
        >>> line = Line(ax)
        >>> line.draw(ax)
    """

    __slots__ = ("_meta", "_ax", "logger",)

    rng: RNG = get_rng(thread_safe=True)  # class-level RNG shared by all instances

    def __init__(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        """
        Create a primitive and immediately generate its geometry.

        Args:
            ax (matplotlib.axes.Axes): Target Matplotlib axis.
            **kwargs: Optional arguments for `make_geometry`.
        """
        self._meta: Dict[str, Any] = {}
        if isinstance(ax, Axes):
            self._ax = ax
            self.make_geometry(**kwargs)  # always generate metadata
        self.logger = logging.getLogger(LOGGER_NAME)


    # ---------------------------------------------------------------------------
    # Metadata accessors
    # ---------------------------------------------------------------------------
    @property
    def meta(self) -> Dict[str, Any]:
        """Deep copy of the primitive’s current metadata (safe to mutate)."""
        return copy.deepcopy(self._meta)

    @property
    def json(self) -> str:
        """JSON-encoded metadata string (sorted, compact)."""
        return json.dumps(self._meta, sort_keys=True, separators=(",", ":"), default=str)    
    
    @classmethod
    def reseed(cls, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG (for deterministic replay)."""
        cls.rng.seed(seed)

    @property
    def jsonpp(self) -> str:
      """Return pretty-printed JSON (good for debugging / logs)."""
      return json.dumps(self._meta, sort_keys=True, indent=4, default=str)

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------
    @abstractmethod
    def make_geometry(self, ax: Optional[Axes] = None, **kwargs) -> Primitive:
        """Generate metadata describing the primitive's geometry.
        Subclasses must override this method.
    
        Responsibilities:
          - Compute and assign all necessary geometric + stylistic metadata.
          - Populate `self._meta` with a dictionary compatible with Matplotlib
            rendering calls (e.g., for `ax.plot`, `ax.add_patch`, etc.).
          - Return `self` for chaining.
        """                
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> Primitive:
        """Render the primitive onto the given Matplotlib axis."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    @classmethod
    def _get_linestyle(cls, pattern: Optional[str], hand_drawn: bool
                      ) -> Union[str, Tuple[int, Tuple[float, ...]]]:
        rng: RNG = cls.rng
        if pattern is None or (isinstance(pattern, str) and not pattern.strip()):
            if not hand_drawn:
                pattern = tuple(rng.randint(1, 3) for _ in range(2 * rng.randint(1, 5)))
                return (0, pattern)

            base_len = float(rng.randint(1, 3))
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
    def _get_angle(cls, orientation: Union[str, int, None], hand_drawn: bool) -> Union[float, None]:
        rng: RNG = cls.rng
        if orientation is None:
            return None

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
            angle += max(-MAX_ANGLE_JITTER, min(rng.normal(0.0, 2.0), MAX_ANGLE_JITTER))
            if angle > 90:
                angle -= 180

        return angle

    # ---------------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Readable summary showing available metadata keys."""
        return f"<{self.__class__.__name__} keys={list(self.meta.keys())}>"

    def __str__(self) -> str:
        """
        Return a detailed, human-readable string representation.
      
        Example output:
            Line(id=0x1f2c4fa2):
            {
              "alpha": 0.75,
              "color": "steelblue",
              "orientation": "horizontal",
              ...
            }
        """
        cls_name: str = self.__class__.__name__
        obj_id: str = hex(id(self))
        return f"{cls_name}(id={obj_id}):\n{self.jsonpp}"
