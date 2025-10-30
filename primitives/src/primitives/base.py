"""
base.py
-------

Defines the abstract base class for drawable geometric primitives.

Each primitive encapsulates:
  - Metadata describing its geometry and appearance.
  - A reproducible RNG context (external).
  - A resettable lifecycle for high-throughput reuse.

Thread model:
  - Each worker or thread creates a single Primitive-derived instance.
  - Instances are reused via `.reset(ax, **kwargs)` without reallocation.

Example:
    >>> from primitives.line import Line
    >>> line = Line(ax)
    >>> line.draw(ax)
    >>> line.reset(ax, orientation="vertical").draw(ax)
"""

from __future__ import annotations

import os
import sys
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from matplotlib.axes import Axes

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
      - Shared metadata storage (`self.meta`)
      - Deterministic RNG compatibility
      - Efficient object reuse via `reset()`
      - Dict-based interface (`to_dict()`)
      - Extensible `.draw(ax)` and `.make_geometry()` methods

    Design:
      Subclasses (e.g. Line, Circle, Ellipse) implement:
        - `make_geometry(ax, **kwargs)` -- returns dict of geometry/style data
        - `draw(ax, **kwargs)` -- renders using current metadata

    Attributes:
        meta (dict[str, Any]): Metadata describing the primitive's geometry and style.

    Example:
        >>> line = Line(ax)
        >>> line.draw(ax)
        >>> line.reset(ax, orientation="vertical").draw(ax)
    """

    __slots__ = ("meta",)

    rng: RNG = get_rng(thread_safe=True)  # class-level RNG shared by all instances

    def __init__(self, meta: Optional[Dict[str, Any]] = None):
        """
        Initialize the primitive.

        Args:
            meta: Optional precomputed metadata dictionary.
            rng: Optional RNG instance for deterministic behavior.
                 If None, a thread-local RNG is used.
        """
        self.meta: Dict[str, Any] = meta or {}

    @classmethod
    def reseed(cls, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG (for deterministic replay)."""
        cls.rng.seed(seed)

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------
    @abstractmethod
    def make_geometry(self, ax: Axes, **kwargs) -> Dict[str, Any]:
        """Generate metadata describing the primitive's geometry."""
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax: Axes, **kwargs) -> Primitive:
        """Render the primitive onto the given Matplotlib axis."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Reuse and conversion utilities
    # -------------------------------------------------------------------------
    def reset(self, ax: Axes, **kwargs) -> Primitive:
        """
        Recompute metadata in place for object reuse.

        Clears and regenerates `self.meta` using subclass-specific
        `make_geometry()`, allowing the same object to be reused
        across many iterations or frames.

        Args:
            ax: Target Matplotlib Axes.
            **kwargs: Parameters forwarded to `make_geometry()`.

        Returns:
            self: Updated object for chaining.
        """
        self.make_geometry(ax, **kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a deep copy of the metadata dictionary.

        Useful for safe serialization (e.g. JSON export) or
        inspection without mutating the live state.
        """
        return copy.deepcopy(self.meta)

    def __repr__(self) -> str:
        """Readable summary showing available metadata keys."""
        return f"<{self.__class__.__name__} keys={list(self.meta.keys())}>"
