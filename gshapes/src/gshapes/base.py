"""
base.py
-------

Defines the abstract base class for all geometric primitives.
Provides common metadata handling, RNG-safe initialization,
and efficient object reuse through the `reset()` interface.
"""

from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from matplotlib.axes import Axes


class Primitive(ABC):
    """
    Base class for drawable geometry primitives (line, circle, arc, etc.).

    Provides:
      - Shared metadata storage (`self.meta`)
      - Deterministic RNG compatibility
      - Efficient object reuse via `reset()`
      - Dict-based interface (`to_dict()`)
      - Extensible `.draw(ax)` and `.make_geometry()` methods

    Design:
      Each worker or thread can hold a single instance of a subclass (e.g., Line)
      and repeatedly call `reset(ax, **kwargs)` to update geometry and re-render.

    Attributes:
        meta (dict[str, Any]): Metadata describing the primitive’s geometry and style.

    Example:
        >>> line = Line(ax)
        >>> line.draw(ax)
        >>> line.reset(ax, orientation="vertical").draw(ax)
    """

    __slots__ = ("meta",)

    def __init__(self, meta: Optional[Dict[str, Any]] = None):
        """Initialize with optional metadata."""
        self.meta: Dict[str, Any] = meta or {}

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------
    @abstractmethod
    def make_geometry(self, ax: Axes, **kwargs) -> Dict[str, Any]:
        """Generate the metadata dictionary describing this primitive."""
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax: Axes, **kwargs) -> Primitive:
        """Render the primitive onto a Matplotlib axis."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Reuse & conversion utilities
    # -------------------------------------------------------------------------
    def reset(self, ax: Axes, **kwargs) -> Primitive:
        """
        Recompute metadata in-place for reuse.

        Args:
            ax: Target Matplotlib Axes object.
            **kwargs: Primitive-specific arguments for regeneration.

        Returns:
            self (Primitive): Updated object for chaining.
        """
        self.meta.clear()
        self.meta.update(self.make_geometry(ax, **kwargs))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a deep copy of the current metadata dictionary.

        This ensures that mutable sub-objects (e.g., lists) cannot
        be modified externally without affecting the original.
        """
        return copy.deepcopy(self.meta)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} keys={list(self.meta.keys())}>"
