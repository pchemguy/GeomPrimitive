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
import json
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

    __slots__ = ("_meta",)

    rng: RNG = get_rng(thread_safe=True)  # class-level RNG shared by all instances

    def __init__(self, ax: Axes, **kwargs: Any) -> None:
        """
        Create a primitive and immediately generate its geometry.

        Args:
            ax (matplotlib.axes.Axes): Target Matplotlib axis.
            **kwargs: Optional arguments for `make_geometry`.
        """
        if not isinstance(ax, Axes):
            raise TypeError("Primitive constructor requires a Matplotlib Axes object")

        self._meta: Dict[str, Any] = {}
        self.make_geometry(ax, **kwargs)  # always generate metadata

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
        return json.dumps(self._meta, sort_keys=True, separators=(",", ":"))    
    
    @classmethod
    def reseed(cls, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG (for deterministic replay)."""
        cls.rng.seed(seed)

    @property
    def jsonpp(self) -> str:
      """Return pretty-printed JSON (good for debugging / logs)."""
      return json.dumps(self._meta, sort_keys=True, indent=4)

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------
    @abstractmethod
    def make_geometry(self, ax: Axes, **kwargs) -> Primitive:
        """Generate metadata describing the primitive's geometry."""
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax: Axes, **kwargs) -> Primitive:
        """Render the primitive onto the given Matplotlib axis."""
        raise NotImplementedError

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
