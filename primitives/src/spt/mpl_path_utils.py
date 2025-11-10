"""
mpl_path_utils.py
-----------

Core API:

    join_paths(paths: list[mplPath], preserve_moveto: bool = False) -> mplPath

        Joins muptiple paths into a single continous or disjoint matplotlib.path.Path
        object.
    
    ellipse_or_arc_path(x0: float, y0: float, r: float, y_compress: float = 1.0,
                        start_angle: float = 0.0, end_angle: float = 360.0,
                        angle_offset: float = 0.0, close: bool = False) -> mplPath:

        This routine creates basic primitives using Matplotlib Circle, Ellipse, and Arc
        patches. For these patches, Matplotlib internally creates an associated path
        that is extracted after the appropriate patch object is created. The primary
        purpose of this routine is satisfy potential needs of testing or basic demos.
        Flexible generation of primary elliptical arcs objects is performed via separate
        routines.
"""

from __future__ import annotations

"""
__all__ = [
    "join_paths", "ellipse_or_arc_path",
    "JITTER_ANGLE_DEG",
]
"""

import os
import sys
import time
import random
import math
from typing import TypeAlias, Union
import numpy as np

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Arc
from matplotlib.path import Path as mplPath

from mpl_utils import *


numeric = Union[int, float]
PointXY = tuple[numeric, numeric]
CoordRange = tuple[numeric, numeric]
JITTER_ANGLE_DEG = 5


# ---------------------------------------------------------------------------
# Path joining utility
# ---------------------------------------------------------------------------
def join_paths(paths: list[mplPath], preserve_moveto: bool = False) -> mplPath:
    """
    Join multiple Matplotlib ``Path`` objects into a single composite path.
    
    Args:
        paths (list[matplotlib.path.Path]):
            Input list of path objects to join.
        preserve_moveto (bool, optional):
            Whether to keep the initial ``MOVETO`` command for each path.
            If ``False`` (default), subsequent paths are joined seamlessly
            into one continuous path.
    
    Returns:
        matplotlib.path.Path:
            The concatenated composite path.
    
    Raises:
        ValueError: If ``paths`` is empty.
        TypeError: If any element of ``paths`` is not a ``Path`` instance.
    """
    if not paths:
        raise ValueError("Expected a non-empty list of Matplotlib paths.")

    for path in paths:
        if not isinstance(path, mplPath):
            raise TypeError(f"Expected a list of Matplotlib paths, got {type(path).__name__}.")

    verts_list, codes_list = [paths[0].vertices], [paths[0].codes]

    start = 0 if preserve_moveto else 1
    for path in paths[1:]:
        if path.vertices.size == 0:
            continue
        verts_list.append(path.vertices[start:])
        codes_list.append(path.codes[start:])

    return mplPath(np.concatenate(verts_list), np.concatenate(codes_list))


# ---------------------------------------------------------------------------
# Basic Ellipse / Arc path generator
# ---------------------------------------------------------------------------
def ellipse_or_arc_path(x0: float, y0: float, r: float, y_compress: float = 1.0,
                        start_angle: float = 0.0, end_angle: float = 360.0,
                        angle_offset: float = 0.0, close: bool = False) -> mplPath:
    """
    Create a basic Matplotlib Path representing a circle, ellipse, or elliptical arc.
  
    Supports anisotropic vertical scaling via ``y_compress`` and rotation via
    ``angle_offset``. Optionally closes the arc to form a filled sector (pie slice).

    Args:
        x0: Center X-coordinate.
        y0: Center Y-coordinate.
        r: Horizontal radius before compression (Rx).
        y_compress: Vertical compression factor (Ry = r * y_compress).
        start_angle: Start angle in degrees (0deg = +X axis).
        end_angle: End angle in degrees.
        angle_offset: Rotation angle in degrees for the ellipse major axis.
        close: If True, closes the arc to the center (for filled sectors).
  
    Returns:
        mplPath: Path representing the circle, ellipse, or arc segment.
    """
    start_angle = float(start_angle)
    end_angle = float(end_angle)
    span = (end_angle - start_angle) % 360.0

    rx = r
    ry = r * y_compress

    # --- Full circle / ellipse ---

    if abs(span) < 1e-6 or abs(span - 360.0) < 1e-6:
        if abs(y_compress - 1.0) < 1e-9 and abs(angle_offset) < 1e-9:
            patch = Circle((x0, y0), rx)
        else:
            patch = Ellipse((x0, y0), 2 * rx, 2 * ry, angle=angle_offset)
        return patch.get_transform().transform_path(patch.get_path())

    # --- Partial arc / elliptical arc ---

    arc = Arc((x0, y0), 2 * rx, 2 * ry, angle=angle_offset,
              theta1=start_angle, theta2=end_angle)
    path = arc.get_path()
    transformed = arc.get_transform().transform_path(path)

    if not close:
        return transformed

    # --- Close arc to center (filled sector) ---

    verts = transformed.vertices
    codes = transformed.codes.copy()

    verts_closed = np.vstack([
        verts,
        [x0, y0],           # line to center
        verts[0],           # close back to start
    ])
    codes_closed = np.concatenate([
        codes,
        [mplPath.LINETO, mplPath.CLOSEPOLY],
    ])

    return mplPath(verts_closed, codes_closed)
