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
from matplotlib.transforms import Affine2D

from mpl_utils import *
from rng import RNGBackend, RNG, get_rng

numeric: TypeAlias = Union[int, float]
PointXY: TypeAlias = tuple[numeric, numeric]
CoordRange: TypeAlias = tuple[numeric, numeric]

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
# Applies random SRT transform to Path.
# ---------------------------------------------------------------------------
def random_srt_path(shape: mplPath,
                    canvas_x1x2: PointXY,
                    canvas_y1y2: PointXY,
                    y_compress: float = None,
                    angle_deg: numeric = None,
                    origin: PointXY = None,
                    rng: RNGBackend = None,
                   ) -> tuple[mplPath, dict]:
    """
    Apply a random Scale-Rotate-Translate (SRT) transform to a Path so it fits
    inside a given canvas box with some jitter.

    The shape is:
        1) scaled uniformly in X and Y, plus by `y_compress` in Y,
        2) rotated around `origin` (default: path bbox center),
        3) translated into the canvas with small random offsets.

    Args:
        shape:
            Input Matplotlib Path.
        canvas_x1x2:
            (xmin, xmax) of the target canvas.
        canvas_y1y2:
            (ymin, ymax) of the target canvas.
        y_compress:
            Optional vertical compression factor. If None, sampled in [0.5, 1.0].
        angle_deg:
            Base rotation in degrees. If None, sampled uniformly in [0, 360).
            If non-zero, a small +/-JITTER_ANGLE_DEG jitter is added.
            Note, for non-zero angle, the value is rounded. To have jitter with
            0 deg, set it to abs() < 0.5 deg, such as 0.1.
        origin:
            Optional rotation center (x, y) in path coordinates. If None, uses
            the path bounding-box center.
        rng:
            Optional RNG backend. If None, uses get_rng(thread_safe=True).

    Returns:
        (new_path, meta) where:
            new_path: transformed Path
            meta: dict with scale, rotation, and translation parameters.
    """
    # --- Type checks --------------------------------------------------------
    if not isinstance(shape, mplPath):
        raise TypeError(f"Expected a Matplotlib Path, got {type(shape).__name__}.")

    if not isinstance(canvas_x1x2, tuple) or not isinstance(canvas_y1y2, tuple):
        raise TypeError(
            f"canvas_x1x2: {type(canvas_x1x2).__name__}\n"
            f"canvas_y1y2: {type(canvas_y1y2).__name__}\n"
            "Both must be tuple[float, float]."
        )

    if origin is not None and not isinstance(origin, tuple):
        raise TypeError(
            "If provided, origin should be tuple[float, float]. "
            f"Received: {type(origin).__name__}"
        )

    # --- RNG ---------------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)

    # --- Canvas geometry ---------------------------------------------------
    cxmin, cxmax = map(float, canvas_x1x2)
    cymin, cymax = map(float, canvas_y1y2)

    cw, ch = cxmax - cxmin, cymax - cymin
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Canvas width/height must be positive; got ({cw}, {ch}).")

    cx0, cy0 = (cxmax + cxmin) / 2, (cymax + cymin) / 2

    # --- Vertical compression ----------------------------------------------
    if not isinstance(y_compress, (int, float)):
        y_compress = rng.uniform(0.5, 1.0)
    y_compress = max(0.2, float(y_compress))    
    
    # --- Angle with jitter -------------------------------------------------
    if not isinstance(angle_deg, (int, float)):
        angle_deg = rng.randrange(360)
    elif angle_deg != 0:
        jitter = JITTER_ANGLE_DEG * max(-1.0, min(1.0, rng.normalvariate(0.0, 1/3)))
        angle_deg = float(round(angle_deg) + jitter)
    # Normalize to [-180, 180)
    angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
    angle_rad = math.radians(angle_deg)

    # --- Original path bbox -----------------------------------------------
    bbox = shape.get_extents()
    bxmin, bymin, bxmax, bymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    bx0, by0 = (bxmax + bxmin) / 2, (bymax + bymin) / 2
    bw, bh = bbox.width, bbox.height

    # --- Rotated unscaled bounding dimensions (approximate) ---------------
    bwsize = max(1e-6, abs(bw * math.cos(angle_rad)) + abs(bh * math.sin(angle_rad)))
    bhsize = max(1e-6, abs(bh * math.cos(angle_rad)) + abs(bw * math.sin(angle_rad)))
    

    # --- Scaling to fit canvas --------------------------------------------
    # Base uniform scale in X chosen to fit rotated bbox into canvas
    sfx = rng.uniform(0.2, 1) * min(cw / bwsize, ch / bhsize)
    sfy = sfx * y_compress

    # --- Translation jitter ------------------------------------------------
    tx_range = (cw - bwsize * sfx) / 2
    ty_range = (ch - bhsize * sfy) / 2

    tx = cx0 - bx0 * sfx + tx_range * rng.uniform(-1, 1)
    ty = cy0 - by0 * sfy + ty_range * rng.uniform(-1, 1)
    
    # --- Rotation origin ---------------------------------------------------
    if origin is None:
        origin = (bx0, by0)


    # --- Affine transform: scale -> rotate_around -> translate ------------
    trans: Affine2D = (
        Affine2D()
        .scale(sfx, sfy)
        .rotate_around(origin[0], origin[1], angle_rad)
        .translate(tx, ty)
    )

    verts_array = trans.transform(shape.vertices)
    
    meta: dict = {
        "scale_x": float(sfx),
        "scale_y": float(sfy),
        "rot_x":   float(origin[0]),
        "rot_y":   float(origin[1]),
        "rot_deg": float(angle_deg),
        "trans_x": float(tx),
        "trans_y": float(ty),
    }
    
    return mplPath(verts_array, shape.codes), meta


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

