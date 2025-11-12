"""
mpl_path_utils.py
-----------

The primary focus of this module is generation of random Paths of geometric
primitives, such as lines, circles/ellipses, circular/elliptical arcs, triangles,
squares/recatngles, and tabulated functions.

The core generation workflow is performed in stages:
 1. Generation of a random Path of basic shapes:
      - line segment,
      - unit circle and arc,
      - square,
      - arbitrary triangle.
      - arbitrary tabulated functions {(x, y) or (x, y, y')}.
    Circular arc is part of the unit circle, while line segments, triangles, and
    squares are inscribed into the unit circle. This convention, paired with polar
    coordinates enables convinient workflow for randomization of shape during this
    stage essentially independent of size, orientation, and position (randomized in
    subsequent stages).
    
    Particularly convinient are polar coordinates, which provide an expressive full
    control over the shape of triangles. Generation of elliptical and general
    rectangular shaoes is unnecessary at this stage, as the full spectrum of
    elliptical and rectangular shapes is generated from circular and square shapes
    via anisotropic scaling (using separate scaling factors for Cartesian X and Y)
    during the random SRT transform. Because SRT transform is responsible for
    randomizing size/scale, no need to worry about size randomization at this stage
    either, that is only polar angles of vertices on the unit cicle are randomized,
    but not circle radius.
    
    This stage also introduces jitter on initial angaular coordinates and then
    Cartesian XY coordinates, after coordinate switching. Jitter distorts idealized
    shapes (circles, regular polygons, right angles, specific triangle subtypes),
    imitating realistic non-idealized rendering.

    Because circles are rendered as piecewise cubic Bezier splines, jittering
    cordinates introduces distortions to arc segments, imitating non-idealized hand
    drawing.

    For other primitives, which are essentially straight segment polylines, hand
    drawing imitation is peformed in a separate stage.

 2. Hand drawing imitation.
    The second stage focuses on distorting idealized straight line segments forming
    polyine primitives (line segments and polygons). Each straight segment is split
    into a configurable number of subsegments of randomized length. Each subsegment
    is replaced with Matplotlib cubic Bezier segment, slightly and randomly
    deviating from a straight line.

    Tabulated functions Paths are not subjected to this process.

 3. Random Scale->Rotate->Translate (SRT) transform. This stage randomizes the size,
    orientation, and position of Paths from prior stages. For cicles, arcs, and
    squares, independent (anisotropic) XY scaling generates the full spectrum of
    rectangles and elliprical shapes.

Separate processes are used for genration of randomized
 - line rendering style (linewidth, linestyle, and color);
 - background color;
 - idealized or randomly distorted grid.

Note, grid is generated as four LineCollection objects (X and Y - major and minor),
added to the Matplotlib plot via corresponding four drawing calls. Paths are wrapped
in Patch objects with generated styles. Use of PatchCollection is generally unnecessary
here, as the main objective is generating scenes with small number of independently
styled shapes. While PathCollection probably provides sufficient flaxibility for this
purpose, its use would complicate the code, while typically not providing considerable
performance benefit.

Core API:

    join_paths(paths: list[mplPath], preserve_moveto: bool = False) -> mplPath

        Joins muptiple paths into a single continous or disjoint matplotlib.path.Path
        object.
    
    
    random_srt_path(shape: mplPath, canvas_x1x2: PointXY, canvas_y1y2: PointXY,
                    y_compress: float = None, angle_deg: numeric = None,
                    origin: PointXY = None, rng: RNGBackend = None) -> tuple[mplPath, dict]

        Applies a random Scale->Rotate->Translate (SRT) transform to given Path within
        the specified canvas. the primary use is to transform a random Path defined on
        the unit box ([-1, 1]) to a rnadom Path within the specified canvas. Scaling is
        defined assymetrically. Isotropic scaling within the ratio of the sizes of the
        original bounding box and canvas is performed isotropically (XY). y_compress
        defines additional compression of the y coordinate, turning circles, circular
        arcs, and squares into elliptical and rectangular shapes.
    
    
    unit_circle_diameter(base_angle: numeric = None,
                         jitter_angle_deg: int = JITTER_ANGLE_DEG,
                         rng: RNGBackend = None) -> tuple[mplPath, dict]

        Generates a randomly oriented unit circle diamter (line segment).
    
    
    unit_circular_arc(start_deg: float = 0.0, end_deg: float = 90.0,
                      jitter_amp: float = 0.02, jitter_y: float = 0.1,
                      max_angle_step_deg: float = 20.0, min_angle_steps: int = 3,
                      rng: RNGBackend = None) -> mplPath

        Generates a Path representation of a unit circular arc modeled as a chain of
        cubic Bezier splines. The routine optionally introduces angular jitter on angular
        cordinates, aspect ration (distorting the circle), line shape, imitating hand
        drawing.

    
    unit_rectangle_path(equal_sides: int = None, jitter_angle_deg: int = 5,
                        base_angle: int = None, rng: RNGBackend = None) -> mplPath

        Generates a Path representation of a rectangle inscribed into a unit circle.


    unit_triangle_path(equal_sides: int = None, angle_category: int = None,
                       jitter_angle_deg: int = 5, base_angle: int = None,
                       rng: RNGBackend = None) -> tuple[mplPath, dict]

        Generates a Path representation of a triangle inscribed into a unit circle.

    
    random_cubic_spline_segment(start: PointXY, end: PointXY, amp: float = 0.15,
                                tightness: float = 0.3, rng: RNGBackend = None) -> mplPath

        Transforms a single straight line segment into a single randomized cubic Bezier
        segment, imitating hand drawing. Suitable for short segments.
            
    
    handdrawn_polyline_path(points: list[PointXY], splines_per_segment: int = 5,
                            amp: float = 0.15, tightness: float = 0.3,
                            rng: RNGBackend = None) -> mplPath

        Given open or closed polyline Path, creates a new Path that imitates hand
        drawn style by replacing each straight segment with configurable number of
        Bezier sections.

    
    bezier_from_xy_dy(x: NDarray, y: NDarray, dy: NDarray = None, tension: float = 0.0,
                      endpoint_style: str = "default") -> mplPath
    
        Creates a Path containing piecewise spline representation of a tabulated
        function.

    
    unit_circular_arc_segment(start_deg: float = 0.0, end_deg: float = 90.0) -> mplPath
    
        Creates a single Bezier segment approximation of an acute circular arc.
    
    
    basic_ellipse_or_arc_path(x0: float, y0: float, r: float, y_compress: float = 1.0,
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
from numpy.typing import NDArray

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle, Ellipse, Arc
from matplotlib.path import Path as mplPath
from matplotlib.transforms import Affine2D

from mpl_utils import *
from utils.rng import RNGBackend, RNG, get_rng

numeric: TypeAlias = Union[int, float]
PointXY: TypeAlias = tuple[numeric, numeric]
CoordRange: TypeAlias = tuple[numeric, numeric]

JITTER_ANGLE_DEG = 5


# ---------------------------------------------------------------------------
# Path joining utility
# ---------------------------------------------------------------------------
def join_paths(
        paths           : list[mplPath],
        preserve_moveto : bool           = False,
    ) -> mplPath:
    """Join multiple Matplotlib ``Path`` objects into a single composite path.
    
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
def random_srt_path(
        shape            : mplPath,
        canvas_x1x2      : PointXY,
        canvas_y1y2      : PointXY,
        y_compress       : float      = None,
        angle_deg        : numeric    = None,
        jitter_angle_deg : int        = JITTER_ANGLE_DEG,
        origin           : PointXY    = None,
        rng              : RNGBackend = None,
    ) -> tuple[mplPath, dict]:
    """Apply a random Scale-Rotate-Translate (SRT) transform to a Path so it fits
    inside a given canvas box with some jitter.

    The shape is:
        1) scaled uniformly in X and Y, plus by `y_compress` in Y,
        2) rotated around `origin` (default: path bbox center),
        3) translated into the canvas with small random offsets.

    The scaling process aims for the final shape to be within 0.1x to 0.9x fraction
    of the smaller canvas dimension. However, at high extreme ends (combined with
    translation and rotation), part of the Path may ocassionaly extend beyond the
    canvas, also ocasionally failing the associated test.

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
        jitter_angle_deg: Angular jitter amplitude in degrees.
            Controls small random deviations around the base angle.
        origin:
            Optional rotation center (x, y) in path coordinates. If None, uses
            the path bounding-box center.
        rng:
            Optional RNG backend. If None, uses get_rng(thread_safe=True).

    Returns:
        (new_path, meta) where:
            new_path: transformed Path
            meta: dict with scale, rotation, and translation parameters.

    TODO: Overall calculations are inaccurate (pivot corrdinates are not taken
          into account, Matplotlib's .get_extents() is inaccurate also).
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
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

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
        angle_deg = round(angle_deg) + jitter_angle_deg * normal3s()
    # Normalize to [-180, 180)
    angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
    angle_rad = math.radians(angle_deg)

    # --- Original path bbox -----------------------------------------------
    # TODO: get_extents() does not really work correctly and should  be replaced.
    # TODO: See mpl_artist_preview.py - ax_autofit()
    bbox = shape.get_extents()
    bxmin, bymin, bxmax, bymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    bx0, by0 = (bxmax + bxmin) / 2, (bymax + bymin) / 2
    bw, bh = bbox.width, bbox.height

    # --- Rotated unscaled bounding dimensions (approximate) ---------------
    # TODO: This does not take into account pivot coordinates, and so is relatively innacurate.
    bwsize = max(1e-6, abs(bw * math.cos(angle_rad)) + abs(bh * math.sin(angle_rad)))
    bhsize = max(1e-6, abs(bh * math.cos(angle_rad)) + abs(bw * math.sin(angle_rad)))

    # --- Scaling to fit canvas --------------------------------------------
    # Base uniform scale in X chosen to fit rotated bbox into canvas
    sfx = rng.uniform(0.2, 0.9) * min(cw / bwsize, ch / bhsize)
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
        "operation" : "SRT",
        "scale_x"   : sfx,
        "scale_y"   : sfy,
        "rot_x"     : origin[0],
        "rot_y"     : origin[1],
        "rot_deg"   : angle_deg,
        "trans_x"   : tx,
        "trans_y"   : ty,
    }
    
    return mplPath(verts_array, shape.codes), meta


# ---------------------------------------------------------------------------
# Random unit circle diameter (straight line segment)
# ---------------------------------------------------------------------------
def unit_circle_diameter(
            base_angle       : numeric    = None,
            jitter_angle_deg : int        = JITTER_ANGLE_DEG,
            rng              : RNGBackend = None,
        ) -> tuple[mplPath, dict]:
    """Generates a diameter (straight line) within a unit circle.

    The line passes through the circle center (0, 0) and connects opposite
    points on the unit circle at a given or random orientation.

    Args:
        base_angle: Optional base orientation angle in degrees.
            If None, randomly sampled from [-90, 90].
        jitter_angle_deg: Angular jitter amplitude in degrees.
            Controls small random deviations around the base angle.
        rng: Optional RNG backend (RNG, random.Random, or np.random.Generator).
            If None, uses `get_rng(thread_safe=True)`.

    Returns:
        tuple[mplPath, dict]:
            - Path: Matplotlib path representing the circle diameter.
            - dict: Metadata including the final angle in degrees.
    """
    # --- RNG ----------------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    # --- Determine base angle and jitter -----------------------------------
    if base_angle is None:
        base_angle = rng.uniform(-90, 90)
    elif not isinstance(base_angle, (int, float)):
        raise TypeError(
            f"angle_deg must be of type integer or float.\n"
            f"Received type: {type(base_angle).__name__}; value: {base_angle}."
        )

    # Use RNG for normal jitter to stay consistent with other random primitives
    jitter = normal3s() * jitter_angle_deg
    angle_deg = ((base_angle + jitter + 90) % 180) - 90
    angle_rad = math.radians(angle_deg)

    # --- Compute endpoints on unit circle ----------------------------------
    x1, y1 = math.cos(angle_rad), math.sin(angle_rad)
    x2, y2 = -x1, -y1  # opposite side

    # --- Build Path ---------------------------------------------------------
    verts = [(x1, y1), (x2, y2)]
    codes = [mplPath.MOVETO, mplPath.LINETO]
    path = mplPath(verts, codes)

    meta: dict = {
        "shape_kind" : "line",
        "angle_deg"  : angle_deg,
    }

    return path, meta


# ---------------------------------------------------------------------------
# Random unit circular arc
# ---------------------------------------------------------------------------
def unit_circular_arc(
        start_deg          : float      = 0.0,
        end_deg            : float      = 90.0,
        jitter_amp         : float      = 0.02,
        jitter_y           : float      = 0.1,
        max_angle_step_deg : float      = 20.0,
        min_angle_steps    : int        = 3,
        rng                : RNGBackend = None,
    ) -> mplPath:
    """Generate a unit circular arc as a multi-segment cubic Bezier path.

    Each sub-arc spans <= `max_angle_step_deg` (default 20deg) and uses the
    analytic 4/3*tan(Dtheta/4) handle length for optimum circular curvature.
    Additive and multiplicative jitter simulate hand-drawn irregularities.

    Note: Uses vectorized random function - requires NumPy backend.

    Args:
        start_deg: Starting angle in degrees.
        end_deg: Ending angle in degrees.
        jitter_amp: Additive jitter amplitude (fraction of radius).
        jitter_y: Multiplicative jitter (vertical squish).
        max_angle_step_deg: Maximum sub-arc span (degrees).
        min_angle_steps: Minimum number of Bezier segments.
        rng: Optional RNG backend (`RNG`, `random.Random`, or `np.random.Generator`).

    Returns:
        tuple:
            matplotlib.path.Path - Bezier path approximating the arc.
            dict - Metadata about span, closure, and parameters.
    """
    # --- RNG setup -----------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    # --- Angle normalization -------------------------------------------------
    if start_deg is None or end_deg is None:
        start_deg = rng.uniform(0, 360 - JITTER_ANGLE_DEG * 3)
        end_deg = rng.uniform(start_deg + JITTER_ANGLE_DEG, 360)
    span_deg = end_deg - start_deg

    if span_deg < 1 or span_deg > 359:
        start_deg, end_deg, span_deg = 0, 360, 360
        closed = True
    else:
        closed = False

    # --- Segmentation -------------------------------------------------------
    theta_steps = int(max(min_angle_steps, round(span_deg / max_angle_step_deg)))
    start, end = math.radians(start_deg), math.radians(end_deg)
    span = end - start
    step_theta = span / theta_steps
    t = 4.0 / 3.0 * math.tan(step_theta / 4.0)

   # --- Vertex generation --------------------------------------------------
    P0 = (math.cos(start), math.sin(start))
    verts: list[PointXY] = [P0]
    theta_beg = start
    for _ in range(theta_steps):
        theta_end = theta_beg + step_theta
        cos_b, sin_b = math.cos(theta_beg), math.sin(theta_beg)
        cos_e, sin_e = math.cos(theta_end), math.sin(theta_end)

        P1 = (cos_b - t * sin_b, sin_b + t * cos_b)
        P2 = (cos_e + t * sin_e, sin_e - t * cos_e)
        P3 = (cos_e, sin_e)

        verts.extend([P1, P2, P3])
        theta_beg = theta_end

    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * theta_steps

    if closed:
        verts.append((np.nan, np.nan))
        codes.append(mplPath.CLOSEPOLY)

    verts = np.array(verts, dtype=float)

    # --- Y-axis multiplicative jitter ----------------------------------------
    if jitter_y:
        verts[:, 1] *= 1 - rng.uniform(0, 1) * jitter_y

    # --- Additive jitter -----------------------------------------------------
    if jitter_amp:
        verts += rng.uniform(-1, 1, size=verts.shape) * jitter_amp

    path = mplPath(verts, codes)

    meta: dict = {
        "shape_kind" : "circle",
        "start_deg"  : start_deg,
        "end_deg"    : end_deg,
    }
    return path, meta


# ---------------------------------------------------------------------------
# Random unit rectangle
# ---------------------------------------------------------------------------
def unit_rectangle_path(
        diagonal_angle   : numeric    = None,
        jitter_angle_deg : int        = JITTER_ANGLE_DEG,
        base_angle       : int        = None,
        rng              : RNGBackend = None,
    ) -> mplPath:
    """Generates a rectangle or square inscribed in a unit circle.

    Args:
        diagonal_angle: Angle between digonals in degrees (90 for square and
                        0-180 for rectangle). Randomly chosen if None.
        jitter_angle_deg: Maximum angular deviation (degrees).
        base_angle: Base rotation of the figure (degrees).
        rng: Optional RNG backend (RNG, random.Random, np.random.Generator).

    Returns:
        Matplotlib Path representing the shape.
    """
    # --- RNG setup ---------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    # --- Shape type --------------------------------------------------------
    if not diagonal_angle is None and not isinstance(diagonal_angle, (int, float)):
        raise TypeError(f"diagonal_angle must be numeric. "
                        f"Received {type(diagonal_angle)}")

    angle_margin = max(jitter_angle_deg, JITTER_ANGLE_DEG) * 2
    if diagonal_angle is None:
        # Using equal chances for square vs. irregular rectangle
        diagonal_angle = 90 + (
            rng.choice((-1, 0, 1, 0)) * rng.uniform(angle_margin, 90 - angle_margin)
        )
    diagonal_angle = max(angle_margin, min(180 - angle_margin, diagonal_angle))

    # --- Angular offsets ---------------------------------------------------

    top_right_angle = diagonal_angle / 2
    offset = top_right_angle - 45

    #  thetas = [
    #      45 + offset + base_angle + normal3s() * jitter_angle_deg,
    #     135 - offset + base_angle + normal3s() * jitter_angle_deg,
    #    -135 + offset + base_angle + normal3s() * jitter_angle_deg,
    #     -45 - offset + base_angle + normal3s() * jitter_angle_deg,
    #  ]

    if not isinstance(base_angle, (int, float)):
        base_angle = rng.uniform(-90, 90)
    else:
        base_angle += normal3s() * jitter_angle_deg

    # --- Corner angles -----------------------------------------------------
    thetas = [
         0 + top_right_angle + base_angle + normal3s() * jitter_angle_deg,
       180 - top_right_angle + base_angle + normal3s() * jitter_angle_deg,
      -180 + top_right_angle + base_angle + normal3s() * jitter_angle_deg,
         0 - top_right_angle + base_angle + normal3s() * jitter_angle_deg,
    ]

    thetas = [math.radians(((theta + 180) % 360) - 180) for theta in thetas]

    # --- Vertices and Path -------------------------------------------------
    verts = [(math.cos(t), math.sin(t)) for t in thetas]
    verts.append(verts[0])  # close shape

    codes = [mplPath.MOVETO] + [mplPath.LINETO] * (len(verts) - 2) + [mplPath.CLOSEPOLY]

    meta: dict = {
        "shape_kind"     : "rectangle",
        "angle_deg"      : base_angle,
        "diagonal_angle" : diagonal_angle,
        "offset_deg"     : offset,
    }
    
    return mplPath(verts, codes), meta


# ---------------------------------------------------------------------------
# Random unit triangle
# ---------------------------------------------------------------------------
def unit_triangle_path(
        equal_sides      : int        = None,
        angle_category   : int        = None,
        jitter_angle_deg : int        = JITTER_ANGLE_DEG,
        base_angle       : int        = None,
        rng              : RNGBackend = None,
    ) -> tuple[mplPath, dict]:
    """Generates vertices of a triangle inscribed into a unit circle.

    Args:
        equal_sides: 1, 2, or 3.
            - 3 -> Equilateral
            - 2 -> Isosceles
            - 1 -> Scalene
        angle_category: Nominal angular type (compared with 90deg):
            <90 -> Acute
            =90 -> Right
            >90 -> Obtuse
        jitter_angle_deg: Standard deviation (3sigma) of angular jitter in degrees.
        base_angle: Optional base rotation of the figure (degrees).
        rng: Optional RNG backend (RNG, random.Random, or np.random.Generator).

    Returns:
        tuple:
            mplPath: Path representing the triangle.
            dict: Metadata including parameters and applied offsets.
    """
    # --- RNG setup ---------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(
        rng, "normal3s",
        lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))),
    )

    # --- Input validation --------------------------------------------------
    if not equal_sides:
        equal_sides = rng.choice((1, 2, 3))
    if equal_sides not in (1, 2, 3):
        raise ValueError(
            f"equal_sides must be an integer in [1, 3]. "
            f"Got {equal_sides!r}."
        )

    if not angle_category:
        angle_category = rng.choice((60, 90, 120))
    if not isinstance(angle_category, (int, float)):
        raise TypeError(
            f"angle_category must be numeric. "
            f"Got {type(angle_category).__name__}."
        )

    # --- Base geometry -----------------------------------------------------
    if equal_sides == 3:
        # Equilateral triangle
        thetas = [90, -30, 210]
        top_offset = base_offset = 0.0
    else:
        top_offset = (
            0 if equal_sides > 1 else rng.choice([-1, 1])
            * rng.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        base_offset = (
            ((angle_category > 90) - (angle_category < 90))
            * rng.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        thetas = [90 + top_offset, 0 + base_offset, 180 - base_offset]

    # --- Jitter and rotation -----------------------------------------------
    thetas[0] += normal3s() * jitter_angle_deg

    if not isinstance(base_angle, (int, float)):
        base_angle = rng.uniform(-90, 90)
    else:
        base_angle += normal3s() * jitter_angle_deg

    thetas = [(theta + base_angle) for theta in thetas]
    thetas = [math.radians(((theta + 180) % 360) - 180) for theta in thetas]

    # --- Path construction -------------------------------------------------
    verts = [(math.cos(theta), math.sin(theta)) for theta in thetas]
    verts.append(verts[0])
    codes = [mplPath.MOVETO, mplPath.LINETO, mplPath.LINETO, mplPath.CLOSEPOLY]

    meta = {
        "shape_kind"      : "triangle",
        "equal_sides"     : equal_sides,
        "angle_category"  : angle_category,
        "base_angle_deg"  : base_angle,
        "top_offset_deg"  : top_offset,
        "base_offset_deg" : base_offset,
    }

    return mplPath(verts, codes), meta


# ---------------------------------------------------------------------------
# Random cubic spline segment (hand-drawn imitation)
# ---------------------------------------------------------------------------
def random_cubic_spline_segment(
        start     : PointXY,
        end       : PointXY,
        amp       : float      = 0.15,
        tightness : float      = 0.3,
        rng       : RNGBackend = None,
    ) -> mplPath:
    """Generate a cubic spline Path segment imitating a hand-drawn line.

    The function creates a 4-point cubic Bezier curve between two points
    with randomized perpendicular deviation to simulate "hand-drawn" jitter.

    Args:
        start: Starting point (x0, y0).
        end: Ending point (x1, y1).
        amp: Amplitude of perpendicular deviation (typ. 0.1-0.3).
        tightness: Controls curvature bias toward endpoints (typ. 0.2-0.5).
        rng: Optional RNG backend (RNG, random.Random, or np.random.Generator).

    Returns:
        mplPath: A cubic Bezier Path with one MOVETO and three CURVE4 vertices.
    """
    # --- RNG ----------------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    if not (isinstance(start, tuple) and isinstance(end, tuple)):
        raise TypeError(
            f"start and end must be tuple[float, float]. "
            f"Received {type(start)} and {type(end)}."
        )
    if len(start) != 2 or len(end) != 2:
        raise ValueError("start and end tuples must each have exactly two elements.")

    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0

    # Generate two normal deviations, clipped to [-1, 1]
    dev1 = normal3s() * amp
    dev2 = normal3s() * amp

    # Define control points with small perpendicular offsets
    P0 = (x0, y0)
    P1 = (x0 + dx * tightness - dy * dev1, y0 + dy * tightness + dx * dev1)
    P2 = (x1 - dx * tightness - dy * dev2, y1 - dy * tightness + dx * dev2)
    P3 = (x1, y1)

    verts = [P0, P1, P2, P3]
    codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
    return mplPath(verts, codes)


# ---------------------------------------------------------------------------
# Hand-drawn polyline using chained cubic splines
# ---------------------------------------------------------------------------
def handdrawn_polyline_path(
        points              : list[PointXY],
        splines_per_segment : int            = 5,
        amp                 : float          = 0.15,
        tightness           : float          = 0.3,
        rng                 : RNGBackend     = None,
    ) -> mplPath:
    """Generate a hand-drawn style polyline represented as a cubic Bezier chain.

    Each straight segment between consecutive points is subdivided into multiple
    short cubic spline sections with randomized curvature and jitter, producing
    a continuous, organic "hand-drawn" appearance.

    Each original line segment is divided into ``splines_per_segment`` equal-length
    subsections. For each subsection, an inner point is randomly *slid* along the
    parent segment by up to +/-0.4 x step length. This random shift changes the
    actual spacing between neighboring spline anchor points, which may range from
    roughly 0.2x to 1.8x the nominal step length - while always preserving the
    overall point order (no self-intersections due to inversion).

    Args:
        points: Sequence of (x, y) coordinates forming the base polyline.
        splines_per_segment: Number of spline subdivisions per straight segment.
        amp: Amplitude of random perpendicular deviation applied to each spline
            (controls "waviness").
        tightness: Fraction controlling how close control points are pulled toward
            endpoints (higher values yield tighter, less curved splines).
        rng: Optional RNG backend (``RNG``, ``random.Random``, or ``np.random.Generator``)
            used for reproducible randomization. If not provided, a thread-safe
            global RNG is used.

    Returns:
        mplPath: A continuous cubic Bezier chain path mimicking a hand-drawn polyline.
    """
    # --- Validation --------------------------------------------------------
    if not isinstance(points, (list, tuple)) or isinstance(points, (str, bytes)):
        raise TypeError("'points' must be an iterable of (x, y) pairs.")
    if len(points) < 2:
        raise ValueError("At least two points are required to form a polyline.")
    if not isinstance(splines_per_segment, int) or splines_per_segment < 1:
        raise ValueError("splines_per_segment must be a positive integer.")
    for name, val in {"amp": amp, "tightness": tightness}.items():
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}")

    JITTER_AMPLITUDE = 0.4 # Fraction of a step for sliding amplitude

    # --- RNG ----------------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True)
    normal3s = getattr(rng, "normal3s",
                   lambda: max(-1, min(1, rng.normalvariate(0, 1.0 / 3.0))))

    # --- Core setup --------------------------------------------------------
    P1, Pn = points[0], points[-1]
    closed = (
        (np.isnan(Pn[0]) and np.isnan(Pn[1])) or
        math.hypot(Pn[0] - P1[0], Pn[1] - P1[1]) < 1e-6
    )

    verts: list[PointXY] = [points[0]]

    # --- Loop through polyline segments ------------------------------------
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        xn, yn = end
        dx, dy = xn - x0, yn - y0
        stepx, stepy = dx / splines_per_segment, dy / splines_per_segment
        xp, yp = x0, y0

        # --- Loop through spline sections ----------------------------------
        for i in range(1, splines_per_segment + 1):
            slide = normal3s() * JITTER_AMPLITUDE
            xi = x0 + (i + slide) * stepx # end point of the current section
            yi = y0 + (i + slide) * stepy # end point of the current section
            dxs, dys = xi - xp, yi - yp

            dev1 = normal3s() * amp
            dev2 = normal3s() * amp

            P1 = (xp + dxs * tightness - dys * dev1, yp + dys * tightness + dxs * dev1)
            P2 = (xi - dxs * tightness - dys * dev2, yi - dys * tightness + dxs * dev2)
            P3 = (xi, yi)

            verts.extend([P1, P2, P3])
            xp, yp = xi, yi # set next section start (xp, yp) to curent section end (xi, yi)

        verts[-1] = end

    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * (len(verts) - 1)

    if closed:
        codes.append(mplPath.CLOSEPOLY)
        verts.append(points[0])

    meta: dict = {
        "operation" : "spline",
    }

    return mplPath(verts, codes), meta


# ---------------------------------------------------------------------------
# Demo: visualize hand-drawn polyline parameter effects
# ---------------------------------------------------------------------------
def demo_handdrawn_polyline_path(base_points: list[PointXY] = None,
                                 amps: list[float] = None,
                                 tightness_values: list[float] = None,
                                 seed: int = None) -> None:
    """Visual diagnostic demo for :func:`handdrawn_polyline_path`.

    Generates a grid of examples showing how amplitude and tightness affect
    the curvature and jitter of hand-drawn splines.

    Args:
        base_points: Optional sequence of control points defining the base polyline.
            Defaults to a 4-point zigzag pattern if None.
        amps: Sequence of amplitude values to visualize (controls "waviness").
        tightness_values: Sequence of tightness values to visualize.
        seed: Optional seed for deterministic reproducibility.

    Example:
        >>> from spt.mpl_path_utils import demo_handdrawn_polyline_path
        >>> demo_handdrawn_polyline_path()
    """
    # --- Defaults ----------------------------------------------------------
    if base_points is None:
        base_points = [(0, 0), (1, 0.2), (2, -0.3), (3, 0.1)]
    if amps is None:
        amps = [0.05, 0.1, 0.2, 0.3]
    if tightness_values is None:
        tightness_values = [0.1, 0.2, 0.4, 0.6]

    rng = get_rng(thread_safe=True)
    if seed is not None:
        rng.seed(seed)

    n_rows, n_cols = len(amps), len(tightness_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.0 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # --- Plot each combination --------------------------------------------
    for i, amp in enumerate(amps):
        for j, tight in enumerate(tightness_values):
            ax = axes[i, j]
            path = handdrawn_polyline_path(
                base_points, amp=amp, tightness=tight, rng=rng
            )
            patch = PathPatch(path, facecolor="none", lw=2.0, alpha=0.8)
            ax.add_patch(patch)
            ax.plot(*zip(*base_points), "--", lw=0.8, color="gray", alpha=0.5)
            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"amp={amp:.2f}, tight={tight:.2f}")

    plt.suptitle("Hand-Drawn Polyline Parameter Sweep", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Construct a Matplotlib Path composed of cubic Bezier segments with C1 continuity.
# ---------------------------------------------------------------------------
def bezier_from_xy_dy(
        x              : NDarray,
        y              : NDarray,
        dy             : NDarray  = None,
        tension        : float    = 0.0,
        endpoint_style : str      = "default",
    ) -> mplPath:
    """
    Construct a Matplotlib Path composed of cubic Bezier segments with C1 continuity.
    
    Each segment between consecutive (x, y) pairs is defined by four control points:
    P0, P1, P2, P3, where P1 and P2 determine tangent behavior. The default scaling
    (1/3 of the segment length) yields a standard cubic Bezier interpolant.
    
    The curve shape can be tuned using:
      - `tension`: adjusts derivative magnitude (smoothness)
      - `endpoint_style`: modifies tangent scaling at endpoints
    
    Args:
        x: 1D strictly increasing array of x-coordinates.
        y: 1D array of function values y(x).
        dy: Optional 1D array of derivatives y'(x).
            If None, estimated via finite differences.
        tension: Smoothness control in [0, 1].
            0 - fully smooth (Catmull-Rom-like)
            1 - polyline (zero curvature)
            Nonlinear scaling ( (1-tension)^2 ) is applied for gentler decay.
        endpoint_style:
            - 'default': uniform 1/3 tangent scaling (standard Bezier)
            - 'catmull': uniform 1/6 scaling (softer curvature)
            - 'relaxed': adaptive - 1/6 at endpoints, 1/3 inside
            - numeric value: custom uniform scaling factor (e.g., 0.25)
    
    Returns:
        matplotlib.path.Path:
            Path representing the continuous Bezier spline.
    
    Raises:
        ValueError: if fewer than two points are provided or x is not strictly increasing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least two data points.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    # --- Derivative estimation if not provided ---
    if dy is None:
        dy = np.empty_like(y)
        dx = np.diff(x)
        dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        dy[0] = (y[1] - y[0]) / dx[0]
        dy[-1] = (y[-1] - y[-2]) / dx[-1]
        dy *= (1.0 - tension) ** 2  # Nonlinear tension scaling

    # --- Segment geometry ---
    h = np.diff(x)
    x0, y0, dy0 = x[:-1], y[:-1], dy[:-1]
    x1, y1, dy1 = x[1:], y[1:], dy[1:]

    # --- Tangent scaling factor selection ---
    if isinstance(endpoint_style, (int, float)):
        scale = np.full_like(h, float(endpoint_style))
    elif endpoint_style == "catmull":
        scale = np.full_like(h, 1 / 6)
    elif endpoint_style == "relaxed":
        scale = np.full_like(h, 1 / 3)
        if len(scale) >= 2:
            scale[0] = scale[-1] = 1 / 6
    else:  # default
        scale = np.full_like(h, 1 / 3)

    # --- Compute control points ---
    B0 = np.column_stack([x0, y0])
    B1 = np.column_stack([x0 + scale * h, y0 + scale * h * dy0])
    B2 = np.column_stack([x1 - scale * h, y1 - scale * h * dy1])
    B3 = np.column_stack([x1, y1])

    # --- Assemble vertices and codes sequentially ---
    verts = [B0[0]]
    codes = [mplPath.MOVETO]
    for b1, b2, b3 in zip(B1, B2, B3):
        verts.extend([b1, b2, b3])
        codes.extend([mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4])

    verts = np.array(verts, dtype=float)
    codes = np.array(codes, dtype=np.uint8)
    return mplPath(verts, codes)


# ---------------------------------------------------------------------------
# Single segment Bezier approximation for an acute unit circular arc
# ---------------------------------------------------------------------------
def unit_circular_arc_segment(
        start_deg : float = 0.0,
        end_deg   : float = 90.0
    ) -> mplPath:
    """ Construct a cubic Bezier Path approximating a circular arc between two angles.

    The arc lies on the unit circle centered at (0, 0) and spans from `start_deg`
    to `end_deg` degrees, measured counter-clockwise. For spans larger than 90deg,
    multiple segments should be joined for accurate curvature.

    Args:
        start_deg: Start angle in degrees (0deg = +X axis).
        end_deg:   End angle in degrees.
    
    Returns:
        mplPath: Path consisting of four vertices (MOVETO + 3xCURVE4)
                 representing a single cubic Bezier arc segment.

    Raises:
        ValueError: If the absolute angular span exceeds 90deg.

    Notes:
        - The handle length `t = 4/3 * tan(theta / 4)` ensures tangent continuity.
        - Approximation error < 0.00027 R for theta = 90deg.
        - Suitable for composing smooth multi-segment arcs.
    """
    # --- Validate span -------------------------------------------------------
    span = abs(end_deg - start_deg)
    if span > 90.0 + 1e-9:
        raise ValueError(
            f"Span too large ({span:.2f}deg) for single cubic Bezier; "
            "split into <= 90deg segments."
        )

    # --- Core geometry -------------------------------------------------------
    start = np.radians(start_deg)
    end = np.radians(end_deg)
    delta = end - start
    t = 4.0 / 3.0 * np.tan(delta / 4.0)  # handle scaling

    # --- Compute control points ---------------------------------------------
    cos_s, sin_s = np.cos(start), np.sin(start)
    cos_e, sin_e = np.cos(end), np.sin(end)

    P0 = (cos_s, sin_s)
    P1 = (cos_s - t * sin_s, sin_s + t * cos_s)
    P2 = (cos_e + t * sin_e, sin_e - t * cos_e)
    P3 = (cos_e, sin_e)

    verts = np.array([P0, P1, P2, P3], dtype=float)
    codes = np.array(
        [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4],
        dtype=np.uint8,
    )
    return mplPath(verts, codes)


# ---------------------------------------------------------------------------
# Basic Ellipse / Arc path generator
# ---------------------------------------------------------------------------
def basic_ellipse_or_arc_path(
        x0           : float,
        y0           : float,
        r            : float,
        y_compress   : float = 1.0,
        start_angle  : float = 0.0,
        end_angle    : float = 360.0,
        angle_offset : float = 0.0,
        close        : bool  = False
    ) -> mplPath:
    """Create a basic Matplotlib Path representing a circle, ellipse, or elliptical arc.
  
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


# ---------------------------------------------------------------------------
# Ellipse / Arc
# ---------------------------------------------------------------------------
def elliptical_arc(
        canvas_x1x2        : CoordRange = (0, 1023),
        canvas_y1y2        : CoordRange = (0, 1023),
        start_deg          : float      = None,
        end_deg            : float      = None,
        y_compress         : float      = None,
        angle_deg          : int        = None,
        origin             : PointXY    = None,
        jitter_angle_deg   : int        = JITTER_ANGLE_DEG,
        jitter_amp         : float      = 0.04,
        jitter_y           : float      = 0.1,
        max_angle_step_deg : int        = 20,
        min_angle_steps    : int        = 3,
        rng                : RNGBackend = None,
    ) -> tuple[mplPath, dict]:
    """Creates a generalized elliptical arc or ellipse.

    The function first generates a unit circular arc using piecewise cubic
    Bezier curves. It then applies multi-stage transformations:

      1. Jittered arc generation (minor spatial & angular irregularities)
      2. Elliptical scaling (via y_compress)
      3. Random rotation and translation (via random_srt_path)

    Args:
        canvas_x1x2: Horizontal bounding range (pixels).
        canvas_y1y2: Vertical bounding range (pixels).
        start_deg, end_deg: Arc angles in degrees. Defaults to random values.
        y_compress: Optional y-axis compression (<1.0 -> ellipse).
        angle_deg: Optional global rotation angle.
        origin: Optional translation origin.
        jitter_angle_deg: Angular jitter magnitude (3sigma).
        jitter_amp: Additive positional jitter magnitude.
        jitter_y: Multiplicative jitter applied to y-axis only.
        max_angle_step_deg: Max angular span per cubic Bezier segment.
        min_angle_steps: Minimum number of Bezier segments.
        rng: Optional random number generator backend.

    Returns:
        tuple:
            mplPath: Final elliptical arc path.
            dict: Metadata with applied parameters and transformations.
    """
    # --- RNG setup ---------------------------------------------------------
    if rng is None:
        rng = get_rng(thread_safe=True, use_numpy=True)

    if not isinstance(rng._rng, np.random.Generator):
        raise TypeError(f"elliptical_arc requires np.random.Generator. Received '{type(rng._rng)}'")

    # --- Stage 1: unit circular arc ---------------------------------------
    shape, shape_meta = unit_circular_arc(
        start_deg=start_deg,
        end_deg=end_deg,
        jitter_amp=jitter_amp,
        jitter_y=jitter_y,
        max_angle_step_deg=max_angle_step_deg,
        min_angle_steps=min_angle_steps,
        rng=rng,
    )

    # --- Stage 2: apply SRT (scale / rotate / translate) -------------------
    shape, srt_meta = random_srt_path(
        shape=shape,
        canvas_x1x2=canvas_x1x2,
        canvas_y1y2=canvas_y1y2,
        y_compress=y_compress,
        angle_deg=angle_deg,
        origin=origin,
        jitter_angle_deg=jitter_angle_deg,
        rng=rng,
    )

    # --- Merge metadata ----------------------------------------------------
    """
    meta: dict = {
        "shape_meta": {
            "shape_kind": "circle",
            "start_deg" : start_deg,
            "end_deg"   : end_deg,
        },
        "srt_meta": {
            "scale_x": srt_meta["scale_x"],
            "scale_y": srt_meta["scale_y"],
            "rot_x"  : srt_meta["rot_x"],
            "rot_y"  : srt_meta["rot_y"],
            "rot_deg": srt_meta["rot_deg"],
            "trans_x": srt_meta["trans_x"],
            "trans_y": srt_meta["trans_y"],
        }
    }
    """
    meta = {
        "shape_meta"  : shape_meta,
        "srt_meta"    : srt_meta,
    }

    return shape, meta


# ---------------------------------------------------------------------------
# Line Segment
# ---------------------------------------------------------------------------
def line_segment(
        canvas_x1x2         : PointXY,
        canvas_y1y2         : PointXY,
        base_angle          : int        = None,
        jitter_angle_deg    : int        = JITTER_ANGLE_DEG,
        splines_per_segment : int        = 5,
        amp                 : float      = 0.3,
        tightness           : float      = 0.25,
        rng                 : RNGBackend = None,
    ) -> mplPath:
    """Creates a random line segment."""

    # --- Stage 1: unit line ------------------------------------------------
    shape, shape_meta = unit_circle_diameter(
            base_angle=base_angle,
            jitter_angle_deg=jitter_angle_deg,
            rng=rng,
    )
    
    # --- Stage 2: apply hand-drawn style -----------------------------------
    shape, spline_meta = handdrawn_polyline_path(
        points=list(shape.vertices),
        splines_per_segment=splines_per_segment,
        amp=amp,
        tightness=tightness,
        rng=rng,
    )
    
    # --- Stage 3: apply SRT (scale / rotate / translate) -------------------
    shape, srt_meta = random_srt_path(
        shape=shape,
        canvas_x1x2=canvas_x1x2,
        canvas_y1y2=canvas_y1y2,
        y_compress=None,
        angle_deg=base_angle,
        jitter_angle_deg=jitter_angle_deg,
        rng=rng,
    )

    meta = {
        "shape_meta"  : shape_meta,
        "spline_meta" : spline_meta,
        "srt_meta"    : srt_meta,
    }

    return shape, meta


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------
def rectangle(
        canvas_x1x2         : PointXY,
        canvas_y1y2         : PointXY,
        diagonal_angle      : numeric    = None,
        base_angle          : int        = None,
        origin              : PointXY    = None,
        jitter_angle_deg    : int        = JITTER_ANGLE_DEG,
        splines_per_segment : int        = 5,
        amp                 : float      = 0.3,
        tightness           : float      = 0.25,
        rng                 : RNGBackend = None,
    ) -> mplPath:
    """Creates a random rectangle.

    Composes three primitives:
    1. unit_rectangle_rnd() - geometric primitive generation
    2. random_srt_path()    - geometric transformation to canvas space
    3. polyline_path()      - stylization into hand-drawn form
    
    This function performs *no* parameter interpretation or mutation
    beyond connecting compatible interfaces between the components.
    """
    # --- Stage 1: unit rectangle ------------------------------------------
    shape, shape_meta = unit_rectangle_path(
        diagonal_angle=diagonal_angle,
        jitter_angle_deg=jitter_angle_deg,
        base_angle=base_angle,
        rng=rng,
    )

    # --- Stage 2: apply hand-drawn style -----------------------------------
    shape, spline_meta = handdrawn_polyline_path(
        points=list(shape.vertices),
        splines_per_segment=splines_per_segment,
        amp=amp,
        tightness=tightness,
        rng=rng,
    )
    
    # --- Stage 3: apply SRT (scale / rotate / translate) -------------------
    shape, srt_meta = random_srt_path(
        shape=shape,
        canvas_x1x2=canvas_x1x2,
        canvas_y1y2=canvas_y1y2,
        y_compress=None,
        angle_deg=base_angle,
        origin=origin,
        jitter_angle_deg=jitter_angle_deg,
        rng=rng,
    )

    meta = {
        "shape_meta"  : shape_meta,
        "spline_meta" : spline_meta,
        "srt_meta"    : srt_meta,
    }

    return shape, meta


def demo():
    rng = get_rng(thread_safe=True, use_numpy=True)
    
    canvas_x1x2=(-10, 30)
    canvas_y1y2=(-10, 20)
        
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlim(*canvas_x1x2)
    ax.set_ylim(*canvas_y1y2)

    line_shape, meta = line_segment(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2, base_angle=None, 
    )
    ax.add_patch(PathPatch(line_shape, edgecolor="green", lw=2, facecolor="none", linestyle="dashdot"))

    arc_shape, meta = elliptical_arc(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2,
        start_deg=None, end_deg=None, angle_deg=None, origin=(0, 0),
    )
    ax.add_patch(PathPatch(arc_shape, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))

    rect_shape, meta = rectangle(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2, diagonal_angle=None, base_angle=None, origin=None,
    )
    ax.add_patch(PathPatch(rect_shape, edgecolor="purple", lw=2, facecolor="none", linestyle="dashed"))
    
    """

    triangle = triangle_path(
        canvas_x1x2, canvas_y1y2, equal_sides = None, angle_category = None, base_angle = None
    )
    ax.add_patch(PathPatch(triangle, edgecolor="orange", lw=3, facecolor="none", linestyle="dotted"))


    x = np.linspace(-1, 1, 10)
    y = np.sin(np.pi * x)
    dy = np.cos(np.pi * x) * np.pi
    function1 = random_srt_path(
        bezier_from_xy_dy(x, y, dy=None, tension=0.25), canvas_x1x2, canvas_y1y2, None, None, (0, 0)
    )

    ax.add_patch(PathPatch(function1, edgecolor="gold", lw=2, facecolor="none", linestyle="solid"))
    function2 = random_srt_path(
        bezier_from_xy_dy(x, y, dy=dy, tension=0.25), canvas_x1x2, canvas_y1y2, None, None, (0, 0)
    )
    ax.add_patch(PathPatch(function2, edgecolor="violet", lw=2, facecolor="none", linestyle="solid"))
    
    x = np.linspace(0.1, 10, 100)
    y = 1 / x
    dy = -1 / (x * x)
    function3 = random_srt_path(
        join_paths([bezier_from_xy_dy(-x, -y, dy=None, tension=0.25), bezier_from_xy_dy(x, y, dy=None, tension=0.25)], True), 
        canvas_x1x2, canvas_y1y2, None, None, (0, 0)
    )
    ax.add_patch(PathPatch(function3, edgecolor="magenta", lw=2, facecolor="none", linestyle="solid"))
    """

    plt.show()


if __name__ == "__main__":
    demo()
