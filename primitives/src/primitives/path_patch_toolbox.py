"""
path_patch_toolbox.py
---------------------

https://chatgpt.com/c/69025b9d-b044-8333-8159-aac740d7bf70
https://chatgpt.com/c/6905add2-1ff8-8328-ba21-6c370614dcc4
"""

from __future__ import annotations

import numpy as np
import random
import math
from enum import Enum, auto
from typing import Optional, Union
from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath
from matplotlib.transforms import Affine2D

numeric = Union[int, float]
PointXY = tuple[numeric, numeric]


def print_locals(local_vars: dict) -> None:
    """
    Pretty-prints the top level of a dict with sorted keys.
    Collapses non-scalar values to a single-line string.
    """
    max_key_len = max(len(k) for k in local_vars) + 1 # +1 for a space
    scalar_types = (int, float, str, bool, type(None))

    for key, value in sorted(local_vars.items()):        
        if key[0] == "_":
            pass
        elif isinstance(value, scalar_types):
            print(f"{key:<{max_key_len}}: {value}")
        else:
            value_str = str(value).replace('\n', ' ')
            print(f"{key:<{max_key_len}}: {value_str}")


def join_paths(paths: list[mplPath]) -> mplPath:
    """
    Joins a list of paths into a single continuous path.
    Assumes the end of path N is the start of path N+1.
    """
    if not paths:
        raise ValueError(f"Expected a list of Matplotlib paths.")

    for path in paths:
        if not isinstance(path, mplPath):
            raise TypeError(f"Expected a list of Matplotlib paths. Received {type(path).__name__}.")

    # Start with the first path's vertices and codes
    all_verts = [paths[0].vertices]
    all_codes = [paths[0].codes]

    # Iterate over the rest of the paths
    for path in paths[1:]:
        if path.vertices.size > 0:
            # Get vertices and codes, skipping the first (MOVETO)
            all_verts.append(path.vertices[1:])
            all_codes.append(path.codes[1:])

    # Concatenate all at once at the end (more efficient)
    final_verts = np.concatenate(all_verts)
    final_codes = np.concatenate(all_codes)
    
    return mplPath(final_verts, final_codes)


def random_srt_path(shape: mplPath,
                    canvas_x1x2: PointXY,
                    canvas_y1y2: PointXY,
                    y_compress: float = None,
                    angle_deg: numeric = None,
                    origin: PointXY = None
                   ) -> mplPath:
    if not isinstance(shape, mplPath):
        raise TypeError(f"Expected a Matplotlib path. Received {type(shape).__name__}.")
    if not isinstance(canvas_x1x2, tuple) or not isinstance(canvas_y1y2, tuple):
        raise TypeError(
            f"canvas_x1x2: {type(canvas_x1x2).__name__}\n"
            f"canvas_y1y2: {type(canvas_y1y2).__name__}\n"
            f"Both must be tuple[float, float]."
        )
    if origin and not isinstance(origin, tuple):
        raise TypeError(
            f"If provided, origin should be tuple[float, float].\n"
            f"Received: {type(origin).__name__}"
        )
    if not isinstance(y_compress, (int, float)):
        y_compress = random.uniform(0.5, 1.0)
    y_compress = max(0.2, float(y_compress))

    JITTER_ANGLE_DEG = 5
    cxmin, cxmax = canvas_x1x2
    cymin, cymax = canvas_y1y2
    cx0, cy0 = (cxmax + cxmin) / 2, (cymax + cymin) / 2
    cw, ch = cxmax - cxmin, cymax - cymin

    if not isinstance(angle_deg, (int, float)):
        angle_deg = random.randrange(360)
    elif angle_deg != 0:
        angle_deg = round(angle_deg) + JITTER_ANGLE_DEG * max(-3, min(3, random.normalvariate(0, 1))) / 3
    else:
        pass
    angle_deg = ((angle_deg + 180) % 360) - 180

    angle_rad = math.radians(angle_deg)
    bbox = shape.get_extents()
    bxmin, bymin, bxmax, bymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    bx0, by0 = (bxmax + bxmin) / 2, (bymax + bymin) / 2
    bw, bh = bbox.width, bbox.height
    bwsize = max(1e-6, abs(bw * math.cos(angle_rad)) + abs(bh * math.sin(angle_rad)))
    bhsize = max(1e-6, abs(bh * math.cos(angle_rad)) + abs(bw * math.sin(angle_rad)))

    sf = random.uniform(0.2, 1) * min(cw / bwsize, ch / bhsize)

    tx_range = (cw - bwsize * sf) / 2
    ty_range = (ch - bhsize * sf * y_compress) / 2

    tx = cx0 - bx0 * sf + tx_range * random.uniform(-1, 1)
    ty = cy0 - by0 * sf * y_compress + ty_range * random.uniform(-1, 1)    

    if not origin:
        origin = (bx0, by0)
    trans = (
        Affine2D()
        .scale(sf, sf * y_compress)
        .rotate_around(*origin, angle_rad)
        .translate(tx, ty)
    )
    verts_array = trans.transform(shape.vertices)
    
    return mplPath(verts_array, shape.codes)


def unit_box_rand_srt(shape_path: mplPath,
                      canvas_x1x2: Optional[PointXY] = (0, 1023),
                      canvas_y1y2: Optional[PointXY] = (0, 1023),
                      y_compress: Optional[float] = None,
                      angle_deg: Optional[int] = None,
                      jitter_angle_deg: Optional[int] = 5,
                     ) -> mplPath:
    """ Performs a random Scale -> Rotate -> Translate of the unit box.

    Applies a random Scale-Rotate-Translate (SRT) affine transform to a path defined
    in a unit box ([-1, 1], hence, ubox_side=2 used for scaling). Note, alternatively,
    the assumption of unit box can be replaced with the bounding box of the path.

    Scale and translation are random uniform within the target canvas. Rotates by
    `angle_deg` (random, if not specified), If angle is specified, uniform
    +/- `jitter_angle_deg` is added. `y_compress` (randomized, if not specified) is
    used to compress y coordinates, e.g., transforming unit circles and squares into
    ellipses and rectangles.
    """
    # -------------------------
    # Scale params.
    # -------------------------
    UNIT_BOX_SIDE = 2
    xmin, xmax = canvas_x1x2
    ymin, ymax = canvas_y1y2
    width, height = xmax - xmin, ymax - ymin
    print(f"width: {width}\nheight: {height}")
    scale_factor_range: tuple[float, float] = (0.2, 0.9)
    bbox_side = min(width, height) * random.uniform(*scale_factor_range)
    bbox_diag = bbox_side * math.sqrt(2)
    print(f"bbox_side: {bbox_side}\nbbox_diag: {bbox_diag}")

    if not isinstance(y_compress, (float, int)):
        y_compress = 1 - abs(random.normalvariate(0, 0.5))
    y_compress = max(0.25, min(y_compress, 1))

    x_scale = bbox_side / UNIT_BOX_SIDE
    y_scale = bbox_side / UNIT_BOX_SIDE * y_compress
    print(f"===== SCALE =====")
    print(f"[x_scale, y_scale]: {[x_scale, y_scale]}.")

    # -------------------------
    # Rotate params.
    # -------------------------
    if not isinstance(angle_deg, (int, float)):
        angle_deg = random.uniform(-90, 90)
    else:
        angle_deg += jitter_angle_deg * random.uniform(-1, 1)
    print(f"===== ROTATE =====")
    print(f"angle_deg: {angle_deg}.")
    
    # -------------------------
    # Translate params.
    # -------------------------
    x0, y0 = (xmin + xmax) / 2, (ymin + ymax) / 2
    x_translate = x0 + max(0, 0.5 * (width - bbox_diag)) * random.uniform(-1, 1)
    y_translate = y0 + max(0, 0.5 * (height - bbox_diag)) * random.uniform(-1, 1)
    print(f"===== TRANSLATE =====")
    print(f"[x_translate, y_translate]: {[x_translate, y_translate]}.")

    # -------------------------
    # Apply SRT (Scale, Rotate, Translate)
    # -------------------------
    trans = (
        Affine2D()
        .scale(x_scale, y_scale)
        .rotate_deg(angle_deg)
        .translate(x_translate, y_translate)
    )
    verts = trans.transform(shape_path.vertices)
    
    # -------------------------
    # 9. Create Path
    # -------------------------
    trans_path: mplPath = mplPath(verts, shape_path.codes)

    return trans_path


def unit_circular_arc_segment(start_deg: numeric = 0, end_deg: numeric = 90) -> mplPath:
    """Circular Arc - Utility and reference implementation. Not currently used.

    Creates a cubic Bezier Path approximating a circular arc between
    start_deg and end_deg (degrees). Works best for spans <= 90deg.
    """
    if abs(end_deg - start_deg) > 90:
        raise ValueError(
            "Span too large for single cubic Bezier; split into <=90deg segments.")

    start, end = np.radians(start_deg), np.radians(end_deg)
    delta = end - start

    # 4/3 * tan(delta/4) gives the correct control handle length
    t = 4 / 3 * np.tan(delta / 4)

    # Start, end, and control points
    P0 = (np.cos(start), np.sin(start))
    P3 = [np.cos(end), np.sin(end)]
    P1 = [np.cos(start) - t * np.sin(start), np.sin(start) + t * np.cos(start)]
    P2 = [np.cos(end) + t * np.sin(end), np.sin(end) - t * np.cos(end)]

    verts = [P0, P1, P2, P3]
    codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
    return mplPath(verts, codes)


def random_cubic_spline_segment(start: PointXY, end: PointXY,
                                amp: float = 0.15, tightness: float = 0.3) -> mplPath:
    """Cubic Spline Segment - Utility and reference implementation. Not currently used.

    Creates a single cubic spline section imitating a hand-drawn segment.

    """
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    
    dev1 = max(-1, min(random.normalvariate(0, 1 / 3), 1)) * amp
    dev2 = max(-1, min(random.normalvariate(0, 1 / 3), 1)) * amp

    P0 = (x0, y0)
    P1 = (x0 + dx * tightness - dy * dev1, y0 + dy * tightness + dx * dev1)
    P2 = (x1 - dx * tightness - dy * dev2, y1 - dy * tightness + dx * dev2)
    P3 = (x1, y1)

    verts = [P0, P1, P2, P3]
    codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
    return mplPath(verts, codes)


def line_path(canvas_x1x2: PointXY,
              canvas_y1y2: PointXY,
              angle_deg: (Optional[float]) = None,
              jitter_angle_deg: Optional[int] = 5,
              spline_count: Optional[int] = 5,
              amp: Optional[float] = 0.15,
              tightness: Optional[float] = 0.3,
             ) -> mplPath:
    """Creates a hand-drawn-style line segment centered within the target canvas.

    Provied or randomized segment is split into `spline_count` sections using
    `spline_count -1` points. All points are on the source segment, but are randomly
    shifted along the line by JITTER_FACTOR of the step with respect to the equal
    split position. So long as JITTER_FACTOR < 0.5, the maximum distance reduaction
    of adjacent points is less than full step, ensuring a well behaving split.
    """
    if not isinstance(canvas_x1x2, tuple) or not isinstance(canvas_y1y2, tuple):
        raise TypeError(
            f"canvas_x1x2: {type(canvas_x1x2).__name__}\n"
            f"canvas_y1y2: {type(canvas_y1y2).__name__}\n"
            f"Both must be tuple[float, float]."
        )

    xmin, xmax = canvas_x1x2
    ymin, ymax = canvas_y1y2
    canvas_xcenter, canvas_ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2
    canvas_width, canvas_height = xmax - xmin, ymax - ymin
    canvas_size = min(canvas_width, canvas_height)
    UNIT_BOX_SIZE = 2 # Side length of canonical coordinate frame (-1..+1)
    
    if not isinstance(angle_deg, (int, float)):
        angle_deg = random.randint(-90, 90)
    angle_deg = ((angle_deg + 90) % 180) - 90
    if not isinstance(jitter_angle_deg, int): jitter_angle_deg = 5
    angle_deg += jitter_angle_deg * max(-3, min(3, random.normalvariate(0, 1))) / 3
    angle_rad = math.radians(angle_deg)


    sf = random.uniform(0.2, 1) * canvas_size / UNIT_BOX_SIZE

    x1, y1 = math.cos(angle_rad) * sf, math.sin(angle_rad) * sf
    x2, y2 = -x1, -y1

    bbox_width  = abs(x2 - x1)
    bbox_height = abs(y2 - y1)

    tx_range = (canvas_width  - bbox_width) / 2
    ty_range = (canvas_height - bbox_height) / 2

    tx = canvas_xcenter + tx_range * random.uniform(-1, 1)
    ty = canvas_ycenter + ty_range * random.uniform(-1, 1)    

    x1, x2 = x1 + tx, x2 + tx
    y1, y2 = y1 + ty, y2 + ty

    start = (x1, y1)
    end = (x2, y2)
    
    x0, y0 = start
    xn, yn = end
    dx, dy = xn - x0, yn - y0
    stepx, stepy = dx / spline_count, dy / spline_count
    
    JITTER_FACTOR = 0.4
    xp, yp = start
    verts: list[PointXY] = [start]
    for i in range(1, spline_count + 1):
        slide = random.uniform(-1, 1) * JITTER_FACTOR
        xi, yi = x0 + (i + slide) * stepx, y0 + (i + slide) * stepy
        dx, dy = xi - xp, yi - yp

        dev1 = max(-1, min(random.normalvariate(0, 1 / 3), 1)) * amp
        dev2 = max(-1, min(random.normalvariate(0, 1 / 3), 1)) * amp
        P1 = (xp + dx * tightness - dy * dev1, yp + dy * tightness + dx * dev1)
        P2 = (xi - dx * tightness - dy * dev2, yi - dy * tightness + dx * dev2)
        P3 = (xi, yi)

        verts.extend([P1, P2, P3])        
        xp, yp = xi, yi

    verts[-1] = end
    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * spline_count
    
    print(f"~~~~~~~~~~LINE PATH~~~~~~~~~~~~~ line_path() verts: \n {verts} \n codes: \n {codes}")
    print_locals(locals())
    return mplPath(verts, codes)


def polyline_path(points: list[PointXY],
                  spline_count: Optional[int] = 5,
                  amp: Optional[float] = 0.15,
                  tightness: Optional[float] = 0.3,
                 ) -> mplPath:
    """Creates a wavy polyline."""

    if not isinstance(points, Iterable) or isinstance(points, (str, bytes)):
        raise TypeError(
            f"'points' must be an iterable of (x, y) pairs, got {type(points).__name__}"
        )
    if len(points) < 2:
        raise ValueError(
            f"At least two points are required to form a polyline: got {len(points)}."
        )
    if not isinstance(spline_count, int) or spline_count < 1:
        spline_count = 5
    for name, val in {"amp": amp, "tightness": tightness}.items():
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}")

    P1 = points[0]
    Pn = points[-1]
    closed = math.hypot(Pn[0]-P1[0], Pn[1]-P1[1]) < 1e-6

    end = points[0]
    verts: list[PointXY] = [points[0]]
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        xn, yn = end
        dx, dy = xn - x0, yn - y0
        stepx, stepy = dx / spline_count, dy / spline_count
        
        JITTER_FACTOR = 0.4
        xp, yp = start
        for i in range(1, spline_count + 1):
            slide = random.uniform(-1, 1) * JITTER_FACTOR
            xi, yi = x0 + (i + slide) * stepx, y0 + (i + slide) * stepy
            dx, dy = xi - xp, yi - yp
    
            dev1 = max(-1, min(1, random.normalvariate(0, 1 / 3))) * amp
            dev2 = max(-1, min(1, random.normalvariate(0, 1 / 3))) * amp

            P1 = (
                xp + dx * tightness - dy * dev1, 
                yp + dy * tightness + dx * dev1
            )
            P2 = (
                xi - dx * tightness - dy * dev2,
                yi - dy * tightness + dx * dev2
            )
            P3 = (xi, yi)
    
            verts.extend([P1, P2, P3])        
            xp, yp = xi, yi
    
        verts[-1] = end
        
    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * spline_count * (len(points) - 1)

    if closed:
        codes.append(mplPath.CLOSEPOLY)
        verts.append(points[0])

    return mplPath(verts, codes)


def unit_triangle_rnd(equal_sides: int = None,
                      angle_category: int = None,
                      jitter_angle_deg: int = 5,
                      base_angle: int = None,
                     ) -> mplPath:
    """Generates vertices of triangle inscribed into a unit circle.

    Arguments:
        "equal_sides":      1, 2, or 3.
        "angle_category":   This value is compared with 90 to determine
                            requested triangle (actual value is not used):
                            <90 - ACUTE
                            =90 - RIGHT
                            >90 - OBTUSE
    """
    if not equal_sides:
        equal_sides = random.choice((1, 2, 3))
    if not equal_sides in (1, 2, 3):
        raise ValueError(
            f"equal_sides must be an integer in [1, 3].\n"
            f"Received type: {type(equal_sides).__name__}; value: {equal_sides}."
        )
    if not angle_category:
        angle_category = random.choice((60, 90, 120))
    if not isinstance(angle_category, (int, float)):
        raise TypeError(
            f"angle_category must be ot type integer or float.\n"
            f"Received type: {type(angle_category).__name__}; value: {angle_category}."
        )

    if equal_sides == 3:
        thetas = [90, -30, 210]
    else:
        top_offset = (
            0 if equal_sides > 1 else random.choice([-1, 1]) *
            random.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        base_offset = (
            ((angle_category > 90) - (angle_category < 90)) *
            random.uniform(jitter_angle_deg, 90 - jitter_angle_deg)
        )
        thetas = [90 + top_offset, 0 + base_offset, 180 - base_offset]

    if not isinstance(base_angle, (int, float)):
        base_angle = random.uniform(-90, 90)
    else:
        base_angle += jitter_angle_deg / 3 * max(-3, min(3, random.normalvariate(0, 1)))
    top_jitter = jitter_angle_deg / 3 * max(-3, min(3, random.normalvariate(0, 1)))
    thetas[0] += top_jitter
    thetas = [math.radians(theta + base_angle) for theta in thetas]
    verts = [(math.cos(theta_rad), math.sin(theta_rad)) for theta_rad in thetas]
    verts.append(verts[0])
    codes = [mplPath.MOVETO, mplPath.LINETO, mplPath.LINETO, mplPath.CLOSEPOLY]

    return mplPath(verts, codes)


def triangle_path(canvas_x1x2: PointXY,
             canvas_y1y2: PointXY,
             equal_sides: int = None,
             angle_category: int = None,
             base_angle: int = None,
             jitter_angle_deg: int = 5,
             spline_count: int = 5,
             amp: float = 0.15,
             tightness: float = 0.3,
            ) -> mplPath:
    """Creates a random triangle.

    Composes three primitives:
    1. unit_triangle_rnd() - geometric primitive generation
    2. random_srt_path()   - geometric transformation to canvas space
    3. polyline_path()     - stylization into hand-drawn form
    
    This function performs *no* parameter interpretation or mutation
    beyond connecting compatible interfaces between the components.
    """
    # Creates unit triangle

    unit_shape: mplPath = unit_triangle_rnd(
        equal_sides, angle_category, jitter_angle_deg, base_angle
    )

    # Transforms (random SRT) unit triangle to canvas
    
    shape_srt: mplPath = random_srt_path(
        unit_shape, canvas_x1x2, canvas_y1y2, None, 0, (0, 0)
    )

    # Creates hand-drawn style

    shape_handdrawn = polyline_path(list(shape_srt.vertices), spline_count, amp, tightness)
    
    return shape_handdrawn


def unit_circular_arc(start_deg: Optional[numeric] = 0,
                      end_deg: Optional[numeric] = 90,
                      jitter_amp: Optional[float] = 0.02,
                      jitter_y: Optional[float] = 0.1,
                      max_angle_step_deg: Optional[int] = 20,
                      min_angle_steps: Optional[int] = 3,
                     ) -> mplPath:
    """ Creates a unit circular arc.

    Uses piecewise cubic Bezier curves provided by Matplotlib to approximate a unit 
    circular arc. In principle, a 90 deg arc can be approximated by a single cubic
    curve very well. However, to imitate hand drawing, smaller steps are used, 
    set at the default value of 20 deg `max_angle_step_deg`. For the same reason,
    the smallest number of sections is also set (`min_angle_steps`).
    """
    if start_deg is None or end_deg is None:
        start_deg = random.uniform(0, 270)
        end_deg = random.uniform(start_deg + 5, 360)
    span_deg: float = end_deg - start_deg
    if span_deg < 1 or span_deg > 359:
        start_deg = 0
        end_deg = 360
        span_deg = 360
        closed = True
    else:
        closed = False

    theta_steps: int = int(max(min_angle_steps, round(span_deg / max_angle_step_deg)))
    start, end = np.radians(start_deg), np.radians(end_deg)
    span: float = end - start
    step_theta: float = span / theta_steps
    t = 4 / 3 * np.tan(step_theta / 4)

    P0: PointXY = (float(np.cos(start)), float(np.sin(start)))
    verts: list[PointXY] = [P0]
    theta_beg: float = start
    theta_end: float = start + step_theta
    for i in range(theta_steps):
        P1: PointXY = [float(np.cos(theta_beg) - t * np.sin(theta_beg)),
                       float(np.sin(theta_beg) + t * np.cos(theta_beg))]
        P2: PointXY = [float(np.cos(theta_end) + t * np.sin(theta_end)),
                       float(np.sin(theta_end) - t * np.cos(theta_end))]
        P3: PointXY = [float(np.cos(theta_end)),
              float(np.sin(theta_end))]
        verts.extend([P1, P2, P3])
        theta_beg += step_theta
        theta_end += step_theta
    
    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * theta_steps
    if closed:
        codes.append(mplPath.CLOSEPOLY)
        verts.append(P0)

    verts_ndarray: np.ndarray = np.array(verts)

    # -------------------------
    # Apply Y Jitter (Multiplicative)
    # -------------------------
    if jitter_y:
        # We apply one random scale to the y-axis of all vertices
        verts_ndarray[:, 1] *= 1 - np.random.uniform(0, 1) * jitter_y # Scales the entire y-column

    # -------------------------
    # Apply Additive Jitter
    # -------------------------
    if jitter_amp:
        verts_ndarray += np.random.uniform(-1, 1, size=verts_ndarray.shape) * jitter_amp

    # Replace the last point with the first point
    if closed:
        verts_ndarray[-1] = verts_ndarray[0]

    return mplPath(verts_ndarray, codes)


def elliptical_arc(canvas_x1x2: tuple[float, float] = (0, 1023),
                   canvas_y1y2: tuple[float, float] = (0, 1023),
                   start_deg: Optional[float] = None,
                   end_deg: Optional[float] = None,
                   y_compress: Optional[float] = None,
                   angle_deg: Optional[int] = None,
                   jitter_angle_deg: Optional[int] = 5,
                   jitter_amp: Optional[float] = 0.02,
                   jitter_y: Optional[float] = 0.1,
                   max_angle_step_deg: Optional[int] = 20,
                   min_angle_steps: Optional[int] = 3,
                  ) -> mplPath:
    """ Creates a generalized elliptical arc or an ellipse.

    The code first creates a unit circular arc using piecewise cubic Bezier
    curves provided by Matplotlib. In principle, a 90 deg arc can be approximated
    by a single cubic curve very well. However, to imitate hand drawing, smaller
    steps are used, set at the default value of 20 deg `max_angle_step_deg`.

    For smaller arcs, the smallest number of sections is set to 3 (`min_angle_steps`).

    Once the unit arc or a full circle is created, Jitter is applied to individual
    points (magnitude controlled by `jitter_amp`), as well as to aspect ratio
    (`jitter_y` controls scaling of the y coordinates only. The latter is only
    useful for creating non-ideal circular arcs, as elliptical transform will absorb
    this factor.

    The circular arc is scaled to yield an ellipse, rotated (with angle jitter), and
    translated to yield the final generalized elliptical arc with jitter.
    """
    # Create a unit circular arc using piecewise cubic Bezier curves.
    #
    unit_arc_path: mplPath = unit_circular_arc(
        start_deg, end_deg, jitter_amp, jitter_y, max_angle_step_deg, min_angle_steps
    )
    
    arc_path: mplPath = unit_box_rand_srt(
        unit_arc_path, canvas_x1x2, canvas_y1y2, y_compress, angle_deg, jitter_angle_deg
    )

    return arc_path


def demo():
    canvas_x1x2=(-10, 30)
    canvas_y1y2=(-10, 20)
        
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlim(*canvas_x1x2)
    ax.set_ylim(*canvas_y1y2)

    arc = elliptical_arc(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2,
        start_deg=0, end_deg=360, angle_deg=None
    )
    ax.add_patch(PathPatch(arc, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))

    segment = line_path(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2, angle_deg=0, jitter_angle_deg=5
    )
    ax.add_patch(PathPatch(segment, edgecolor="green", lw=2, facecolor="none", linestyle="dashdot"))

    polyline = polyline_path([(-5,5), (5,-5), (15,5), (-5,5)])
    ax.add_patch(PathPatch(polyline, edgecolor="brown", lw=5, facecolor="none", linestyle="dotted"))

    triangle = triangle_path(
        canvas_x1x2, canvas_y1y2, equal_sides = None, angle_category = None, base_angle = None
    )
    ax.add_patch(PathPatch(triangle, edgecolor="orange", lw=3, facecolor="none", linestyle="dotted"))

    
    plt.show()


if __name__ == "__main__":
    demo()
