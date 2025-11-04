"""
path_patch_toolbox.py
"""

from __future__ import annotations

import numpy as np
import random
import math
from typing import Optional, Union
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
              spline_count: int = 5,
              amp: float = 0.15,
              tightness: float = 0.3,
              **kwargs) -> mplPath:
    """Creates piecewise splined segment imitating hand drawing.

    This function is designed to create a line segment between within.

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
            f"Both must be tuple[flaot, float]."
        )

    xmin, xmax = canvas_x1x2
    ymin, ymax = canvas_y1y2
    width, height = xmax - xmin, ymax - ymin
    
    if not isinstance(angle_deg, (int, float)):
        angle_deg = random.randint(-90, 90)
    angle_deg = ((angle_deg + 90) % 180) - 90
    if not isinstance(jitter_angle_deg, int): jitter_angle_deg = 5
    angle_deg += jitter_angle_deg * random.uniform(-1, 1)

    if abs(90 - abs(angle_deg)) < 0.1:
        x1 = x2 = xmin + 0.5 * width + random.uniform(-1,1) * 0.4 * width
        y1 = ymin + 0.25 * height * min(3, abs(random.normalvariate(0, 1)))
        y2 = ymax - 0.25 * height * min(3, abs(random.normalvariate(0, 1)))
        y2 = max(y2, y1 + 0.25 * height)
    elif abs(angle_deg) < 0.1:
        x1 = xmin + 0.25 * width * min(3, abs(random.normalvariate(0, 1)))
        x2 = xmax - 0.25 * width * min(3, abs(random.normalvariate(0, 1)))
        x2 = max(x2, x1 + 0.25 * width)
        y1 = y2 = ymin + 0.5 * height + random.uniform(-1,1) * 0.4 * height
    else:
        x1 = xmin + 0.25 * width * min(3, abs(random.normalvariate(0, 1)))
        if angle_deg > 45:
            y1 = ymin + 0.25 * height * min(3, abs(random.normalvariate(0, 1)))
        elif angle_deg < -45:
            y1 = ymax - 0.25 * height * min(3, abs(random.normalvariate(0, 1)))
        else:
            y1 = ymin + 0.5 * height + 0.25 * height / 3 * max(-3, min(3, random.normalvariate(0, 1)))
    
        slope = math.tan(math.radians(angle_deg))
        xmax_adj = min(xmax, x1 + (ymax if slope > 0 else ymin - y1) / slope)
        x2 = min(xmax, random.uniform(x1 + 1, xmax_adj))
        y2 = min(ymax, y1 + slope * (x2 - x1))
    
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

        dev1 = max(-1, min(random.normalvariate(0, 1) / 3, 1)) * amp
        dev2 = max(-1, min(random.normalvariate(0, 1) / 3, 1)) * amp
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
    

def unit_circular_arc(start_deg: numeric = 0,
                      end_deg: numeric = 90,
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


def unit_box_rand_srt(shape_path: mplPath,
                      canvas_x1x2: Optional[PointXY] = (0, 1023),
                      canvas_y1y2: Optional[PointXY] = (0, 1023),
                      compress_y: Optional[float] = None,
                      angle_deg: Optional[int] = None,
                      jitter_angle_deg: Optional[int] = 5,
                     ) -> mplPath:
    """ Performs a random Scale -> Rotate -> Translate of the unit box.

    Applies a random Scale-Rotate-Translate (SRT) affine transform to a path defined
    in a unit box ([-1, 1], hence, ubox_side=2 used for scaling). Note, alternatively,
    the assumption of unit box can be replaced with the bounding box of the path.

    Scale and translation are random uniform within the target canvas. Rotates by
    `angle_deg` (random, if not specified), If angle is specified, uniform
    +/- `jitter_angle_deg` is added. `compress_y` (randomized, if not specified) is
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

    if not isinstance(compress_y, (float, int)):
        compress_y = 1 - abs(random.normalvariate(0, 0.5))
    compress_y = max(0.25, min(compress_y, 1))

    x_scale = bbox_side / UNIT_BOX_SIDE
    y_scale = bbox_side / UNIT_BOX_SIDE * compress_y
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


def elliptical_arc(canvas_x1x2: tuple[float, float] = (0, 1023),
                   canvas_y1y2: tuple[float, float] = (0, 1023),
                   start_deg: Optional[float] = None,
                   end_deg: Optional[float] = None,
                   compress_y: Optional[float] = None,
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
        unit_arc_path, canvas_x1x2, canvas_y1y2, compress_y, angle_deg, jitter_angle_deg
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
    segment = line_path(
        canvas_x1x2=canvas_x1x2, canvas_y1y2=canvas_y1y2, angle_deg=None, jitter_angle_deg=0
    )

    ax.add_patch(PathPatch(arc, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))
    ax.add_patch(PathPatch(segment, edgecolor="green", lw=2, facecolor="none", linestyle="dashdot"))
    
    plt.show()


if __name__ == "__main__":
    demo()
