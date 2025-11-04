import numpy as np
import random
import math
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath


def elliptical_arc(hrange: tuple[float, float] = (0, 1023),
                   vrange: tuple[float, float] = (0, 1023),
                   start_deg: Optional[float] = None,
                   end_deg: Optional[float] = None,
                   aspect_ratio: Optional[float] = None,
                   jitter: Optional[float] =0.02,
                   max_angle_delta_deg: Optional[int] = 20,
                   min_angle_steps: Optional[int] = 3,
                  ) - > mplPath:
    xmin, xmax = hrange
    ymin, ymax = vrange
    if start_deg is None:
        start_deg = random.uniform(0, 270)
        end_deg = random.uniform(end_deg + 5, 360)
    span_deg: float = end_deg - start_deg
    if span_deg < 1 or span_span > 359:
        start_deg = 0
        end_deg = 360
        closed = True

    step_count: float = min(min_angle_steps, span / max_angle_delta_deg)
    start: float = np.radians(start_deg)
    end: float = np.radians(end_deg)
    span: float = end - start
    delta: float = span / step_count

    verts = []
    codes = []
    start_section = start
    end_section = start_section + delta
    for i in range(step_count):
        start_section += delta
        end_section += delta



def cubic_arc_segment(start_deg=0, end_deg=90, amp=0.02):
    """
    Return a cubic Bezier Path approximating a circular arc
    between start_deg and end_deg (degrees). Works best for spans <= 90deg.
    """
    a0, a1 = np.radians(start_deg), np.radians(end_deg)
    delta = a1 - a0
    if abs(delta) > np.pi / 2:
        raise ValueError("Span too large for single cubic Bezier; split into <=90deg segments.")

    # 4/3 * tan(delta/4) gives the correct control handle length
    t = 4 / 3 * np.tan(delta / 4)

    # Start, end, and control points
    P0 = [np.cos(a0), np.sin(a0)]
    P3 = [np.cos(a1), np.sin(a1)]
    P1 = [np.cos(a0) - t * np.sin(a0) + random.uniform(-1, 1) * amp, np.sin(a0) + t * np.cos(a0) + random.uniform(-1, 1) * amp]
    P2 = [np.cos(a1) + t * np.sin(a1) + random.uniform(-1, 1) * amp, np.sin(a1) - t * np.cos(a1) + random.uniform(-1, 1) * amp]

    verts = [P0, P1, P2, P3]
    codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
    return mplPath(verts, codes)


def join_paths(path1: mplPath, path2: mplPath) -> mplPath:
    """
    Joins two paths into a single continuous path.
    Assumes the end of path1 is the start of path2.
    """
    # If path1 is empty, just return path2
    if not path1.vertices.size:
        return path2
    # If path2 is empty, just return path1
    if not path2.vertices.size:
        return path1

    # Get vertices and codes, skipping the first (MOVETO) from path2
    v1, c1 = path1.vertices, path1.codes
    v2, c2 = path2.vertices[1:], path2.codes[1:]

    # Concatenate them
    verts = np.concatenate((v1, v2))
    codes = np.concatenate((c1, c2))
    
    return mplPath(verts, codes)



arc1 = cubic_arc_segment(0, 30, 0.04)  # any angle span <=90deg
arc2 = cubic_arc_segment(30, 60, 0.04)
arc3 = cubic_arc_segment(60, 90, 0.04)
arc4 = cubic_arc_segment(90, 120, 0.04)
arc5 = cubic_arc_segment(120, 150, 0.04)
arc6 = cubic_arc_segment(150, 180, 0.04)



arc = join_paths(join_paths(join_paths(arc1, arc2), join_paths(arc3, arc4)), join_paths(arc5, arc6))

print(arc)


fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(arc, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))
ax.set_aspect("equal")
ax.grid(True, ls="--", alpha=0.5)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(0, 1.2)
plt.show()
