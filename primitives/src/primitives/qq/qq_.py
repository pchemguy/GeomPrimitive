import numpy as np
from matplotlib.path import Path

def cubic_arc_segment(start_deg=0, end_deg=90):
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
    P1 = [np.cos(a0) - t * np.sin(a0), np.sin(a0) + t * np.cos(a0)]
    P2 = [np.cos(a1) + t * np.sin(a1), np.sin(a1) - t * np.cos(a1)]

    verts = [P0, P1, P2, P3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

arc = cubic_arc_segment(-45, 0)  # any angle span <=90deg

fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(arc, edgecolor="red", lw=2, facecolor="none"))
ax.set_aspect("equal")
ax.grid(True, ls="--", alpha=0.5)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.show()
