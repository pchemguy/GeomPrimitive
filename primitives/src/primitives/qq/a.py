import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def cubic_arc_segment():
    """Return two 90deg circular arcs (0-90deg, 270-360deg) as cubic Bezier segments."""
    k = 4 / 3 * np.tan(np.pi / 8)

    # First arc: 0deg - 90deg (top-right)
    v1 = [
        (1, 0),        # start
        (1, k),        # control 1
        (k, 1),        # control 2
        (0, 1),        # end
    ]
    c1 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    # Second arc: 270deg - 360deg (bottom-right)
    # Start at (0,-1), end at (1,0)
    v2 = [
        (0, -1),
        (k, -1),
        (1, -k),
        (1, 0),
    ]
    c2 = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    verts = v1 + v2
    codes = c1 + c2
    return Path(verts, codes)


# ---- Demo plot ----
arc = cubic_arc_segment()

fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(arc, edgecolor="red", lw=2.0, facecolor="none", label="Cubic Bezier arcs"))

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.show()
