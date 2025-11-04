import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def full_circle_path():
    """Return a full circle using 4 cubic Bezier arcs (each 90deg)."""
    k = 4 / 3 * np.tan(np.pi / 8)

    verts = [
        (1, 0), (1, k), (k, 1), (0, 1),      # 0deg-90deg
        (-k, 1), (-1, k), (-1, 0),           # 90deg-180deg
        (-1, -k), (-k, -1), (0, -1),         # 180deg-270deg
        (k, -1), (1, -k), (1, 0)             # 270deg-360deg
    ]

    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4
    ]

    return Path(verts, codes)


# ---- Demo plot ----
circle_path = full_circle_path()

fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(circle_path, edgecolor="red", lw=2.0, facecolor="none", label="Full circle (4 Beziers)"))

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.show()
