import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D


def cubic_arc_segment():
    """Return a single 90deg circular arc as a cubic Bezier (unit radius)."""
    k = 4 / 3 * np.tan(np.pi / 8)
    verts = [(1, 0), (1, k), (k, 1), (0, 1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)



# Cubic Bezier approximation
arc = cubic_arc_segment()

# Plot comparison
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.add_patch(PathPatch(arc, edgecolor="red", lw=2.0, facecolor="none", label="Cubic Bezier"))

ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.show()
