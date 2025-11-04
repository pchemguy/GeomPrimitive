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



# Parameters
rx, ry = 2.0, 1.0
angle = 30


# Cubic Bezier approximation
arc = cubic_arc_segment()
verts = Affine2D().scale(rx, ry).rotate_deg(angle).transform(arc.vertices)
bezier_arc = Path(verts, arc.codes)

# Plot comparison
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 2)
ax.set_ylim(0.5, 1.5)

ax.add_patch(PathPatch(bezier_arc, edgecolor="red", lw=2.0, facecolor="none", label="Cubic Bezier"))

ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.show()
