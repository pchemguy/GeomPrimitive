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


def sampled_ellipse_arc(rx, ry, angle_deg, n=200):
    """Return points of a true elliptical 90deg arc by sampling."""
    t = np.linspace(0, np.pi / 2, n)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    trans = Affine2D().rotate_deg(angle_deg)
    return trans.transform(np.column_stack([x, y]))


# Parameters
rx, ry = 2.0, 1.0
angle = 30

# True sampled arc
sampled = sampled_ellipse_arc(rx, ry, angle)

# Cubic Bezier approximation
arc = cubic_arc_segment()
verts = Affine2D().scale(rx, ry).rotate_deg(angle).transform(arc.vertices)
bezier_arc = Path(verts, arc.codes)

# Plot comparison
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(sampled[:, 0], sampled[:, 1], color="blue", lw=1.5, label="True ellipse (sampled)")
ax.add_patch(PathPatch(bezier_arc, edgecolor="red", lw=2.0, facecolor="none", label="Cubic Bezier"))

ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.show()
