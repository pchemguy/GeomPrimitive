"""
grid_demo.py
------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def make_perspective_grid(
    W=200, H=150,
    dmajor=10, dminor=1,
    ax_rot=20, ay_rot=25,
    f=600,
    k1=0.0, k2=0.0,
):
  axr, ayr = np.radians(ax_rot), np.radians(ay_rot)

  # Rotation matrices
  Rx = np.array([[1, 0, 0],
                 [0, np.cos(axr), -np.sin(axr)],
                 [0, np.sin(axr),  np.cos(axr)]])
  Ry = np.array([[np.cos(ayr), 0, np.sin(ayr)],
                 [0, 1, 0],
                 [-np.sin(ayr), 0, np.cos(ayr)]])
  R = Ry @ Rx

  def project(X, Y):
    pts = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X).ravel()])
    Xr, Yr, Zr = R @ pts
    xp = f * Xr / (f - Zr)
    yp = f * Yr / (f - Zr)
    # radial distortion
    r2 = xp**2 + yp**2
    distortion = 1 + k1 * r2 + k2 * r2**2
    xp *= distortion
    yp *= distortion
    return xp.reshape(X.shape), yp.reshape(Y.shape)

  xs_major = np.arange(-W/2, W/2 + dmajor, dmajor)
  ys_major = np.arange(-H/2, H/2 + dmajor, dmajor)
  xs_minor = np.arange(-W/2, W/2 + dminor, dminor)
  ys_minor = np.arange(-H/2, H/2 + dminor, dminor)

  lines_major, lines_minor = [], []

  for x in xs_minor:
    X, Y = np.array([[x, x]]), np.array([[-H/2, H/2]])
    xp, yp = project(X, Y)
    lines_minor.append(np.column_stack([xp[0], yp[0]]))
  for y in ys_minor:
    X, Y = np.array([[-W/2, W/2]]), np.array([[y, y]])
    xp, yp = project(X, Y)
    lines_minor.append(np.column_stack([xp[0], yp[0]]))

  for x in xs_major:
    X, Y = np.array([[x, x]]), np.array([[-H/2, H/2]])
    xp, yp = project(X, Y)
    lines_major.append(np.column_stack([xp[0], yp[0]]))
  for y in ys_major:
    X, Y = np.array([[-W/2, W/2]]), np.array([[y, y]])
    xp, yp = project(X, Y)
    lines_major.append(np.column_stack([xp[0], yp[0]]))

  return lines_minor, lines_major


# --- Demo with barrel distortion ---
minor, major = make_perspective_grid(
    ax_rot=25, ay_rot=15, f=400,
    k1=-1e-6, k2=0
)

fig, ax = plt.subplots(figsize=(8, 6))

lc_minor = LineCollection(minor, colors="blue", linewidths=0.5, alpha=0.6)
lc_major = LineCollection(major, colors="red", linewidths=2.0,
                          linestyles="dashed", alpha=0.9)

ax.add_collection(lc_minor)
ax.add_collection(lc_major)

# Compute bounds dynamically
all_pts = np.concatenate(minor + major)
xmin, ymin = all_pts.min(axis=0)
xmax, ymax = all_pts.max(axis=0)
pad = 0.05 * max(xmax - xmin, ymax - ymin)
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
