"""
grid_demo_bezier.py
-------------------
Perspective + lens-distorted grid drawn with cubic Bezier curves
for smooth curvature without subdividing lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


def make_perspective_grid(
    W=200,
    H=150,
    dmajor=10,
    dminor=1,
    ax_rot=20,
    ay_rot=25,
    f=600,
    k1=0.0,
    k2=0.0,
):
  """Return projected grid lines as (minor, major) lists of 2D point pairs."""
  axr, ayr = np.radians(ax_rot), np.radians(ay_rot)

  # Rotation matrices
  Rx = np.array(
      [[1, 0, 0],
       [0, np.cos(axr), -np.sin(axr)],
       [0, np.sin(axr),  np.cos(axr)]])
  Ry = np.array(
      [[np.cos(ayr), 0, np.sin(ayr)],
       [0, 1, 0],
       [-np.sin(ayr), 0, np.cos(ayr)]])
  R = Ry @ Rx

  def project(X, Y):
    """Project 3-D plane points with optional lens distortion."""
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

  xs_major = np.arange(-W / 2, W / 2 + dmajor, dmajor)
  ys_major = np.arange(-H / 2, H / 2 + dmajor, dmajor)
  xs_minor = np.arange(-W / 2, W / 2 + dminor, dminor)
  ys_minor = np.arange(-H / 2, H / 2 + dminor, dminor)

  lines_major, lines_minor = [], []

  # helper: vertical or horizontal line endpoints in 3-D projected space
  def project_line(x0, x1, y0, y1):
    X, Y = np.array([[x0, x1]]), np.array([[y0, y1]])
    return project(X, Y)

  for x in xs_minor:
    xp, yp = project_line(x, x, -H / 2, H / 2)
    lines_minor.append(np.column_stack([xp[0], yp[0]]))
  for y in ys_minor:
    xp, yp = project_line(-W / 2, W / 2, y, y)
    lines_minor.append(np.column_stack([xp[0], yp[0]]))

  for x in xs_major:
    xp, yp = project_line(x, x, -H / 2, H / 2)
    lines_major.append(np.column_stack([xp[0], yp[0]]))
  for y in ys_major:
    xp, yp = project_line(-W / 2, W / 2, y, y)
    lines_major.append(np.column_stack([xp[0], yp[0]]))

  return lines_minor, lines_major, project


def bezier_from_line(A, B, project, alpha=0.5):
  """
  Build a cubic Bezier Path approximating the lens-curved line between A,B.
  The midpoint of the original 3-D line (before projection) is used to
  estimate bulge; alpha controls curvature magnitude.
  """
  # find midpoint in 3-D before projection
  # we reverse-engineer 3-D midpoint from 2-D A,B assuming same z=0 plane
  # then re-project to include distortion nonlinearity
  X_mid = np.mean([A[0], B[0]])
  Y_mid = np.mean([A[1], B[1]])

  # small offset sampling around midpoint to estimate curvature
  Xs = np.linspace(A[0], B[0], 3)
  Ys = np.linspace(A[1], B[1], 3)
  xp, yp = project(Xs[None, :], Ys[None, :])
  M = np.array([xp[0, 1], yp[0, 1]])

  # control points pulled toward the distorted midpoint
  C1 = A + alpha * (M - A)
  C2 = B + alpha * (M - B)

  verts = [tuple(A), tuple(C1), tuple(C2), tuple(B)]
  codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
  return Path(verts, codes)


# --- Demo with barrel distortion drawn using Bezier curves ---
minor, major, project = make_perspective_grid(
    ax_rot=25,
    ay_rot=15,
    f=400,
    k1=-1e-6,
    k2=0.0,
)

fig, ax = plt.subplots(figsize=(8, 6))

patches_minor = []
for seg in minor:
  path = bezier_from_line(seg[0], seg[1], project, alpha=0.6)
  patches_minor.append(PathPatch(path, lw=0.5, edgecolor="#5aa3ff", facecolor="none", alpha=0.6))

patches_major = []
for seg in major:
  path = bezier_from_line(seg[0], seg[1], project, alpha=0.6)
  patches_major.append(PathPatch(path, lw=1.5, edgecolor="#ff5555", facecolor="none",
                                 linestyle="--", alpha=0.9))

pc_minor = PatchCollection(patches_minor, match_original=True)
pc_major = PatchCollection(patches_major, match_original=True)
ax.add_collection(pc_minor)
ax.add_collection(pc_major)

# Compute bounds dynamically
all_pts = np.concatenate([seg for seg in minor + major])
xmin, ymin = all_pts.min(axis=0)
xmax, ymax = all_pts.max(axis=0)
pad = 0.05 * max(xmax - xmin, ymax - ymin)
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
