"""
grid_demo_perspective_path.py
-----------------------------
Perspective + lens-distorted grid drawn with cubic Bezier curves,
and arbitrary Path objects (e.g. circle/polygon) projected through
the same 3D + lens transform for consistent realism.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


# ================================================================
# 1 Perspective + Lens Transform Class
# ================================================================
class PerspectiveProjector:
  """Callable 3D->2D perspective + lens distortion transform."""

  def __init__(self, ax_rot=25, ay_rot=15, f=400, k1=-1e-6, k2=0.0):
    axr, ayr = np.radians(ax_rot), np.radians(ay_rot)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(axr), -np.sin(axr)],
                   [0, np.sin(axr),  np.cos(axr)]])
    Ry = np.array([[np.cos(ayr), 0, np.sin(ayr)],
                   [0, 1, 0],
                   [-np.sin(ayr), 0, np.cos(ayr)]])
    self.R = Ry @ Rx
    self.f = f
    self.k1 = k1
    self.k2 = k2

  def __call__(self, X, Y, Z=None):
    """Transform coordinate arrays of same shape."""
    if Z is None:
      Z = np.zeros_like(X)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    Xr, Yr, Zr = self.R @ pts
    f = self.f
    xp = f * Xr / (f - Zr)
    yp = f * Yr / (f - Zr)
    r2 = xp**2 + yp**2
    d = 1 + self.k1 * r2 + self.k2 * r2**2
    xp *= d
    yp *= d
    return xp.reshape(X.shape), yp.reshape(Y.shape)

  def transform_points(self, pts):
    """Transform an (Nx2) array of XY points."""
    X, Y = pts[:, 0], pts[:, 1]
    x2d, y2d = self(X, Y)
    return np.column_stack([x2d, y2d])

  def transform_path(self, path):
    """Return a new Path whose vertices are transformed."""
    new_verts = self.transform_points(path.vertices)
    return Path(new_verts, path.codes)


# ================================================================
# 2 Grid Generator (uses projector)
# ================================================================
def make_perspective_grid(
    projector,
    W=200,
    H=150,
    dmajor=10,
    dminor=1,
):
  """Return projected grid line endpoints (minor, major)."""
  xs_major = np.arange(-W / 2, W / 2 + dmajor, dmajor)
  ys_major = np.arange(-H / 2, H / 2 + dmajor, dmajor)
  xs_minor = np.arange(-W / 2, W / 2 + dminor, dminor)
  ys_minor = np.arange(-H / 2, H / 2 + dminor, dminor)

  def project_line(x0, x1, y0, y1):
    X, Y = np.array([[x0, x1]]), np.array([[y0, y1]])
    return projector(X, Y)

  lines_major, lines_minor = [], []

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

  return lines_minor, lines_major


# ================================================================
# 3 Bezier Approximation for Lens Curvature
# ================================================================
def bezier_from_line(A, B, projector, alpha=0.5):
  """Approximate a curved, distorted line between A and B with a cubic Bezier."""
  # sample midpoint (in 2-D coordinates)
  Xs = np.linspace(A[0], B[0], 3)
  Ys = np.linspace(A[1], B[1], 3)
  xp, yp = projector(Xs[None, :], Ys[None, :])
  M = np.array([xp[0, 1], yp[0, 1]])  # distorted midpoint
  # control points pulled toward midpoint
  C1 = A + alpha * (M - A)
  C2 = B + alpha * (M - B)
  verts = [tuple(A), tuple(C1), tuple(C2), tuple(B)]
  codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
  return Path(verts, codes)


# ================================================================
# 4 Demo
# ================================================================
if __name__ == "__main__":
  # same projector for grid and shape
  proj = PerspectiveProjector(ax_rot=25, ay_rot=15, f=400, k1=-1e-6, k2=0.0)
  minor, major = make_perspective_grid(proj)

  fig, ax = plt.subplots(figsize=(8, 6))

  # Bezier grid
  patches_minor, patches_major = [], []
  for seg in minor:
    path = bezier_from_line(seg[0], seg[1], proj, alpha=0.6)
    patches_minor.append(
        PathPatch(path, lw=0.5, edgecolor="#5aa3ff", facecolor="none", alpha=0.6))
  for seg in major:
    path = bezier_from_line(seg[0], seg[1], proj, alpha=0.6)
    patches_major.append(
        PathPatch(path, lw=1.5, edgecolor="#ff5555", facecolor="none",
                  linestyle="--", alpha=0.9))

  pc_minor = PatchCollection(patches_minor, match_original=True)
  pc_major = PatchCollection(patches_major, match_original=True)
  ax.add_collection(pc_minor)
  ax.add_collection(pc_major)

  # Example shape: circle drawn "on the paper"
  circle = Path.circle(center=(30, 20), radius=25)
  distorted_circle = proj.transform_path(circle)
  ax.add_patch(PathPatch(distorted_circle, facecolor="none",
                         edgecolor="black", lw=2))

  # Example polygon: square
  square = Path([[-70, -70], [70, -70], [70, 70], [-70, 70], [-70, -70]],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
  distorted_square = proj.transform_path(square)
  ax.add_patch(PathPatch(distorted_square, facecolor="none",
                         edgecolor="green", lw=1.5))

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
