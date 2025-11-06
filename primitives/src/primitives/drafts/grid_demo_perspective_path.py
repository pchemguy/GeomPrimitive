"""
grid_demo_perspective_path_polyline.py
--------------------------------------
Perspective + lens-distorted grid and shapes (square, circle)
where *all* edges are actually curved using the same optical
model (3D rotation + perspective + radial distortion).

Everything is defined in "paper" coordinates (WxH, Z=0)
and then projected to image space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection


# ================================================================
# 1) Perspective + lens distortion transform
# ================================================================
class PerspectiveProjector:
  """3D->2D perspective + radial lens distortion for points on Z=0."""

  def __init__(self, ax_rot=25, ay_rot=15, f=400, k1=-1e-4, k2=0.0):
    """
    Args:
      ax_rot, ay_rot: rotations about X and Y in degrees.
      f: focal length (larger = weaker perspective).
      k1, k2: radial distortion coefficients.
    """
    axr, ayr = np.radians(ax_rot), np.radians(ay_rot)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(axr), -np.sin(axr)],
                   [0.0, np.sin(axr),  np.cos(axr)]])
    Ry = np.array([[np.cos(ayr), 0.0, np.sin(ayr)],
                   [0.0,         1.0, 0.0],
                   [-np.sin(ayr), 0.0, np.cos(ayr)]])
    self.R = Ry @ Rx
    self.f = float(f)
    self.k1 = float(k1)
    self.k2 = float(k2)

  def __call__(self, X, Y, Z=None):
    """Transform coordinate arrays X, Y (and optional Z) of same shape."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if Z is None:
      Z = np.zeros_like(X)
    else:
      Z = np.asarray(Z, dtype=float)

    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])   # shape (3, N)
    Xr, Yr, Zr = self.R @ pts                             # rotated

    f = self.f
    xp = f * Xr / (f - Zr)
    yp = f * Yr / (f - Zr)

    # radial distortion in image plane
    r2 = xp**2 + yp**2
    d = 1.0 + self.k1 * r2 + self.k2 * r2**2
    xp *= d
    yp *= d

    return xp.reshape(X.shape), yp.reshape(Y.shape)

  def project_polyline(self, xs, ys):
    """Project a polyline given by 1D arrays xs, ys on Z=0 plane."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xp, yp = self(xs, ys)
    return np.column_stack([xp, yp])


# ================================================================
# 2) Grid as polylines (curved under distortion)
# ================================================================
def make_perspective_grid_polylines(
    projector,
    W=200,
    H=150,
    dmajor=10,
    dminor=1,
    n_samples=64,
):
  """
  Build minor/major grid as lists of polylines (each is (Nx2) array),
  sampling each world-space line at n_samples points before projection.
  """
  xs_major = np.arange(-W / 2, W / 2 + dmajor, dmajor)
  ys_major = np.arange(-H / 2, H / 2 + dmajor, dmajor)
  xs_minor = np.arange(-W / 2, W / 2 + dminor, dminor)
  ys_minor = np.arange(-H / 2, H / 2 + dminor, dminor)

  ys_line = np.linspace(-H / 2, H / 2, n_samples)
  xs_line = np.linspace(-W / 2, W / 2, n_samples)

  minor_segments = []
  major_segments = []

  # vertical minor lines
  for x in xs_minor:
    xs = np.full_like(ys_line, x)
    seg = projector.project_polyline(xs, ys_line)
    minor_segments.append(seg)

  # horizontal minor lines
  for y in ys_minor:
    ys = np.full_like(xs_line, y)
    seg = projector.project_polyline(xs_line, ys)
    minor_segments.append(seg)

  # vertical major lines
  for x in xs_major:
    xs = np.full_like(ys_line, x)
    seg = projector.project_polyline(xs, ys_line)
    major_segments.append(seg)

  # horizontal major lines
  for y in ys_major:
    ys = np.full_like(xs_line, y)
    seg = projector.project_polyline(xs_line, ys)
    major_segments.append(seg)

  return minor_segments, major_segments


# ================================================================
# 3) General Path -> polylines under same optics
# ================================================================
def transform_path_to_polylines(path, projector, n_samples=64):
  """
  Convert a Path (defined in paper coords) into a list of polylines
  (each an (Nx2) array) after perspective + lens distortion.

  Straight segments (LINETO) are resampled into n_samples points.
  Curved segments (CURVE3/CURVE4) are also resampled in param t.
  """
  verts = path.vertices
  codes = path.codes

  if codes is None:
    codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
    codes[0] = Path.MOVETO

  segments = []
  current_world = []   # world-space points along current subpath
  last = None

  for i, (v, c) in enumerate(zip(verts, codes)):
    if c == Path.MOVETO:
      # flush previous segment
      if current_world:
        current_world = []
      current_world.append(v)
      last = v

    elif c == Path.LINETO and last is not None:
      xs = np.linspace(last[0], v[0], n_samples)
      ys = np.linspace(last[1], v[1], n_samples)
      poly_world = np.column_stack([xs, ys])

      # avoid duplicating last point
      if len(current_world) > 0:
        current_world = current_world[:-1]
      current_world.extend(poly_world.tolist())
      last = v

    elif c == Path.CURVE3 and last is not None:
      # quadratic Bezier: last -> ctrl -> v
      ctrl = verts[i - 1]
      t = np.linspace(0.0, 1.0, n_samples)
      P = (1 - t)[:, None]**2 * last + \
          2 * (1 - t)[:, None] * t[:, None] * ctrl + \
          t[:, None]**2 * v
      if len(current_world) > 0:
        current_world = current_world[:-1]
      current_world.extend(P.tolist())
      last = v

    elif c == Path.CURVE4 and last is not None:
      # cubic Bezier: last -> ctrl1 -> ctrl2 -> v
      ctrl1, ctrl2 = verts[i - 2], verts[i - 1]
      t = np.linspace(0.0, 1.0, n_samples)
      P = ((1 - t)**3)[:, None] * last + \
          3 * ((1 - t)**2 * t)[:, None] * ctrl1 + \
          3 * ((1 - t) * t**2)[:, None] * ctrl2 + \
          (t**3)[:, None] * v
      if len(current_world) > 0:
        current_world = current_world[:-1]
      current_world.extend(P.tolist())
      last = v

    elif c == Path.CLOSEPOLY and current_world:
      # close back to first point in current_world
      first = np.array(current_world[0], dtype=float)
      last = np.array(current_world[-1], dtype=float)
      xs = np.linspace(last[0], first[0], n_samples)
      ys = np.linspace(last[1], first[1], n_samples)
      P = np.column_stack([xs, ys])
      current_world = current_world[:-1]
      current_world.extend(P.tolist())
      # now project the closed polyline
      world_arr = np.array(current_world, dtype=float)
      seg_proj = projector.project_polyline(world_arr[:, 0], world_arr[:, 1])
      segments.append(seg_proj)
      current_world = []
      last = None

    else:
      last = v

  # leftover open segment (if any)
  if current_world:
    world_arr = np.array(current_world, dtype=float)
    seg_proj = projector.project_polyline(world_arr[:, 0], world_arr[:, 1])
    segments.append(seg_proj)

  return segments


# ================================================================
# 4) Demo
# ================================================================
if __name__ == "__main__":
  # Stronger k1 so curvature is obvious; dial back later.
  proj = PerspectiveProjector(ax_rot=25, ay_rot=15, f=600, k1=2e-6, k2=2e-10)


  # Grid
  minor, major = make_perspective_grid_polylines(
      projector=proj,
      W=200,
      H=150,
      dmajor=10,
      dminor=1,
      n_samples=64,
  )

  fig, ax = plt.subplots(figsize=(8, 6))

  lc_minor = LineCollection(minor, colors="#aaccff", linewidths=0.5, alpha=0.6)
  lc_major = LineCollection(major, colors="#3366cc", linewidths=1.2, alpha=0.9)
  ax.add_collection(lc_minor)
  ax.add_collection(lc_major)

  # Shapes defined in *paper* coordinates (before projection)
  circle = Path.circle(center=(30, 20), radius=25)
  square = Path([[-70, -70],
                 [70, -70],
                 [70,  70],
                 [-70, 70],
                 [-70, -70]],
                [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY])

  circle_segs = transform_path_to_polylines(circle, proj, n_samples=64)
  square_segs = transform_path_to_polylines(square, proj, n_samples=64)

  lc_circle = LineCollection(circle_segs, colors="black", linewidths=2.0)
  lc_square = LineCollection(square_segs, colors="green", linewidths=2.0)
  ax.add_collection(lc_circle)
  ax.add_collection(lc_square)

  # Bounds from all geometry
  all_pts = np.concatenate(minor + major + circle_segs + square_segs)
  xmin, ymin = all_pts.min(axis=0)
  xmax, ymax = all_pts.max(axis=0)
  pad = 0.05 * max(xmax - xmin, ymax - ymin)
  ax.set_xlim(xmin - pad, xmax + pad)
  ax.set_ylim(ymin - pad, ymax + pad)

  ax.set_aspect("equal")
  ax.axis("off")
  plt.show()
