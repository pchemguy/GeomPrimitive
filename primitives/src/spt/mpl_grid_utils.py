"""
mpl_grid_utils.py
-----------------

https://chatgpt.com/c/69120de6-5468-832d-8bff-88120cb94daa

PROMPT

Let's work on a flexible Matplotlib grid generator utility.
Output:
    Four line collections - x_major_lc, x_minor_lc, y_major_lc, y_minor_lc
The arguments:
- bounding box (bottom-left and top-tight coners)
- grid angle (alpha=90 - for ordinary Cartesian coords, 90>alpha>0 for generic grids
- theta in [0, 90) - rotation of the grid in CCW
- x_major, x_minor, y_major, y_minor - respective sub-grid spacings

I want to be able to jitter both alpha and theta symmetrically (normal distribution
with 3sigma = 5 deg). Further, I want to jitter shift of individual lines symmetrically
about exact positions with 3sigma = 0.4*step (minor for minor lines and major for major
lines.) I also want to jitter angles of individual lines symmetrically with
3sigma = 3 deg. Not all lines should be affected by jitter however. I want select
a random fraction for each of the four types (selected fraction normally distributed
with mu=0% and sigma=25%). I also want to randomly drop a fraction (mu=0%, sigma=5%)
lines for each group.
"""

from __future__ import annotations

__all__ = ["GridJitterConfig", "generate_grid_collections",]

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rng import RNGBackend

numeric = Union[int, float]
PointXY = tuple[numeric, numeric]
BBoxBounds = tuple[PointXY, PointXY, PointXY, PointXY]
BBox = Union[tuple[PointXY, PointXY], BBoxBounds]


@dataclass
class GridJitterConfig:
    """Configuration for grid-level and line-level jitter.

    Attributes:
        global_angle_3sigma_deg:   3sigma for global jitter of (alpha, theta), degrees.
        line_angle_3sigma_deg:     3sigma for per-line angle jitter, degrees.
        line_offset_3sigma_factor: 3sigma for per-line offset jitter as a fraction of
            the corresponding step (major/minor).
        jitter_fraction_sigma:     sigma for the fraction of lines that receive per-line
            jitter in each group (x_major, x_minor, y_major, y_minor). The fraction
            itself is sampled as |N(0, jitter_fraction_sigma)| and clipped to [0, 1].
        drop_fraction_sigma:       sigma for the fraction of lines randomly dropped in
            each group, sampled as |N(0, drop_fraction_sigma)|, clipped to [0, 1].
    """
    global_angle_3sigma_deg:   float = 5.0
    line_angle_3sigma_deg:     float = 3.0
    line_offset_3sigma_factor: float = 0.4
    jitter_fraction_sigma:     float = 0.25
    drop_fraction_sigma:       float = 0.05


def generate_grid_collections(bbox: BBox,
                              alpha_deg: float, theta_deg: float,
                              x_major: float, x_minor: float,
                              y_major: float, y_minor: float,
                              jitter: GridJitterConfig = None,
                              rng: RNGBackend = None,
    ) -> tuple[LineCollection, LineCollection, LineCollection, LineCollection]:
    """Generate 4 LineCollections for an oblique, optionally jittered grid.

    Args:
        bbox:
            Bounding box as ((x_min, y_min), (x_max, y_max)) in world coordinates.
        alpha_deg:
            Grid skew angle between x- and y-families in degrees.
            alpha_deg = 90 => ordinary Cartesian (orthogonal) grid.
            0 < alpha_deg <= 90 for generic oblique grids.
        theta_deg:
            Rotation of the x-family axis in degrees (CCW, world coordinates).
            The y-family axis is at theta_deg + alpha_deg.
        x_major:
            Step along the x-coordinate for major x-lines (world-independent).
        x_minor:
            Step along the x-coordinate for minor x-lines.
        y_major:
            Step along the y-coordinate for major y-lines.
        y_minor:
            Step along the y-coordinate for minor y-lines.
        jitter:
            If provided, enables all jitter behavior controlled by GridJitterConfig.
            If None, the grid is perfectly regular (no jitter, no dropping).
        rng:
            Optional NumPy random Generator. If None, a default_rng() is used.

    Returns:
        (x_major_lc, x_minor_lc, y_major_lc, y_minor_lc)
        Each is a matplotlib.collections.LineCollection.
    """
    x_min, y_min, x_max, y_max = _parse_bbox(bbox)

    if rng is None:
        rng = np.random.default_rng()

    # ---------------------------------------------------------------------------
    # 1) Global angle jitter for alpha and theta
    # ---------------------------------------------------------------------------
    if jitter is not None and jitter.global_angle_3sigma_deg > 0:
        sigma_global_rad = math.radians(jitter.global_angle_3sigma_deg / 3.0)
        theta = math.radians(theta_deg) + rng.normal(0.0, sigma_global_rad)
        alpha = math.radians(alpha_deg) + rng.normal(0.0, sigma_global_rad)
    else:
        theta = math.radians(theta_deg)
        alpha = math.radians(alpha_deg)

    # Basis vectors for the oblique grid:
    # x-family coordinate axis (u) at angle theta
    # y-family coordinate axis (v) at angle theta + alpha
    ex = np.array([math.cos(theta), math.sin(theta)])        # x-axis direction
    ey = np.array([math.cos(theta + alpha), math.sin(theta + alpha)])  # y-axis

    # Linear map (u, v) -> (x, y) has columns ex, ey
    M = np.array([[ex[0], ey[0]], [ex[1], ey[1]]], dtype=float)
    Minv = np.linalg.inv(M)

    # ---------------------------------------------------------------------------
    # 2) Transform bbox corners -> (u, v) coord space to get ranges for lines
    # ---------------------------------------------------------------------------
    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max],
    ])
    uv_corners = corners @ Minv.T
    u_vals = uv_corners[:, 0]
    v_vals = uv_corners[:, 1]
    u_min, u_max = float(u_vals.min()), float(u_vals.max())
    v_min, v_max = float(v_vals.min()), float(v_vals.max())

    # ---------------------------------------------------------------------------
    # 3) Common jitter sigmas for per-line effects
    # ---------------------------------------------------------------------------
    if jitter is not None:
        line_angle_sigma_rad = math.radians(jitter.line_angle_3sigma_deg / 3.0)
    else:
        line_angle_sigma_rad = 0.0

    # ---------------------------------------------------------------------------
    # 4) Build each of the four line families
    #    - X-family lines: u = const, direction ~ ey
    #    - Y-family lines: v = const, direction ~ ex
    # ---------------------------------------------------------------------------
    bbox_tuple = (x_min, y_min, x_max, y_max)

    # X major
    x_major_segments = _build_line_family(
        coord_min=u_min,
        coord_max=u_max,
        step=x_major,
        fixed_vec=ex,
        dir_vec=ey,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=x_major,
        line_angle_sigma_rad=line_angle_sigma_rad,
    )

    # X minor
    x_minor_segments = _build_line_family(
        coord_min=u_min,
        coord_max=u_max,
        step=x_minor,
        fixed_vec=ex,
        dir_vec=ey,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=x_minor,
        line_angle_sigma_rad=line_angle_sigma_rad,
    )

    # Y major
    y_major_segments = _build_line_family(
        coord_min=v_min,
        coord_max=v_max,
        step=y_major,
        fixed_vec=ey,
        dir_vec=ex,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=y_major,
        line_angle_sigma_rad=line_angle_sigma_rad,
    )

    # Y minor
    y_minor_segments = _build_line_family(
        coord_min=v_min,
        coord_max=v_max,
        step=y_minor,
        fixed_vec=ey,
        dir_vec=ex,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=y_minor,
        line_angle_sigma_rad=line_angle_sigma_rad,
    )

    # ---------------------------------------------------------------------------
    # 5) Wrap in LineCollections
    # ---------------------------------------------------------------------------
    x_major_lc = LineCollection(x_major_segments)
    x_minor_lc = LineCollection(x_minor_segments)
    y_major_lc = LineCollection(y_major_segments)
    y_minor_lc = LineCollection(y_minor_segments)

    return x_major_lc, x_minor_lc, y_major_lc, y_minor_lc


# =============================================================================
# Internals
# =============================================================================

def _parse_bbox(bbox: BBox) -> BBoxBounds:
    """Normalize bbox to (x_min, y_min, x_max, y_max)."""
    if len(bbox) == 2 and isinstance(bbox[0], (tuple, list)):
        (x0, y0), (x1, y1) = bbox  # type: ignore[misc]
    elif len(bbox) == 4:
        x0, y0, x1, y1 = bbox  # type: ignore[misc]
    else:
        raise ValueError("bbox must be ((x_min, y_min), (x_max, y_max)) or (x0, y0, x1, y1)")

    x_min = float(min(x0, x1))
    x_max = float(max(x0, x1))
    y_min = float(min(y0, y1))
    y_max = float(max(y0, y1))
    return x_min, y_min, x_max, y_max


def _sample_fraction(rng: RNGBackend, sigma: float) -> float:
    """Sample a [0, 1] fraction from |N(0, sigma)|, clipped."""
    if sigma <= 0:
        return 0.0
    val = abs(rng.normal(0.0, sigma))
    return float(np.clip(val, 0.0, 1.0))


def _build_line_family(coord_min: float, coord_max: float, step: float,
                       fixed_vec: NDArray, dir_vec: NDArray, bbox: BBoxBounds,
                       rng: RNGBackend, jitter: GridJitterConfig,
                       base_offset_step: float, line_angle_sigma_rad: float,) -> NDArray:
    """Generate clipped line segments for one family (major/minor X or Y).

    Args:
        coord_min: Minimum coordinate (u or v) over bbox corners.
        coord_max: Maximum coordinate (u or v) over bbox corners.
        step:      Spacing for this family (major or minor) along its coordinate.
        fixed_vec: Unit vector along which the fixed coordinate is applied
            (ex for X-family, ey for Y-family).
        dir_vec:   Nominal direction vector of the lines (ey for X-family,
            ex for Y-family).
        bbox:      (x_min, y_min, x_max, y_max) in world coords.
        rng:       NumPy RNG.
        jitter:   Jitter configuration or None.
        base_offset_step:     The step size used to scale the offset jitter.
        line_angle_sigma_rad: sigma for per-line angle jitter in radians.

    Returns:
        segments: ndarray of shape (n_lines, 2, 2).
    """
    x_min, y_min, x_max, y_max = bbox

    if step <= 0:
        return np.empty((0, 2, 2), dtype=float)

    # Determine index range in coordinate space that covers the bbox, with a small
    # margin to avoid missing lines due to rounding.
    k_min = int(math.floor(coord_min / step)) - 1
    k_max = int(math.ceil(coord_max / step)) + 1
    if k_max < k_min:
        return np.empty((0, 2, 2), dtype=float)

    coord_values = np.arange(k_min, k_max + 1, dtype=float) * step
    n = coord_values.size
    if n == 0:
        return np.empty((0, 2, 2), dtype=float)

    indices = np.arange(n)

    # ---------------------------------------------------------------------------
    # Per-group jitter/drop fractions (independent for each of the 4 families)
    # ---------------------------------------------------------------------------
    if jitter is not None:
        jitter_frac = _sample_fraction(rng, jitter.jitter_fraction_sigma)
        drop_frac = _sample_fraction(rng, jitter.drop_fraction_sigma)
        offset_sigma = (jitter.line_offset_3sigma_factor * base_offset_step) / 3.0
    else:
        jitter_frac = 0.0
        drop_frac = 0.0
        offset_sigma = 0.0

    # Which lines to drop?
    keep_mask = np.ones(n, dtype=bool)
    if drop_frac > 0:
        n_drop = int(round(drop_frac * n))
        n_drop = min(max(n_drop, 0), n)
        if n_drop > 0:
            drop_idx = rng.choice(indices, size=n_drop, replace=False)
            keep_mask[drop_idx] = False

    # Which remaining lines get jitter?
    jitter_mask = np.zeros(n, dtype=bool)
    if jitter_frac > 0 and keep_mask.any():
        n_keep = int(keep_mask.sum())
        n_jitter = int(round(jitter_frac * n_keep))
        n_jitter = min(max(n_jitter, 0), n_keep)
        if n_jitter > 0:
            candidate_idx = indices[keep_mask]
            chosen = rng.choice(candidate_idx, size=n_jitter, replace=False)
            jitter_mask[chosen] = True

    # ---------------------------------------------------------------------------
    # Build segments
    # ---------------------------------------------------------------------------
    base_phi = math.atan2(dir_vec[1], dir_vec[0])
    segments = []

    for i, c0 in enumerate(coord_values):
        if not keep_mask[i]:
            continue

        # Base coordinate (offset) and angle
        c = c0
        phi = base_phi

        if jitter is not None and jitter_mask[i]:
            if offset_sigma > 0:
                c += rng.normal(0.0, offset_sigma)
            if line_angle_sigma_rad > 0:
                phi += rng.normal(0.0, line_angle_sigma_rad)

        # Line direction in world coordinates
        d = np.array([math.cos(phi), math.sin(phi)], dtype=float)
        # Base point displaced along fixed_vec at coordinate c
        p0 = c * fixed_vec

        seg = _clip_infinite_line_to_bbox(p0, d, x_min, y_min, x_max, y_max)
        if seg is not None:
            segments.append(seg)

    if not segments:
        return np.empty((0, 2, 2), dtype=float)

    return np.stack(segments, axis=0)


def _clip_infinite_line_to_bbox(p0: NDArray, d: NDArray,
                                x_min: float, y_min: float,
                                x_max: float, y_max: float) -> NDArray:
    """Clip an infinite line to an axis-aligned bbox.

    The line is p(t) = p0 + t * d.

    Returns:
        A (2, 2) array [[x1, y1], [x2, y2]] if it intersects the bbox with a segment,
        otherwise None.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(d[0]), float(d[1])
    eps = 1e-12

    ts = []

    # Intersect with x = x_min and x = x_max
    if abs(dx) > eps:
        for xb in (x_min, x_max):
            t = (xb - x0) / dx
            y = y0 + t * dy
            if y_min - eps <= y <= y_max + eps:
                ts.append(t)

    # Intersect with y = y_min and y = y_max
    if abs(dy) > eps:
        for yb in (y_min, y_max):
            t = (yb - y0) / dy
            x = x0 + t * dx
            if x_min - eps <= x <= x_max + eps:
                ts.append(t)

    if len(ts) < 2:
        return None

    # Sort and take the extreme intersection points
    ts_sorted = sorted(ts)
    t1, t2 = ts_sorted[0], ts_sorted[-1]

    p1 = np.array([x0 + t1 * dx, y0 + t1 * dy], dtype=float)
    p2 = np.array([x0 + t2 * dx, y0 + t2 * dy], dtype=float)
    return np.stack([p1, p2], axis=0)


def main():
    """
    from mpl_grid_utils import (
        generate_grid_collections,
        GridJitterConfig,
    )
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    bbox = ((-10.0, -10.0), (10.0, 10.0))
    
    x_major_lc, x_minor_lc, y_major_lc, y_minor_lc = generate_grid_collections(
      bbox=bbox,
      alpha_deg=90.0,      # orthogonal
      theta_deg=15.0,      # rotate grid 15deg CCW
      x_major=2.0,
      x_minor=0.5,
      y_major=2.0,
      y_minor=0.5,
      jitter=GridJitterConfig(),  # enable jitter with defaults
    )
    
    # Style the collections however you like
    x_major_lc.set_linewidth(1.2)
    x_major_lc.set_alpha(0.7)
    
    x_minor_lc.set_linewidth(0.6)
    x_minor_lc.set_alpha(0.4)
    
    y_major_lc.set_linewidth(1.2)
    y_major_lc.set_alpha(0.7)
    
    y_minor_lc.set_linewidth(0.6)
    y_minor_lc.set_alpha(0.4)
    
    ax.add_collection(x_major_lc)
    ax.add_collection(x_minor_lc)
    ax.add_collection(y_major_lc)
    ax.add_collection(y_minor_lc)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
