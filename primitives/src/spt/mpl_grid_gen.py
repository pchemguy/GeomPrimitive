"""
mpl_grid_gen.py
-----------------

Utilities for generating flexible 2D grids with optional geometric distortion.

The main API produces four Matplotlib ``LineCollection`` objects corresponding
to the four standard sub-grids:

  - X-major lines
  - X-minor lines
  - Y-major lines
  - Y-minor lines

This layout generalizes the major/minor grid structure commonly found in
spreadsheet applications, plotting libraries, and diagramming tools, while
extending it to support oblique (non-orthogonal) and rotated grids.

In addition to deterministic grid construction, the module provides a set of
stochastic distortion mechanisms that can be applied independently to each
sub-grid:

  - Angle jitter
      - Obliquity (inter-axis) angle
      - Global grid rotation angle
      - Individual line orientation

  - Spacing jitter
      - Randomized offsets of major/minor line positions

  - Line dropout
      - Random removal of individual grid lines

All distortion parameters are expressed as 3sigma bounds for truncated normal
distributions. Each bound K defines both a 3sigma range and a hard cutoff:

  - Symmetric jitter:
        K * normal(0, 1/3) clipped to [-K, +K]

  - One-sided jitter (fractions, dropout rates):
        K * |normal(0, 1/3)| clipped to [0, K]

These conventions ensure stable, statistically well-behaved distortions while
preserving simple, intuitive bounds on all grid perturbations.
"""

from __future__ import annotations

__all__ = ["GridJitterConfig", "generate_grid_collections", "debug_dump_grid_info",]

import os
import sys
import math
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from utils.rng import RNGBackend, RNG, get_rng

BBoxTuple = Tuple[float, float, float, float]
BBoxLike = Union[Tuple[Tuple[float, float], Tuple[float, float]], BBoxTuple]


@dataclass(frozen=True)
class GridJitterConfig:
    """Configuration for grid-level and line-level jitter.

    All parameters represent 3sigma bounds for truncated normal distributions.
    All values must be non-negative real numbers (coerced via float()).

    Attributes:
        global_angle_deg:
            3sigma bound for global jitter of obliquity (alpha) and rotation (theta),
            in degrees.

        line_angle_deg:
            3sigma bound for per-line orientation jitter, in degrees.

        line_offset_factor:
            3sigma bound for per-line positional offset as a fraction of the
            corresponding spacing (major/minor).

        line_offset_fraction:
            3sigma bound for the fraction of lines in each sub-grid that receive
            per-line jitter.

        drop_fraction:
            3sigma bound for the fraction of lines randomly removed in each
            sub-grid.
    """

    global_angle_deg     : float = 6.0
    line_angle_deg       : float = 3.0
    line_offset_factor   : float = 0.4
    line_offset_fraction : float = 0.25
    drop_fraction        : float = 0.05

    def __post_init__(self) -> None:
        """Validate that all fields are non-negative Real and coerce to float."""
        for field_name, value in vars(self).items():
            if not isinstance(value, Real):
                raise TypeError(
                    f"{field_name} must be a real number, "
                    f"got {value!r} of type {type(value).__name__}"
                )

            float_val = float(value)
            if float_val < 0:
                raise ValueError(
                    f"{field_name} must be non-negative, got {float_val!r}"
                )

            object.__setattr__(self, field_name, float_val)

    # ---------------------------------------------------------------------------
    # Cached default instance
    # ---------------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=1)
    def DEFAULT() -> "GridJitterConfig":
        """Return a cached default configuration instance."""
        return GridJitterConfig()

    # ---------------------------------------------------------------------------
    # Internal primitive: symmetric truncated normal in [-1, 1]
    # ---------------------------------------------------------------------------
    @staticmethod
    def normal3s(rng: RNGBackend) -> float:
        """Return a symmetric truncated normal sample in [-1, 1]."""
        if hasattr(rng, "normal"):            
            sample = rng.normal(0, 1/3)
        else:
            sample = rng.normalvariate(0, 1/3)
        return max(-1.0, min(1.0, float(sample)))

    # ---------------------------------------------------------------------------
    # Public sampling helpers: signed and one-sided jitters
    # ---------------------------------------------------------------------------
    def jitter_angle_global(self, rng: object) -> float:
        """Return a signed global angle jitter in degrees for alpha and theta."""
        if self.global_angle_deg <= 0.0:
            return 0.0
        return self.global_angle_deg * self.normal3s(rng)

    def jitter_line_angle(self, rng: object) -> float:
        """Return a per-line signed angle jitter in degrees."""
        if self.line_angle_deg <= 0.0:
            return 0.0
        return self.line_angle_deg * self.normal3s(rng)

    def jitter_line_offset(self, rng: object) -> float:
        """Return a signed offset coefficient in [-line_offset_factor, +line_offset_factor].

        This coefficient is dimensionless and should be multiplied by the
        corresponding spacing (major/minor) to obtain an absolute offset.
        """
        if self.line_offset_factor <= 0.0:
            return 0.0
        return self.line_offset_factor * self.normal3s(rng)

    def jitter_offset_fraction(self, rng: object) -> float:
        """Return the fraction of lines that receive per-line jitter.

        Result is in [0, line_offset_fraction].
        """
        if self.line_offset_fraction <= 0.0:
            return 0.0
        return self.line_offset_fraction * abs(self.normal3s(rng))

    def jitter_drop_fraction(self, rng: object) -> float:
        """Return the fraction of lines dropped in a sub-grid.

        Result is in [0, drop_fraction].
        """
        if self.drop_fraction <= 0.0:
            return 0.0
        return self.drop_fraction * abs(self.normal3s(rng))

    # ---------------------------------------------------------------------------
    # Hand-drawn preset
    # ---------------------------------------------------------------------------
    @staticmethod
    def hand_drawn() -> "GridJitterConfig":
        """Return a jitter configuration suitable for hand-drawn style grids.

        Characteristics:
            - Large global angle jitter (alpha, theta)
            - Strong per-line orientation jitter
            - High positional jitter (offset factor)
            - Large fraction of jittered lines
            - Moderate dropout for visual irregularity
        """
        return GridJitterConfig(
            global_angle_deg=12.0,        # strong wobble of overall frame
            line_angle_deg=8.0,           # strong per-line orientation noise
            line_offset_factor=0.8,       # +/-80% of step at 3sigma
            line_offset_fraction=0.50,    # ~50% of lines receive jitter
            drop_fraction=0.08,           # ~8% dropout
        )    
    
    # ---------------------------------------------------------------------------
    # Style preset factory
    # ---------------------------------------------------------------------------
    @staticmethod
    def preset(name: str) -> "GridJitterConfig":
        """
        Return a deterministic jitter configuration corresponding to a named style.

        Supported styles:
            - "sketchy"
            - "technical"
            - "messy"
            - "blueprint"
            - "handwriting_synthetic"
            - "engineering_paper"
            - "architectural_drift"
            - "printlike_subtle"

        Returns:
            A GridJitterConfig instance.
        """

        style = name.lower().strip()

        if style == "sketchy":
            return GridJitterConfig(global_angle_deg=12.0, line_angle_deg=8.0,
                line_offset_factor=0.8, line_offset_fraction=0.50, drop_fraction=0.07)
        if style == "technical":
            return GridJitterConfig(global_angle_deg=1.5, line_angle_deg=1.0,
                line_offset_factor=0.10, line_offset_fraction=0.10, drop_fraction=0.01)
        if style == "messy":
            return GridJitterConfig(global_angle_deg=20.0, line_angle_deg=18.0,
                line_offset_factor=1.2, line_offset_fraction=0.80, drop_fraction=0.18)
        if style == "blueprint":
            return GridJitterConfig(global_angle_deg=0.3, line_angle_deg=0.3,
                line_offset_factor=0.05, line_offset_fraction=0.05, drop_fraction=0.01)
        if style == "handwriting_synthetic":
            return GridJitterConfig(global_angle_deg=4.0, line_angle_deg=3.0,
                line_offset_factor=0.20, line_offset_fraction=0.40, drop_fraction=0.05)
        if style == "engineering_paper":
            return GridJitterConfig(global_angle_deg=1.5, line_angle_deg=0.8,
                line_offset_factor=0.06, line_offset_fraction=0.10, drop_fraction=0.01)
        if style == "architectural_drift":
            return GridJitterConfig(global_angle_deg=5.5, line_angle_deg=0.3,
                line_offset_factor=0.03, line_offset_fraction=0.05, drop_fraction=0.00)
        if style == "printlike_subtle":
            return GridJitterConfig(global_angle_deg=0.15, line_angle_deg=0.15,
                line_offset_factor=0.01, line_offset_fraction=0.02, drop_fraction=0.00)

        raise ValueError(
            f"Unknown jitter preset '{name}'.\n"
            "Valid presets: sketchy, technical, messy, blueprint, "
            "handwriting_synthetic, engineering_paper, architectural_drift, "
            "printlike_subtle."
        )


def generate_grid_collections(
        bbox          : BBoxLike,
        obliquity_deg : float,
        rotation_deg  : float,
        x_major_step  : float,
        x_minor_step  : float,
        y_major_step  : float,
        y_minor_step  : float,
        jitter        : GridJitterConfig = None,
        rng           : RNGBackend       = None,
    ) -> Tuple[LineCollection, LineCollection, LineCollection, LineCollection]:
    """Generate LineCollections for a 2D oblique grid with optional jitter.

    Args:
        bbox:
            Bounding box as either ((x_min, y_min), (x_max, y_max)) or
            (x_min, y_min, x_max, y_max).

        obliquity_deg:
            Inter-axis (obliquity) angle between the x and y grid directions,
            in degrees. 90deg corresponds to an orthogonal grid.

        rotation_deg:
            Rotation of the x-family axis in degrees (CCW, world coordinates).
            The y-family axis is located at rotation_deg + obliquity_deg.

        x_major_step:
            Spacing for X-major grid lines (in x-coordinate units).

        x_minor_step:
            Spacing for X-minor grid lines.

        y_major_step:
            Spacing for Y-major grid lines.

        y_minor_step:
            Spacing for Y-minor grid lines.

        jitter:
            Optional GridJitterConfig. If provided, enables stochastic
            distortion of grid angles, offsets, and line dropout.

        rng:
            Optional random number generator. If None, a new
            random.Random() instance is created. The RNG must support at
            least one of: normal3s(), normalvariate(), normal().

    Returns:
        A tuple of four LineCollections:
            (x_major_lc, x_minor_lc, y_major_lc, y_minor_lc).
    """
    x_min, y_min, x_max, y_max = _parse_bbox(bbox)

    if rng is None:
        rng = random.Random()

    # ---------------------------------------------------------------------------
    # 1) Global angle setup and jitter
    # ---------------------------------------------------------------------------
    theta_rad = math.radians(rotation_deg)
    alpha_rad = math.radians(obliquity_deg)

    if jitter is not None and jitter.global_angle_deg > 0.0:
        theta_rad += math.radians(jitter.jitter_angle_global(rng))
        alpha_rad += math.radians(jitter.jitter_angle_global(rng))

    # Basis vectors for the oblique grid.
    # X-family axis (u) at angle theta_rad.
    # Y-family axis (v) at angle theta_rad + alpha_rad.
    ex = np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=float)
    ey = np.array(
        [math.cos(theta_rad + alpha_rad), math.sin(theta_rad + alpha_rad)],
        dtype=float,
    )

    # (u, v) -> (x, y) linear map has columns ex, ey.
    M = np.array([[ex[0], ey[0]], [ex[1], ey[1]]], dtype=float)
    Minv = np.linalg.inv(M)

    # ---------------------------------------------------------------------------
    # 2) Express bbox corners in (u, v) coordinates to determine index ranges
    # ---------------------------------------------------------------------------
    corners = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_min, y_max],
            [x_max, y_max],
        ],
        dtype=float,
    )
    uv_corners = corners @ Minv.T
    u_vals = uv_corners[:, 0]
    v_vals = uv_corners[:, 1]
    u_min, u_max = float(u_vals.min()), float(u_vals.max())
    v_min, v_max = float(v_vals.min()), float(v_vals.max())

    bbox_tuple: BBoxTuple = (x_min, y_min, x_max, y_max)

    # ---------------------------------------------------------------------------
    # 3) Build line families: constants in u or v, direction ex or ey
    # ---------------------------------------------------------------------------
    x_major_segments = _build_line_family(
        coord_min=u_min,
        coord_max=u_max,
        step=x_major_step,
        fixed_vec=ex,
        dir_vec=ey,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=x_major_step,
    )

    x_minor_segments = _build_line_family(
        coord_min=u_min,
        coord_max=u_max,
        step=x_minor_step,
        fixed_vec=ex,
        dir_vec=ey,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=x_minor_step,
    )

    y_major_segments = _build_line_family(
        coord_min=v_min,
        coord_max=v_max,
        step=y_major_step,
        fixed_vec=ey,
        dir_vec=ex,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=y_major_step,
    )

    y_minor_segments = _build_line_family(
        coord_min=v_min,
        coord_max=v_max,
        step=y_minor_step,
        fixed_vec=ey,
        dir_vec=ex,
        bbox=bbox_tuple,
        rng=rng,
        jitter=jitter,
        base_offset_step=y_minor_step,
    )

    # ---------------------------------------------------------------------------
    # 4) Wrap into LineCollections
    # ---------------------------------------------------------------------------
    x_major_lc = LineCollection(x_major_segments)
    x_minor_lc = LineCollection(x_minor_segments)
    y_major_lc = LineCollection(y_major_segments)
    y_minor_lc = LineCollection(y_minor_segments)

    return x_major_lc, x_minor_lc, y_major_lc, y_minor_lc


def debug_dump_grid_info(
        bbox          : BBoxLike,
        obliquity_deg : float,
        rotation_deg  : float,
        x_major_step  : float,
        x_minor_step  : float,
        y_major_step  : float,
        y_minor_step  : float,
        jitter        : GridJitterConfig = None,
        rng           : RNGBackend       = None,
        file                             = None,
    ) -> None:
    """
    Print a detailed diagnostic dump of grid construction internals.

    Useful for debugging:

      - coordinate-space (u,v) ranges
      - line index ranges (k_min..k_max)
      - candidate line coordinates
      - which lines intersect the bbox
      - drop/jitter masks
      - final clipped segment counts
      - numerical edge cases

    Args:
        bbox:
            Bounding box in world coordinates.

        obliquity_deg, rotation_deg:
            Grid geometry parameters.

        x_major_step, x_minor_step, y_major_step, y_minor_step:
            Grid spacings.

        jitter:
            Optional jitter configuration.

        rng:
            Optional RNG for reproducible jitter.

        file:
            Optional file-like object to write output (default: sys.stdout).
    """
    if file is None:
        file = sys.stdout

    printf = lambda *a, **k: __builtins__["print"](*a, **k, file=file)

    printf("\n================ GRID DEBUG DUMP ================")
    printf(f"bbox = {bbox}")
    printf(f"obliquity_deg = {obliquity_deg}")
    printf(f"rotation_deg  = {rotation_deg}")
    printf(f"x_major_step  = {x_major_step}")
    printf(f"x_minor_step  = {x_minor_step}")
    printf(f"y_major_step  = {y_major_step}")
    printf(f"y_minor_step  = {y_minor_step}")
    printf(f"jitter        = {jitter}")

    # -----------------------------------------------------------
    # 1) Recompute u/v ranges and basis vectors exactly like main
    # -----------------------------------------------------------

    x_min, y_min, x_max, y_max = _parse_bbox(bbox)

    theta_rad = math.radians(rotation_deg)
    alpha_rad = math.radians(obliquity_deg)

    if rng is None:
        rng = random.Random()

    if jitter is not None and jitter.global_angle_deg > 0.0:
        theta_rad += math.radians(jitter.jitter_angle_global(rng))
        alpha_rad += math.radians(jitter.jitter_angle_global(rng))

    ex = np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=float)
    ey = np.array(
        [math.cos(theta_rad + alpha_rad), math.sin(theta_rad + alpha_rad)],
        dtype=float,
    )

    M = np.array([[ex[0], ey[0]], [ex[1], ey[1]]], dtype=float)
    Minv = np.linalg.inv(M)

    corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_min, y_max],
            [x_max, y_max],
    ], dtype=float)
    uv = corners @ Minv.T
    u_vals = uv[:, 0]
    v_vals = uv[:, 1]
    u_min, u_max = float(u_vals.min()), float(u_vals.max())
    v_min, v_max = float(v_vals.min()), float(v_vals.max())

    printf("\n--- Coordinate-space (u,v) ranges ---")
    printf(f"u_min={u_min:.6f}, u_max={u_max:.6f}")
    printf(f"v_min={v_min:.6f}, v_max={v_max:.6f}")

    # Helper to analyze a single line family
    def analyze_family(name, coord_min, coord_max, step, fixed_vec, dir_vec):
        printf(f"\n=== FAMILY: {name} ===")
        if step <= 0:
            printf("  step <= 0 - no lines")
            return

        k_min = int(math.floor(coord_min / step)) - 1
        k_max = int(math.ceil(coord_max / step)) + 1

        printf(f"  coord range: {coord_min:.3f} .. {coord_max:.3f}")
        printf(f"  step       : {step}")
        printf(f"  k_min={k_min}, k_max={k_max}, total candidates = {k_max-k_min+1}")

        # All candidate raw coordinates
        coords = np.arange(k_min, k_max + 1, dtype=float) * step
        printf(f"  coords (raw): {coords}")

        # Test intersection for each
        def intersects(c0):
            # same method _clip_infinite_line_to_bbox() uses
            p0 = c0 * fixed_vec
            d = dir_vec
            x0, y0 = p0
            dx, dy = d
            eps = 1e-12
            ts = []

            if abs(dx) > eps:
                for xb in (x_min, x_max):
                    t = (xb - x0) / dx
                    y = y0 + t * dy
                    if y_min - eps <= y <= y_max + eps:
                        ts.append(t)
            if abs(dy) > eps:
                for yb in (y_min, y_max):
                    t = (yb - y0) / dy
                    x = x0 + t * dx
                    if x_min - eps <= x <= x_max + eps:
                        ts.append(t)

            return len(ts) >= 2

        mask = [intersects(c) for c in coords]
        printf(f"  intersects: {mask}")
        printf(f"  kept coords: {[float(c) for c, m in zip(coords, mask) if m]}")

    # -----------------------------------------------------------
    # 2) Analyze each line family exactly as generator sees them
    # -----------------------------------------------------------
    analyze_family("X-MAJOR", u_min, u_max, x_major_step, ex, ey)
    analyze_family("X-MINOR", u_min, u_max, x_minor_step, ex, ey)
    analyze_family("Y-MAJOR", v_min, v_max, y_major_step, ey, ex)
    analyze_family("Y-MINOR", v_min, v_max, y_minor_step, ey, ex)

    printf("\n============== END DEBUG DUMP ==============\n")


# =============================================================================
# Internals
# =============================================================================


def _parse_bbox(bbox: BBoxLike) -> BBoxTuple:
    """Normalize bbox into the form (x_min, y_min, x_max, y_max)."""
    if (isinstance(bbox, Sequence)
        and len(bbox) == 2
        and isinstance(bbox[0], Sequence)):
        (x0, y0), (x1, y1) = bbox  # type: ignore[misc]
    elif isinstance(bbox, Sequence) and len(bbox) == 4:
        x0, y0, x1, y1 = bbox  # type: ignore[misc]
    else:
        raise ValueError(
            "bbox must be ((x_min, y_min), (x_max, y_max)) "
            "or (x_min, y_min, x_max, y_max)"
        )

    x_min = float(min(x0, x1))
    x_max = float(max(x0, x1))
    y_min = float(min(y0, y1))
    y_max = float(max(y0, y1))
    return x_min, y_min, x_max, y_max


def _build_line_family(
        coord_min        : float,
        coord_max        : float,
        step             : float,
        fixed_vec        : NDarray,
        dir_vec          : NDarray,
        bbox             : BBoxTuple,
        rng              : RNGBackend,
        jitter           : GridJitterConfig,
        base_offset_step : float,
    ) -> NDarray:
    """Generate clipped line segments for a single grid-line family.

    Args:
        coord_min:
            Minimum coordinate (u or v) over bbox corners.

        coord_max:
            Maximum coordinate (u or v) over bbox corners.

        step:
            Nominal spacing along that coordinate for this family
            (major or minor).

        fixed_vec:
            Unit vector along which the fixed coordinate is applied:
                - ex for X-family lines (u = const)
                - ey for Y-family lines (v = const)

        dir_vec:
            Nominal direction vector of the lines:
                - ey for X-family lines
                - ex for Y-family lines

        bbox:
            Axis-aligned bounding box in world coordinates.

        rng:
            Random number generator compatible with GridJitterConfig.

        jitter:
            Optional GridJitterConfig controlling offsets, orientations,
            and dropout.

        base_offset_step:
            Step used to convert dimensionless offset coefficients into
            absolute displacements.

    Returns:
        segments:
            NumPy array of shape (n_lines, 2, 2) containing line endpoints.
    """
    x_min, y_min, x_max, y_max = bbox

    if step <= 0.0:
        return np.empty((0, 2, 2), dtype=float)

    # Determine index range in coordinate space with a small padding
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
    # Per-group jitter/drop fractions
    # ---------------------------------------------------------------------------
    if jitter is not None:
        jitter_frac = jitter.jitter_offset_fraction(rng)
        drop_frac = jitter.jitter_drop_fraction(rng)
    else:
        jitter_frac = 0.0
        drop_frac = 0.0

    keep_mask = np.ones(n, dtype=bool)

    # Randomly mark lines to be dropped
    if drop_frac > 0.0:
        n_drop = int(round(drop_frac * n))
        n_drop = min(max(n_drop, 0), n)
        if n_drop > 0:
            drop_idx = np.random.default_rng().choice(
                indices, size=n_drop, replace=False
            )
            keep_mask[drop_idx] = False

    jitter_mask = np.zeros(n, dtype=bool)
    if jitter_frac > 0.0 and keep_mask.any():
        n_keep = int(keep_mask.sum())
        n_jitter = int(round(jitter_frac * n_keep))
        n_jitter = min(max(n_jitter, 0), n_keep)
        if n_jitter > 0:
            candidate_idx = indices[keep_mask]
            chosen = np.random.default_rng().choice(
                candidate_idx, size=n_jitter, replace=False
            )
            jitter_mask[chosen] = True

    # ---------------------------------------------------------------------------
    # Build and clip each line
    # ---------------------------------------------------------------------------
    base_phi = math.atan2(dir_vec[1], dir_vec[0])
    segments = []

    for i, c0 in enumerate(coord_values):
        if not keep_mask[i]:
            continue

        coord_val = c0
        phi = base_phi

        if jitter is not None and jitter_mask[i]:
            # Positional jitter along the fixed coordinate
            if jitter.line_offset_factor > 0.0 and base_offset_step > 0.0:
                offset_coeff = jitter.jitter_line_offset(rng)
                coord_val += offset_coeff * base_offset_step

            # Orientation jitter
            if jitter.line_angle_deg > 0.0:
                phi += math.radians(jitter.jitter_line_angle(rng))

        d = np.array([math.cos(phi), math.sin(phi)], dtype=float)
        p0 = coord_val * fixed_vec

        seg = _clip_infinite_line_to_bbox(p0, d, x_min, y_min, x_max, y_max)
        if seg is not None:
            segments.append(seg)

    if not segments:
        return np.empty((0, 2, 2), dtype=float)

    return np.stack(segments, axis=0)


def _clip_infinite_line_to_bbox(
        p0    : NDarray,
        d     : NDarray,
        x_min : float,
        y_min : float,
        x_max : float,
        y_max : float,
    ) -> NDarray:
    """Clip an infinite line to an axis-aligned bounding box.

    The infinite line is given by p(t) = p0 + t * d.

    Args:
        p0:
            Base point on the line.

        d:
            Direction vector of the line.

        x_min, y_min, x_max, y_max:
            Bounding box limits.

    Returns:
        A (2, 2) array [[x1, y1], [x2, y2]] representing the clipped segment,
        or None if the line does not intersect the bbox.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(d[0]), float(d[1])
    eps = 1e-12

    ts = []

    # Intersections with x = constant
    if abs(dx) > eps:
        for xb in (x_min, x_max):
            t = (xb - x0) / dx
            y = y0 + t * dy
            if y_min - eps <= y <= y_max + eps:
                ts.append(t)

    # Intersections with y = constant
    if abs(dy) > eps:
        for yb in (y_min, y_max):
            t = (yb - y0) / dy
            x = x0 + t * dx
            if x_min - eps <= x <= x_max + eps:
                ts.append(t)

    if len(ts) < 2:
        return None

    ts_sorted = sorted(ts)
    t1, t2 = ts_sorted[0], ts_sorted[-1]

    p1 = np.array([x0 + t1 * dx, y0 + t1 * dy], dtype=float)
    p2 = np.array([x0 + t2 * dx, y0 + t2 * dy], dtype=float)
    return np.stack([p1, p2], axis=0)

