"""
pet_geom.py
-----------

Utilities for grid geometry extraction:

Public API:
    detect_grid_segments(img)              -> raw LSD segments
    filter_grid_segments(raw, tol_deg)     -> two filtered segment families
    estimate_vanishing_points(lines_x, ...) 
    refine_principal_point_from_vps(vp_x, vp_y, img_shape)
    separate_line_families_kmeans(lines)   -> simple two-angle clustering
    mark_segments(img, segments)
    mark_segment_families(img, lines_x, lines_y)
"""

from __future__ import annotations
import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional

from scipy.signal import find_peaks
from scipy import stats

import matplotlib.pyplot as plt


# ============================================================================
# 1) RAW LSD SEGMENT EXTRACTION
# ============================================================================

def _create_lsd():
    try:
        # OpenCV 4.5+ signature
        return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except TypeError:
        try:
            # OpenCV 4.7+ signature
            return cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
        except TypeError:
            # Final fallback: manual refine removal
            return cv2.createLineSegmentDetector()


def detect_grid_segments(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run LSD (Line Segment Detector) and return raw segments
    with full per-segment information.

    Returns
    -------
    dict with:
        "lines"      : (N,4) float32 [x1,y1,x2,y2]
        "widths"     : (N,)  float32
        "precisions" : (N,)  float32
        "nfa"        : (N,)  float32
        "lengths"    : (N,)  float32
        "centers"    : (N,2) float32 [xc,yc]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = _create_lsd()
    lines, widths, prec, nfa = lsd.detect(gray)

    # ---------------------------------------
    # Empty detector output
    # ---------------------------------------
    if lines is None:
        return {
            "lines": np.zeros((0, 4), np.float32),
            "widths": np.zeros((0,), np.float32),
            "precisions": np.zeros((0,), np.float32),
            "nfa": np.zeros((0,), np.float32),
            "lengths": np.zeros((0,), np.float32),
            "centers": np.zeros((0,2), np.float32),
        }

    # ---------------------------------------
    # Normalize shapes + dtypes
    # ---------------------------------------
    lines = lines.reshape(-1, 4).astype(np.float32)
    N = lines.shape[0]

    widths = (
        widths.reshape(-1).astype(np.float32)
        if widths is not None else np.zeros(N, np.float32)
    )
    prec = (
        prec.reshape(-1).astype(np.float32)
        if prec is not None else np.zeros(N, np.float32)
    )
    nfa = (
        nfa.reshape(-1).astype(np.float32)
        if nfa is not None else np.zeros(N, np.float32)
    )

    # ---------------------------------------
    # Geometric properties
    # ---------------------------------------
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    dx = x2 - x1
    dy = y2 - y1

    lengths = np.hypot(dx, dy).astype(np.float32)

    # midpoints
    xc = (x1 + x2) * 0.5
    yc = (y1 + y2) * 0.5
    centers = np.stack([xc, yc], axis=1).astype(np.float32)

    return {
        "lines": lines,
        "widths": widths,
        "precisions": prec,
        "nfa": nfa,
        "lengths": lengths,
        "centers": centers,
    }


def detect_grid_segments_full(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Full LSD extraction wrapper.
    Safely extracts:
        - line segments
        - widths
        - precisions
        - nfa
        - lengths
        - centers

    Returns a dictionary with guaranteed shape consistency:
        {
            "lines":      (N,4) float32  [x1,y1,x2,y2]
            "widths":     (N,)  float32
            "precisions": (N,)  float32
            "nfa":        (N,)  float32
            "lengths":    (N,)  float32
            "centers":    (N,2) float32
        }
    """

    # -------------------------------------------------------
    # 1) Convert to grayscale
    # -------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------
    # 2) Create LSD with proper compatibility hacks
    # -------------------------------------------------------
    def _create_lsd():
        """Robustly create LSD across OpenCV versions."""
        try:
            # OpenCV >= 4.5
            return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            try:
                # OpenCV build that uses named argument
                return cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
            except Exception:
                # Very old or non-standard build
                return cv2.createLineSegmentDetector()

    lsd = _create_lsd()

    # -------------------------------------------------------
    # 3) Run LSD detection (may return missing fields!)
    # -------------------------------------------------------
    lines, widths, precisions, nfa = lsd.detect(gray)

    # LSD returns None if no segments detected
    if lines is None:
        return {
            "lines":      np.zeros((0,4), np.float32),
            "widths":     np.zeros((0,),  np.float32),
            "precisions": np.zeros((0,),  np.float32),
            "nfa":        np.zeros((0,),  np.float32),
            "lengths":    np.zeros((0,),  np.float32),
            "centers":    np.zeros((0,2), np.float32),
        }

    # -------------------------------------------------------
    # 4) Normalize shapes
    # -------------------------------------------------------
    lines = lines.reshape(-1, 4).astype(np.float32)
    N = lines.shape[0]

    def _safe_vec(v):
        """Normalize LSD side outputs which may be None."""
        if v is None:
            return np.zeros(N, np.float32)
        v = np.asarray(v).reshape(-1)
        if v.size != N:
            # Some OpenCV builds return broken shapes
            v = np.zeros(N, np.float32)
        return v.astype(np.float32)

    widths     = _safe_vec(widths)
    precisions = _safe_vec(precisions)
    nfa        = _safe_vec(nfa)

    # -------------------------------------------------------
    # 5) Compute geometric properties
    # -------------------------------------------------------
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    dx = x2 - x1
    dy = y2 - y1

    lengths = np.hypot(dx, dy).astype(np.float32)
    centers = np.column_stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5]).astype(np.float32)

    # -------------------------------------------------------
    # 6) Return full dictionary
    # -------------------------------------------------------
    return {
        "lines":      lines,
        "widths":     widths,
        "precisions": precisions,
        "nfa":        nfa,
        "lengths":    lengths,
        "centers":    centers,
    }


def clamp_segment_length(
    raw_lsd: Dict[str, np.ndarray],
    min_len: float = 0.0,
    max_len: float = float("inf"),
    width_percentile: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Filter LSD output by:
        1. Segment length limits: min_len <= length <= max_len
        2. Width percentile: remove widths above percentile(widths, width_percentile)

    Percentile filter applies to **widths only**, not lengths.

    Input/output fields:
        "lines", "widths", "precisions", "nfa",
        "lengths", "centers"
    """

    required_keys = ["lines", "widths", "precisions", "nfa", "lengths", "centers"]
    for k in required_keys:
        if k not in raw_lsd:
            raise KeyError(f"raw_lsd missing required key '{k}'")

    lines      = np.asarray(raw_lsd["lines"])
    widths     = np.asarray(raw_lsd["widths"])
    precisions = np.asarray(raw_lsd["precisions"])
    nfa        = np.asarray(raw_lsd["nfa"])
    lengths    = np.asarray(raw_lsd["lengths"])
    centers    = np.asarray(raw_lsd["centers"])

    if lines.size == 0:
        return {
            "lines":      np.zeros((0,4), np.float32),
            "widths":     np.zeros((0,),  np.float32),
            "precisions": np.zeros((0,),  np.float32),
            "nfa":        np.zeros((0,),  np.float32),
            "lengths":    np.zeros((0,),  np.float32),
            "centers":    np.zeros((0,2), np.float32),
        }

    # ---- validate shapes ----
    N = lines.shape[0]
    for arr, name in [
        (widths, "widths"),
        (precisions, "precisions"),
        (nfa, "nfa"),
        (lengths, "lengths"),
        (centers, "centers"),
    ]:
        if arr.shape[0] != N:
            raise ValueError(
                f"'lines' has {N} entries but '{name}' has {arr.shape[0]}"
            )

    # ---- compute width threshold ----
    width_percentile = float(width_percentile)
    if not (0 <= width_percentile <= 100):
        raise ValueError("width_percentile must be between 0 and 100")

    width_threshold = np.percentile(widths, width_percentile)

    # ---- combined mask (LENGTH + WIDTH) ----
    mask = (
        (lengths >= min_len) &
        (lengths <= max_len) &
        (widths  <= width_threshold)
    )

    # ---- return filtered structure ----
    return {
        "lines":      lines[mask].astype(np.float32),
        "widths":     widths[mask].astype(np.float32),
        "precisions": precisions[mask].astype(np.float32),
        "nfa":        nfa[mask].astype(np.float32),
        "lengths":    lengths[mask].astype(np.float32),
        "centers":    centers[mask].astype(np.float32),
    }


# ============================================================================
# 2) ANGLE UTILITIES
# ============================================================================

def _line_angle_rad(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _line_angle_deg(seg: np.ndarray):
    x1, y1, x2, y2 = seg
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    ang = ang % 180.0
    return ang


def _average_direction_angle(lines: np.ndarray) -> Optional[float]:
    """
    Compute a robust average direction angle (in radians) for a family
    of line segments, using the median of individual segment angles.
    """
    if lines is None or len(lines) == 0:
        return None
    dx = lines[:,2] - lines[:,0]
    dy = lines[:,3] - lines[:,1]
    angles = np.arctan2(dy, dx).astype(float)
    
    return float(np.median(np.mod(angles, np.pi)))


def _line_from_points(p1: Tuple[float, float],
                      p2: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
    """
    Homogeneous line through two points p1=(x1,y1), p2=(x2,y2) in ax+by+c=0 form.

    Returns (a,b,c) or None if points are degenerate.
    """
    (x1, y1), (x2, y2) = p1, p2
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    # Normal is perpendicular to direction
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    return float(a), float(b), float(c)


def _ang_dist_circular_deg(a: np.ndarray,
                           b: np.ndarray,
                           period: float = 180.0) -> np.ndarray:
    """
    Smallest absolute distance between angles a, b on a circle (degrees).
    a and b can be arrays/broadcastable.
    """
    return np.abs(((a - b + 0.5 * period) % period) - 0.5 * period)


def _ang_diff_signed_deg(a: np.ndarray,
                         b: float,
                         period: float = 180.0) -> np.ndarray:
    """
    Signed minimal difference a - b on a circle (degrees), in (-period/2, period/2].
    """
    return ((a - b + 0.5 * period) % period) - 0.5 * period


def compute_segment_angles(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute per-segment angles and basic quality metrics from LSD output.

    Args:
        raw: Dictionary returned by detect_grid_segments(), with keys:
             'lines', 'widths', 'precisions', 'nfa'.

    Returns:
        dict with:
            'angles_rad': (N,) angles in radians, normalized to [-pi/4, 3pi/4)
            'angles_deg': (N,) same in degrees, [-45, 135)
            'lengths':    (N,) segment lengths in pixels
            'widths':     (N,) LSD-reported widths (or zeros if missing)
            'precisions': (N,) LSD-reported precisions (or zeros if missing)
            'nfa':        (N,) LSD-reported NFA values (or zeros if missing)
            'quality':    (N,) heuristic quality weight ~ length * precision
    """
    lines = raw.get("lines")
    if lines is None or len(lines) == 0:
        return {
            "angles_rad": np.zeros((0,), dtype=np.float64),
            "angles_deg": np.zeros((0,), dtype=np.float64),
            "lengths":    np.zeros((0,), dtype=np.float64),
            "widths":     np.zeros((0,), dtype=np.float64),
            "precisions": np.zeros((0,), dtype=np.float64),
            "nfa":        np.zeros((0,), dtype=np.float64),
            "quality":    np.zeros((0,), dtype=np.float64),
        }

    lines = np.asarray(lines, dtype=np.float64)
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    # ------------------------------------------------------------
    # Compute CCW angles properly (image coordinates -> math coords)
    # ------------------------------------------------------------
    # Image: y increases downward
    # Math:  y increases upward
    # Convert by flipping dy: dy_math = -(y2 - y1)
    dx = x2 - x1
    dy_image = y2 - y1
    dy = -dy_image

    # Raw CCW angle in (-pi, pi]
    angles = np.arctan2(dy, dx)

    # Normalize CCW angle into [-pi/4, 3pi/4) = [-45deg, 135deg)
    angles_norm = (angles + np.pi/4) % np.pi - np.pi/4
    angles_deg = np.rad2deg(angles_norm)

    lengths = np.hypot(dx, dy)

    N = lines.shape[0]

    def _safe_meta(key: str) -> np.ndarray:
        arr = raw.get(key)
        if arr is None:
            return np.zeros(N, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.shape[0] != N:
            # Be conservative: mismatch -> zero-fill
            return np.zeros(N, dtype=np.float64)
        return arr

    widths = _safe_meta("widths")
    precisions = _safe_meta("precisions")
    nfa = _safe_meta("nfa")

    # Simple heuristic quality weight: longer + higher precision = better
    quality = lengths * (precisions + 1e-6)

    return {
        "angles_rad": angles_norm.astype(np.float64),
        "angles_deg": angles_deg.astype(np.float64),
        "lengths":    lengths.astype(np.float64),
        "widths":     widths,
        "precisions": precisions,
        "nfa":        nfa,
        "quality":    quality.astype(np.float64),
    }


def compute_segment_lengths(angle_info: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Add per-segment length into angle_info.

    Expected input keys:
        - Either:
            angle_info["lines"]  -> (N,4) array [x1,y1,x2,y2]
        - Or:
            angle_info["x1"], angle_info["y1"], angle_info["x2"], angle_info["y2"]
          (rare but allowed)

    Output:
        angle_info["lengths"] : array (N,) of Euclidean segment lengths

    Notes:
        - Fully vectorized.
        - Invalid / missing lines -> length = 0.
        - Does not alter any existing fields.
    """
    # ------------------------------------------------------------------
    # Case 1: full line array exists
    # ------------------------------------------------------------------
    if "lines" in angle_info:
        lines = np.asarray(angle_info["lines"], float)
        if lines.ndim == 2 and lines.shape[1] == 4 and len(lines) > 0:
            dx = lines[:, 2] - lines[:, 0]
            dy = lines[:, 3] - lines[:, 1]
            lengths = np.hypot(dx, dy)
        else:
            # invalid shape
            lengths = np.zeros(0, dtype=float)

    # ------------------------------------------------------------------
    # Case 2: separate coordinate arrays
    # ------------------------------------------------------------------
    elif all(k in angle_info for k in ("x1", "y1", "x2", "y2")):
        x1 = np.asarray(angle_info["x1"], float)
        y1 = np.asarray(angle_info["y1"], float)
        x2 = np.asarray(angle_info["x2"], float)
        y2 = np.asarray(angle_info["y2"], float)
        if x1.shape == x2.shape == y1.shape == y2.shape:
            lengths = np.hypot(x2 - x1, y2 - y1)
        else:
            lengths = np.zeros_like(x1, dtype=float)

    else:
        # No geometry available -> zero lengths
        N = len(angle_info.get("angles_deg", []))
        lengths = np.zeros(N, dtype=float)

    # Clean invalids
    lengths = np.where(np.isfinite(lengths), lengths, 0.0)

    # Attach to dict and return
    angle_info["lengths"] = lengths
    return angle_info


def compute_angle_histogram(
    angle_info: Dict[str, np.ndarray],
    bins: int = 90,
    angle_range: Tuple[float, float] = (-45.0, 135.0)
) -> Dict[str, np.ndarray]:
    """
    Compute histogram counts for angle data (in degrees).

    Args:
        angle_info: Dictionary returned by compute_segment_angles().
        bins:       Number of histogram bins.
        angle_range: (min_deg, max_deg) range for the histogram.

    Returns:
        {
            "counts"     : (bins,) histogram counts,
            "bin_edges"  : (bins+1,) edges,
            "bin_centers": (bins,) midpoints of each bin,
            "range"      : (min_deg, max_deg),
        }
    """
    angles_deg = np.asarray(angle_info.get("angles_deg", []), dtype=np.float64)

    if angles_deg.size == 0:
        # Return empty histogram in consistent format
        edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {
            "counts":     np.zeros(bins, dtype=np.int64),
            "bin_edges":  edges,
            "bin_centers": centers,
            "range":      angle_range,
        }

    counts, edges = np.histogram(angles_deg, bins=bins, range=angle_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "counts":     counts.astype(np.int64),
        "bin_edges":  edges,
        "bin_centers": centers,
        "range":      angle_range,
    }


def compute_angle_histogram_circular_weighted(
    angle_info: Dict[str, np.ndarray],
    bins: int = 90,
    angle_range: Tuple[float, float] = (-45.0, 135.0),
    kde_kappa: float = 20.0,       # concentration of von Mises kernel
    kde_samples: int = 720,        # resolution of KDE curve
    normalize_weights: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute weighted circular histogram and circular KDE from angle_info.

    angle_info must include:
        angles_deg : array of angles (any range)
        weights    : optional per-segment weights

    Returns a dictionary containing:
        angles_deg, weights, range
        hist_x, hist_y
        kde_x, kde_y
        raw_count, raw_weight
        period
    """
    angles_deg = np.asarray(angle_info["angles_deg"], dtype=float)
    weights = angle_info.get("weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != angles_deg.shape:
            raise ValueError("weights must match angles shape")
    else:
        weights = np.ones_like(angles_deg, float)

    # Normalize to sum=1 if requested
    if normalize_weights:
        total_w = float(np.sum(weights))
        if total_w > 0:
            weights = weights / total_w
        else:
            weights = np.ones_like(weights, float) / weights.size
    
    lo, hi = angle_range
    period = hi - lo

    # Normalize angles into [lo, hi)
    ang = ((angles_deg - lo) % period) + lo

    # Raw diagnostics
    raw_count = ang.size
    raw_weight = float(np.sum(weights))

    # Histogram
    hist_y, hist_edges = np.histogram(
        ang,
        bins=bins,
        range=(lo, hi),
        weights=weights,
    )
    hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    # Circular KDE (wrapped von Mises kernel)
    kde_x = np.linspace(lo, hi, kde_samples, endpoint=False)
    kde_x_rad = np.deg2rad(kde_x)
    ang_rad = np.deg2rad(ang)

    dx = 2 * np.pi / kde_samples
    kde = np.zeros_like(kde_x_rad)

    for th, w in zip(ang_rad, weights):
        kde += w * np.exp(kde_kappa * np.cos(kde_x_rad - th))

    kde /= (2 * np.pi * np.i0(kde_kappa))

    return {
        "angles_deg": ang,
        "weights": weights,
        "range": (lo, hi),
        "period": period,
        "hist_x": hist_x,
        "hist_y": hist_y,
        "kde_x": kde_x,
        "kde_y": kde,
        "raw_count": raw_count,
        "raw_weight": raw_weight,
    }


def compute_family_kdes(
    hist_info: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    kde_samples: int = 720
) -> Dict[str, np.ndarray]:
    """
    Recompute KDE per-family using wrapped Gaussian kernels with
    automatic bandwidth selection (weighted Silverman rule).

    Input:
        hist_info  - from compute_angle_histogram_circular_weighted()
        analysis   - from analyze_two_orientation_families()
    
    Output (same fixed format as compute_angle_histogram_circular_weighted):
        {
            "angles_deg": ang_all,
            "weights": w_all,
            "range": (lo, hi),
            "period": period,
            "hist_x": hist_x,
            "hist_y": hist_y,
            "kde_x": kde_x,
            "kde_y": kde_combined,
            "raw_count": N,
            "raw_weight": total_weight,

            # per-family extras
            "fam1": {
                "angles": ang1,
                "weights": w1,
                "kde_y": kde1,
                "bw": bw1,
                "skewness": skew1,
                "kurtosis": kurt1,
            },
            "fam2": {
                "angles": ang2,
                "weights": w2,
                "kde_y": kde2,
                "bw": bw2,
                "skewness": skew2,
                "kurtosis": kurt2,
            },
        }
    """

    # ------------------------------------------------------------------
    # Extract global info
    # ------------------------------------------------------------------
    lo, hi = hist_info["range"]
    period = hi - lo

    ang_all = np.asarray(hist_info["angles_deg"], float)
    w_all   = np.asarray(hist_info["weights"], float)
    N       = ang_all.size
    total_weight = float(np.sum(w_all))

    # ------------------------------------------------------------------
    # Split into two families according to analysis masks
    # (these are the same masks used in analyze_two_orientation_families)
    # ------------------------------------------------------------------
    p1, p2 = analysis["peaks"]
    d1 = _ang_dist_circular_deg(ang_all, p1, period)
    d2 = _ang_dist_circular_deg(ang_all, p2, period)
    mask1 = d1 <= d2
    mask2 = ~mask1

    ang1 = ang_all[mask1]
    ang2 = ang_all[mask2]
    w1   = w_all[mask1]
    w2   = w_all[mask2]

    # Normalize to [lo, hi)
    ang1 = ((ang1 - lo) % period) + lo
    ang2 = ((ang2 - lo) % period) + lo

    # ------------------------------------------------------------------
    # KDE grid
    # ------------------------------------------------------------------
    kde_x = np.linspace(lo, hi, kde_samples, endpoint=False)
    dx_deg = float(kde_x[1] - kde_x[0])  # uniform grid in degrees

    # ------------------------------------------------------------------
    # Wrapped Gaussian kernel KDE
    #
    # KDE(theta) = sum w_i * sum_{k=-inf..inf} exp(-(theta - a_i + k*period)^2/(2*bw^2))
    # For practical purposes only k = [-1,0,1] needed.
    # ------------------------------------------------------------------
    def wrapped_gaussian_kde(angles, weights, bw):
        if angles.size == 0 or np.sum(weights) == 0:
            return np.zeros_like(kde_x)

        var = (bw * bw * 2.0)
        kde = np.zeros_like(kde_x)
        for a, w in zip(angles, weights):
            delta0 = kde_x - a
            delta_p = delta0 + period
            delta_m = delta0 - period
            kde += w * (
                np.exp(-(delta0 * delta0) / var) +
                np.exp(-(delta_p * delta_p) / var) +
                np.exp(-(delta_m * delta_m) / var)
            )
        # Normalize so integral = Sum weights
        norm = np.sum(weights) * np.sqrt(2 * np.pi) * bw
        return kde / norm

    # ------------------------------------------------------------------
    # Bandwidth selection (weighted Silverman rule)
    #
    # bw = 0.9 * min(std, IQR/1.34) * N^{-1/5}
    # Periodicity is ignored for bandwidth estimation (standard practice).
    # ------------------------------------------------------------------
    def weighted_stats(a, w):
        if a.size == 0:
            return 0.0, 0.0
        m = np.sum(a * w) / np.sum(w)
        var = np.sum(w * (a - m)**2) / np.sum(w)
        std = np.sqrt(var)
        # percentiles (unweighted)
        q25 = np.percentile(a, 25)
        q75 = np.percentile(a, 75)
        IQR = q75 - q25
        return std, IQR

    def silverman_bw(angles, weights):
        if angles.size < 2:
            return 5.0  # fall-back 5 degrees
        std, IQR = weighted_stats(angles, weights)
        s = min(std, IQR / 1.34)
        Nw = np.sum(weights)
        if Nw <= 0:
            return 5.0
        return 0.9 * s * (Nw ** (-0.2))  # N^(-1/5)

    bw1 = silverman_bw(ang1, w1)
    bw2 = silverman_bw(ang2, w2)

    # ------------------------------------------------------------------
    # Compute per-family KDEs
    # ------------------------------------------------------------------
    kde1 = wrapped_gaussian_kde(ang1, w1, bw1)
    kde2 = wrapped_gaussian_kde(ang2, w2, bw2)

    # Combined KDE = sum
    kde_combined = kde1 + kde2

    # ------------------------------------------------------------------
    # Compute histograms again (weighted)
    # ------------------------------------------------------------------
    hist_y, hist_edges = np.histogram(
        ang_all,
        bins=hist_info["hist_x"].shape[0],
        range=(lo, hi),
        weights=w_all
    )
    hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    # ------------------------------------------------------------------
    # Skewness and kurtosis (linear, weighted)
    # NOTE: circular skewness/kurtosis exists but users typically want
    #       classical moment-based ones for small spreads (< 10 deg).
    # ------------------------------------------------------------------
    def weighted_moments(a, w):
        if a.size == 0:
            return 0.0, 0.0
        W = np.sum(w)
        mu = np.sum(a * w) / W
        c = a - mu
        m2 = np.sum(w * c**2) / W
        m3 = np.sum(w * c**3) / W
        m4 = np.sum(w * c**4) / W
        if m2 <= 1e-12:
            return 0.0, -3.0  # degenerate
        skew = m3 / (m2 ** 1.5)
        kurt = m4 / (m2 * m2) - 3.0
        return skew, kurt

    skew1, kurt1 = weighted_moments(ang1, w1)
    skew2, kurt2 = weighted_moments(ang2, w2)

    # ------------------------------------------------------------------
    # Output (same format as compute_angle_histogram_circular_weighted)
    # ------------------------------------------------------------------
    return {
        "angles_deg": ang_all,
        "weights": w_all,
        "range": (lo, hi),
        "period": period,
        "hist_x": hist_x,
        "hist_y": hist_y,
        "kde_x": kde_x,
        "kde_y": kde_combined,
        "raw_count": N,
        "raw_weight": total_weight,

        "fam1": {
            "angles": ang1,
            "weights": w1,
            "kde_y": kde1,
            "bw": bw1,
            "skewness": skew1,
            "kurtosis": kurt1,
        },
        "fam2": {
            "angles": ang2,
            "weights": w2,
            "kde_y": kde2,
            "bw": bw2,
            "skewness": skew2,
            "kurtosis": kurt2,
        },
    }


def _find_two_kde_peaks_circular(
    hist_info: Dict[str, np.ndarray],
    peak_prominence: float = 0.05,
    min_peak_distance_deg: float = 20.0,
) -> Tuple[float, float]:
    """
    Find exactly two dominant KDE peaks on a circular domain.

    Returns:
        (p1_deg, p2_deg) in degrees, ordered, in [lo, hi) of hist_info["range"].

    Raises:
        ValueError if less or more than two peaks are found.
    """
    kde_x = np.asarray(hist_info["kde_x"], dtype=float)
    kde_y = np.asarray(hist_info["kde_y"], dtype=float)

    lo, hi = hist_info.get("range", (kde_x.min(), kde_x.max()))
    period = hi - lo

    if kde_x.ndim != 1 or kde_y.ndim != 1 or kde_x.size != kde_y.size:
        raise ValueError("Invalid KDE arrays in hist_info.")

    if kde_x.size < 4:
        raise ValueError("KDE resolution too low to detect peaks.")

    # Step size in degrees (assume roughly uniform grid)
    dx = float(np.mean(np.diff(kde_x)))
    if dx <= 0:
        raise ValueError("Non-monotonic KDE X axis.")

    # Wrap KDE 3 times to avoid edge issues
    x_ext = np.concatenate([kde_x,
                            kde_x + period,
                            kde_x + 2.0 * period])
    y_ext = np.tile(kde_y, 3)

    ymax = float(y_ext.max())
    if ymax <= 0.0:
        raise ValueError("KDE is flat; cannot find peaks.")

    prominence_abs = peak_prominence * ymax
    distance_pts = max(1, int(min_peak_distance_deg / dx))

    peak_idx, props = find_peaks(
        y_ext,
        prominence=prominence_abs,
        distance=distance_pts,
    )

    if peak_idx.size == 0:
        raise ValueError("No significant KDE peaks found.")

    # Map peak positions back to base interval [lo, hi)
    peak_angles = ((x_ext[peak_idx] - lo) % period) + lo
    peak_heights = y_ext[peak_idx]

    # Sort by angle
    order = np.argsort(peak_angles)
    peak_angles = peak_angles[order]
    peak_heights = peak_heights[order]

    # Merge peaks that are very close (within a few degrees)
    merged: list[float] = []
    merged_h: list[float] = []
    tol = 5.0  # degrees

    for ang, h in zip(peak_angles, peak_heights):
        if not merged:
            merged.append(float(ang))
            merged_h.append(float(h))
            continue

        # Distance to last merged peak on circle
        d = _ang_dist_circular_deg(ang, merged[-1], period=period)
        if d < tol:
            # Keep the higher peak of the two
            if h > merged_h[-1]:
                merged[-1] = float(ang)
                merged_h[-1] = float(h)
        else:
            merged.append(float(ang))
            merged_h.append(float(h))

    if len(merged) != 2:
        raise ValueError(f"Expected exactly 2 dominant peaks, found {len(merged)}.")

    p1, p2 = merged
    # Order them consistently on the circle:
    # we just ensure p1 < p2 in the base interval [lo, hi)
    if p2 < p1:
        p1, p2 = p2, p1

    return float(p1), float(p2)


def _circular_family_stats(
    angles_deg: np.ndarray,
    weights: np.ndarray,
    angle_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Compute circular statistics for a family of orientations.
    Added:
        - circular skewness
        - circular kurtosis
        - kde_kappa  (KDE bandwidth; identical to fitted kappa)
    """

    angles_deg = np.asarray(angles_deg, float)
    weights    = np.asarray(weights, float)

    count = int(angles_deg.size)
    total_weight = float(weights.sum())

    if count == 0 or total_weight <= 0:
        return {
            "count": 0,
            "total_weight": 0.0,
            "mean_deg": None,
            "R": None,
            "circ_var": None,
            "kappa": None,
            "kde_kappa": None,
            "skewness": None,
            "kurtosis": None,
        }

    # Weighted normalization
    w_norm = weights / total_weight
    ang_rad = np.deg2rad(angles_deg)

    # Weighted vector sum
    C = np.sum(w_norm * np.cos(ang_rad))
    S = np.sum(w_norm * np.sin(ang_rad))
    R = float(np.hypot(C, S))

    # Mean direction
    mean_rad = float(np.arctan2(S, C))
    mean_deg = float(np.rad2deg(mean_rad))

    # Circular variance
    circ_var = 1.0 - R

    # --- Kappa estimator (existing logic unchanged) ---
    if R < 1e-6:
        kappa = 0.0
    elif R < 0.53:
        kappa = 2 * R + R**3 + 5*(R**5)/6.0
    elif R < 0.85:
        kappa = -0.4 + 1.39*R + 0.43/(1 - R)
    else:
        kappa = 1.0 / (R**3 - 4*R**2 + 3*R)
    kappa = float(kappa)

    # --- KDE bandwidth autoselection ---
    kde_kappa = kappa   # simply reuse kappa per-family

    # --- Circular skewness ---
    d = ang_rad - mean_rad
    skew = float(np.sum(w_norm * np.sin(d)))

    # --- Circular kurtosis ---
    kurt = float(np.sum(w_norm * np.cos(2.0 * d)))

    return {
        "count": count,
        "total_weight": total_weight,
        "mean_deg": mean_deg,
        "R": R,
        "circ_var": circ_var,
        "kappa": kappa,
        "kde_kappa": kde_kappa,   # new
        "skewness": skew,         # new
        "kurtosis": kurt,         # new
    }


def analyze_two_orientation_families(
    angle_hist: Dict[str, np.ndarray],
    peak_prominence: float = 0.05,
    min_peak_distance_deg: float = 20.0,
) -> Dict[str, Any]:

    # 1) Find the dominant peaks
    p1, p2 = _find_two_kde_peaks_circular(
        angle_hist,
        peak_prominence=peak_prominence,
        min_peak_distance_deg=min_peak_distance_deg,
    )

    lo, hi = angle_hist["range"]
    period = hi - lo

    # 2) Mid-split angle
    diff_p = _ang_diff_signed_deg(p2, p1, period=period)
    split_deg = float(((p1 + 0.5 * diff_p) - lo) % period + lo)

    # 3) Family assignment
    ang = angle_hist["angles_deg"]
    w   = angle_hist["weights"]

    d1 = _ang_dist_circular_deg(ang, p1, period=period)
    d2 = _ang_dist_circular_deg(ang, p2, period=period)

    mask1 = d1 <= d2
    mask2 = ~mask1

    ang1, ang2 = ang[mask1], ang[mask2]
    w1,   w2   = w[mask1],   w[mask2]

    # 4) Stats (extended)
    stats1 = _circular_family_stats(ang1, w1, angle_range=(lo, hi))
    stats2 = _circular_family_stats(ang2, w2, angle_range=(lo, hi))

    # 5) Rotation angle based on PEAKS (correct)
    # Determine dominant family by weight
    if stats1["total_weight"] >= stats2["total_weight"]:
        dom_peak = p1
    else:
        dom_peak = p2
    
    # Normalize dominant peak into [-45, 135)
    peak_n = ((dom_peak - lo) % period) + lo
    
    # Desired targets: 0deg (horizontal) or 90deg (vertical)
    # rotation = target - peak
    rot_h = (0.0  - peak_n + 90.0) % 180.0 - 90.0
    rot_v = (90.0 - peak_n + 90.0) % 180.0 - 90.0
    
    # Choose the smallest magnitude
    rotation_angle = rot_h if abs(rot_h) <= abs(rot_v) else rot_v
    
    # Clamp into [-45, +45] while preserving orientation correctness
    if rotation_angle < -45.0:
        rotation_angle += 90.0
    elif rotation_angle > 45.0:
        rotation_angle -= 90.0

    return {
        "peaks": [p1, p2],
        "split_deg": split_deg,
        "family1": stats1,
        "family2": stats2,
        "rotation_angle_deg": float(rotation_angle),

        # NEW: expose per-family KDE kappa
        "kde_kappa_1": stats1["kde_kappa"],
        "kde_kappa_2": stats2["kde_kappa"],
    }


def apply_rotation_correction(
    img: np.ndarray,
    analysis: Dict[str, Any],
    border_value: Tuple[int, int, int] | int = (255, 255, 255),
) -> np.ndarray:
    """
    Apply compensating rotation to an image based on angle-analysis result.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR uint8 or RGB uint8 or float32).
    analysis : dict
        Output dict from analyze_two_orientation_families().
        Must contain key: "rotation_angle_deg".
        This angle is the correction to *apply* so that the
        dominant family becomes horizontal/vertical.
    border_value : int or (3-tuple)
        Border color used when rotating. Default white=255.

    Returns
    -------
    rotated : np.ndarray
        Deskewed image of the same dtype.
    """
    if "rotation_angle_deg" not in analysis:
        raise KeyError("analysis must contain 'rotation_angle_deg'")

    angle = float(analysis["rotation_angle_deg"])
    if abs(angle) < 1e-12:
        return img.copy()  # no rotation needed

    (h, w) = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Warp - preserve dtype
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def reassign_and_rotate_families_by_image_center(
    families: Dict[str, Dict[str, np.ndarray]],
    analysis: Dict[str, Any],
    img: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Reassign families (xfam/yfam) based on median angle,
    then rotate all segment endpoints, centers, and angles
    around the *image center*.

    The returned structure contains the same fields as input families:
        "lines", "widths", "precisions", "nfa",
        "lengths", "centers", "angles"

    After rotation, centers are recomputed from rotated segments,
    and angles are adjusted by subtracting the rotation angle.

    Output:
        {
            "xfam": {... full rotated dict ...},
            "yfam": {... full rotated dict ...},
            "xfam_angle": median_angle,
            "yfam_angle": median_angle,
            "pivot": (cx, cy)
        }
    """

    # -----------------------------
    # Extract family1 / family2
    # -----------------------------
    f1 = families["family1"]
    f2 = families["family2"]

    lines1 = np.asarray(f1["lines"], float)
    lines2 = np.asarray(f2["lines"], float)
    ang1   = np.asarray(f1["angles"], float)
    ang2   = np.asarray(f2["angles"], float)

    # Empty case
    if lines1.size == 0 and lines2.size == 0:
        empty = {k: np.zeros((0,), float) for k in
                 ["widths","precisions","nfa","lengths","angles"]}
        empty["lines"]   = np.zeros((0,4), float)
        empty["centers"] = np.zeros((0,2), float)
        return {
            "xfam": empty,
            "yfam": empty,
            "xfam_angle": None,
            "yfam_angle": None,
            "pivot": (0.0, 0.0),
        }

    # -----------------------------
    # Determine which is X vs Y
    # -----------------------------
    med1 = float(np.median(ang1)) if ang1.size else 0.0
    med2 = float(np.median(ang2)) if ang2.size else 0.0

    if med1 > med2:
        yfam_raw, xfam_raw = f1, f2
        yang, xang = med1, med2
    else:
        yfam_raw, xfam_raw = f2, f1
        yang, xang = med2, med1

    # -----------------------------
    # Compute image pivot
    # -----------------------------
    H, W = img.shape[:2]
    cx = W * 0.5
    cy = H * 0.5
    pivot = np.array([cx, cy], float)

    # -----------------------------
    # Rotation matrix
    # -----------------------------
    angle_deg = float(analysis["rotation_angle_deg"])
    theta = np.deg2rad(angle_deg)

    R = np.array([
        [ np.cos(theta),  np.sin(theta) ],
        [ -np.sin(theta), np.cos(theta) ],
    ], float)

    # -----------------------------
    # Rotate segments, centers, angles
    # -----------------------------
    def rotate_family(fam: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Rotate lines, centers, and angles for a single family.
        Preserve all other LSD metadata unchanged.
        """
        lines = np.asarray(fam["lines"], float)
        centers = np.asarray(fam["centers"], float)
        angles = np.asarray(fam["angles"], float)

        if lines.size == 0:
            return {k: fam[k] for k in fam}

        # ---- Rotate endpoints ----
        p1 = lines[:, 0:2]
        p2 = lines[:, 2:4]

        p1r = (p1 - pivot) @ R.T + pivot
        p2r = (p2 - pivot) @ R.T + pivot
        lines_r = np.hstack([p1r, p2r])

        # ---- Recompute centers from rotated segments ----
        centers_r = np.column_stack([
            0.5 * (lines_r[:, 0] + lines_r[:, 2]),
            0.5 * (lines_r[:, 1] + lines_r[:, 3])
        ])

        # ---- Rotate angles ----
        angles_r = angles - angle_deg
        angles_r = ((angles_r + 180) % 180) - 90

        # ---- Return same fields ----
        return {
            "lines":      lines_r,
            "widths":     fam["widths"],
            "precisions": fam["precisions"],
            "nfa":        fam["nfa"],
            "lengths":    fam["lengths"],
            "centers":    centers_r,
            "angles":     angles_r,
        }

    xfam_rot = rotate_family(xfam_raw)
    yfam_rot = rotate_family(yfam_raw)

    # -----------------------------
    # Final return
    # -----------------------------
    return {
        "xfam": xfam_rot,
        "yfam": yfam_rot,
        "xfam_angle": xang,
        "yfam_angle": yang,
        "pivot": (cx, cy),
    }


def draw_centerline_arrays(
    img: np.ndarray,
    centers: Dict[str, np.ndarray],
    color_x=(0, 0, 255),      # RED  (horizontal lines)
    color_y=(255, 0, 0),      # BLUE (vertical lines)
    radius: int = 2,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw centerlines derived from xcenters (horizontal) and
    ycenters (vertical) over an image.

    xcenters[i] = [xc, yc, length]  -> horizontal red ticks
    ycenters[i] = [xc, yc, length]  -> vertical blue ticks
    """

    out = img.copy()

    # ---------------------------------------
    # Horizontal (xfam) - RED
    # ---------------------------------------
    xarr = np.asarray(centers.get("xcenters", []), float)
    for (xc, yc, L) in xarr:
        xc_i = int(round(xc))
        yc_i = int(round(yc))

        # Draw center point
        cv2.circle(out, (xc_i, yc_i), radius, color_x, -1, cv2.LINE_AA)

        # Horizontal line segment
        dx = int(L)
        cv2.line(
            out,
            (xc_i - dx, yc_i),
            (xc_i + dx, yc_i),
            color_x,
            thickness,
            cv2.LINE_AA,
        )

    # ---------------------------------------
    # Vertical (yfam) - BLUE
    # ---------------------------------------
    yarr = np.asarray(centers.get("ycenters", []), float)
    for (xc, yc, L) in yarr:
        xc_i = int(round(xc))
        yc_i = int(round(yc))

        # Draw center
        cv2.circle(out, (xc_i, yc_i), radius, color_y, -1, cv2.LINE_AA)

        # Vertical line segment
        dy = int(L)
        cv2.line(
            out,
            (xc_i, yc_i - dy),
            (xc_i, yc_i + dy),
            color_y,
            thickness,
            cv2.LINE_AA,
        )

    return out


def plot_rotated_family_length_histograms(
    rotated_families: Dict[str, Dict[str, np.ndarray]],
    bin_size: float = 10.0,
    title: str = "Segment Length Distribution (Rotated Families)",
) -> None:
    """
    Produce two stacked histograms:
        Top    - Y-family (vertical-ish) segment length distribution (blue)
        Bottom - X-family (horizontal-ish) segment length distribution (red)

    Parameters
    ----------
    rotated_families : dict
        Output of reassign_and_rotate_families_by_image_center().
        Must contain:
            rotated_families["xfam"]["lines"]
            rotated_families["xfam"]["lengths"]
            rotated_families["yfam"]["lines"]
            rotated_families["yfam"]["lengths"]

    bin_size : float
        Histogram bin width in pixels.

    title : str
        Figure title.
    """

    # -----------------------------------------------------
    # Extract lines + lengths safely (new structure)
    # -----------------------------------------------------
    xfam = rotated_families.get("xfam", {})
    yfam = rotated_families.get("yfam", {})

    x_lines = np.asarray(xfam.get("lines", np.zeros((0,4))), float)
    y_lines = np.asarray(yfam.get("lines", np.zeros((0,4))), float)

    # Prefer precomputed LSD lengths (correct)
    x_lengths = np.asarray(xfam.get("lengths", np.zeros(0)), float)
    y_lengths = np.asarray(yfam.get("lengths", np.zeros(0)), float)

    # Fallback: compute if missing
    if x_lengths.size != x_lines.shape[0]:
        dx = x_lines[:, 2] - x_lines[:, 0]
        dy = x_lines[:, 3] - x_lines[:, 1]
        x_lengths = np.hypot(dx, dy)

    if y_lengths.size != y_lines.shape[0]:
        dx = y_lines[:, 2] - y_lines[:, 0]
        dy = y_lines[:, 3] - y_lines[:, 1]
        y_lengths = np.hypot(dx, dy)

    # -----------------------------------------------------
    # Compute histogram edges
    # -----------------------------------------------------
    def compute_edges(lengths: np.ndarray, bin_size: float) -> np.ndarray:
        if lengths.size == 0:
            return np.array([0, 1], float)
        Lmin = float(np.min(lengths))
        Lmax = float(np.max(lengths))
        nbins = max(1, int(np.ceil((Lmax - Lmin) / bin_size)))
        return np.linspace(Lmin, Lmin + nbins * bin_size, nbins + 1)

    edges_y = compute_edges(y_lengths, bin_size)
    edges_x = compute_edges(x_lengths, bin_size)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    plt.suptitle(title, fontsize=14, y=0.97)

    # ---- Y-family (top, blue)
    ax1.hist(
        y_lengths,
        bins=edges_y,
        color="blue",
        alpha=0.75,
        edgecolor="black",
    )
    ax1.set_ylabel("Count")
    ax1.set_title("Y-family Length Distribution (vertical lines)")

    # ---- X-family (bottom, red)
    ax2.hist(
        x_lengths,
        bins=edges_x,
        color="red",
        alpha=0.75,
        edgecolor="black",
    )
    ax2.set_xlabel("Segment length (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("X-family Length Distribution (horizontal lines)")

    plt.tight_layout()
    plt.show()


def yc_hist(
    rotated_families: Dict[str, np.ndarray],
    bin_size: int = 10,
    gap_size: int = 0,
    offset: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build a histogram over the X coordinate of Y-family (vertical) lines,
    but using bin blocks with optional initial offset and inter-bin gaps.

    Each block: [offset + k*(bin_size + gap_size),
                 offset + k*(bin_size + gap_size) + bin_size)

    Parameters
    ----------
    rotated_families : dict
        Output of reassign_and_rotate_families_by_image_center().
    bin_size : int
        Width of each active bin region.
    gap_size : int
        Number of pixels to skip after each bin.
    offset : int
        Pixels to skip at the beginning before first bin.

    Returns
    -------
    dict with:
        "bin_edges"   : array(float), 2 edges per bin (start,end)
        "bin_centers" : centers of ACTIVE bins
        "counts"      : weighted by segment lengths
    """

    yfam = np.asarray(rotated_families["yfam"], float)
    if yfam.size == 0:
        return {
            "bin_edges": np.zeros(0),
            "bin_centers": np.zeros(0),
            "counts": np.zeros(0),
        }

    # ------------------------------
    # Segment midpoints and lengths
    # ------------------------------
    x1 = yfam[:, 0]
    y1 = yfam[:, 1]
    x2 = yfam[:, 2]
    y2 = yfam[:, 3]

    xc = 0.5 * (x1 + x2)
    lengths = np.hypot(x2 - x1, y2 - y1)

    # ------------------------------
    # Build custom "gapped" bins
    # ------------------------------
    xmin = float(np.min(xc))
    xmax = float(np.max(xc))

    # Shift xmin by `offset`
    start = xmin + offset
    if start > xmax:
        # offset too large -> empty histogram
        return {
            "bin_edges": np.zeros(0),
            "bin_centers": np.zeros(0),
            "counts": np.zeros(0),
        }

    # List of active bin intervals
    bin_edges = []
    bin_centers = []
    counts = []

    # Iterate over active bins
    left = start
    step = bin_size + gap_size

    while left < xmax:
        right = left + bin_size
        if right > xmax:
            right = xmax

        # Weighted count: sum lengths of segments whose xc lies in [left,right)
        mask = (xc >= left) & (xc < right)
        intensity = lengths[mask].sum()

        # push new bin
        bin_edges.append((left, right))
        bin_centers.append(0.5 * (left + right))
        counts.append(float(intensity))

        left += step

    # Convert to arrays
    bin_edges = np.array(bin_edges, float)  # shape (nb,2)
    bin_centers = np.array(bin_centers, float)
    counts = np.array(counts, float)

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts,
    }


import numpy as np
import matplotlib.pyplot as plt

def plot_yc_hist(hist: dict, title: str = "Y-family X-Histogram", color="blue"):
    """
    Plot 1D histogram returned by yc_hist_from_clusters().

    hist must contain:
        - hist["bin_centers"]: (K,) float array
        - hist["counts"]:      (K,) float array
        - hist["bin_edges"]:   (K+1,) float array
    """

    x = np.asarray(hist["bin_centers"], float)
    y = np.asarray(hist["counts"], float)
    edges = np.asarray(hist["bin_edges"], float)

    if x.size == 0:
        print("Empty histogram - nothing to plot.")
        return

    # bar width = edge-to-edge distance (ignores gap spacing)
    if len(edges) >= 2:
        width = edges[1] - edges[0]
    else:
        width = 5.0  # fallback

    plt.figure(figsize=(10, 4))
    plt.bar(x, y, width=width, color=color, alpha=0.7, align='center')

    plt.xlabel("Position (pixels)")
    plt.ylabel("Cluster weight")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def yc_hist_from_clusters(
    cluster_centers: np.ndarray,
    bin_size: int = 10,
    gap_size: int = 0,
    offset: float = 0.0,
    weights: np.ndarray | None = None,
):
    """
    Histogram over X positions of Y-family (vertical) gridline cluster centers.

    Parameters
    ----------
    cluster_centers : array
        1D array of X-coordinates of clustered Y-family gridlines.
    bin_size : int
        Width of each histogram bin (in pixels).
    gap_size : int
        Space to skip after each bin (bin spacing).
    offset : float
        Shift histogram start by this many pixels.
    weights : array or None
        Optional per-cluster weights (default: all 1).

    Returns
    -------
    dict with:
        "bin_edges", "bin_centers", "counts"
    """

    centers = np.asarray(cluster_centers, float)
    if centers.size == 0:
        return {"bin_edges": np.zeros(0), "bin_centers": np.zeros(0), "counts": np.zeros(0)}

    if weights is None:
        weights = np.ones_like(centers)

    # ---------------------------------------------
    # Shift by offset
    # ---------------------------------------------
    centers_shifted = centers - offset

    # ---------------------------------------------
    # Compute bin edges with optional gap
    # ---------------------------------------------
    xmin = float(np.min(centers_shifted))
    xmax = float(np.max(centers_shifted))

    # Effective bin width = bin_size + gap_size
    eff_bin = float(bin_size + gap_size)

    nbins = max(1, int(np.ceil((xmax - xmin) / eff_bin)))
    bin_edges = xmin + np.arange(nbins + 1) * eff_bin

    # histogram
    counts, _ = np.histogram(centers_shifted, bins=bin_edges, weights=weights)

    # bin centers (just geometric centers, ignoring gap)
    bin_centers = bin_edges[:-1] + bin_size * 0.5

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts.astype(float),
    }


def xc_hist_from_clusters(
    cluster_centers: np.ndarray,
    bin_size: int = 10,
    gap_size: int = 0,
    offset: float = 0.0,
    weights: np.ndarray | None = None,
):
    """
    Histogram over Y positions of X-family (horizontal) gridline cluster centers.
    Same logic as yc_hist_from_clusters.
    """

    centers = np.asarray(cluster_centers, float)
    if centers.size == 0:
        return {"bin_edges": np.zeros(0), "bin_centers": np.zeros(0), "counts": np.zeros(0)}

    if weights is None:
        weights = np.ones_like(centers)

    centers_shifted = centers - offset

    ymin = float(np.min(centers_shifted))
    ymax = float(np.max(centers_shifted))

    eff_bin = float(bin_size + gap_size)

    nbins = max(1, int(np.ceil((ymax - ymin) / eff_bin)))
    bin_edges = ymin + np.arange(nbins + 1) * eff_bin

    counts, _ = np.histogram(centers_shifted, bins=bin_edges, weights=weights)
    bin_centers = bin_edges[:-1] + bin_size * 0.5

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts.astype(float),
    }


def cluster_gridlines_1d(
    positions: np.ndarray,
    lengths: Optional[np.ndarray] = None,
    max_gap_px: Optional[float] = None,
    min_cluster_members: int = 1,
    robust_factor: float = 3.0,
) -> Dict[str, Any]:
    """
    Cluster 1D gridline candidates (segment midpoints) into physical gridlines.

    This is a 1D, order-preserving clustering:
        - Sort positions
        - Start a new cluster whenever the gap to the previous point
          exceeds max_gap_px
        - If max_gap_px is None, estimate it robustly from the small
          nearest-neighbor differences (jitter scale) and multiply by
          robust_factor.

    Parameters
    ----------
    positions : np.ndarray
        1D array of candidate line positions along an axis (e.g. xc for
        vertical lines, yc for horizontal lines). Shape (N,).
    lengths : np.ndarray or None
        Optional 1D array of per-segment lengths used as weights.
        If None, each point has weight=1. Shape (N,).
    max_gap_px : float or None
        Maximum allowed gap (in pixels) between consecutive sorted
        positions to be considered part of the same cluster. If None,
        it is estimated from the data using a robust heuristic.
    min_cluster_members : int
        Discard clusters with fewer than this many members.
    robust_factor : float
        Multiplier applied to the estimated jitter scale when computing
        max_gap_px if it is not provided explicitly.

    Returns
    -------
    dict
        {
            "centers":        (K,) array of cluster centers (median position),
            "weights":        (K,) array of cluster weights (sum of member weights),
            "median_lengths": (K,) array of median member lengths (NaN if no lengths),
            "counts":         (K,) array of member counts,
            "clusters":       list of np.ndarray index arrays, one per cluster,
            "max_gap_px":     float, final threshold used,
        }

        K is the number of retained clusters (after min_cluster_members filter).

    Notes
    -----
    - This function is designed to convert many fragmented segments that
      belong to the same physical gridline into a single robust
      "virtual line".
    - It should be run separately for the two families (xfam, yfam),
      e.g.:

        xc = xcenters[:, 0]
        Lx = xcenters[:, 2]
        x_clusters = cluster_gridlines_1d(xc, lengths=Lx)

        yc = ycenters[:, 1]
        Ly = ycenters[:, 2]
        y_clusters = cluster_gridlines_1d(yc, lengths=Ly)
    """
    positions = np.asarray(positions, dtype=float).ravel()
    N = positions.size

    if N == 0:
        return {
            "centers":        np.zeros(0, dtype=float),
            "weights":        np.zeros(0, dtype=float),
            "median_lengths": np.zeros(0, dtype=float),
            "counts":         np.zeros(0, dtype=int),
            "clusters":       [],
            "max_gap_px":     max_gap_px,
        }

    if lengths is None:
        weights = np.ones_like(positions, dtype=float)
        lengths_array = None
    else:
        lengths_array = np.asarray(lengths, dtype=float).ravel()
        if lengths_array.size != N:
            raise ValueError("lengths must have the same size as positions")
        weights = lengths_array.copy()

    # -------------------------------------------------
    # Sort by position, keep original indices
    # -------------------------------------------------
    order = np.argsort(positions)
    pos_sorted = positions[order]
    w_sorted = weights[order]
    if lengths_array is not None:
        L_sorted = lengths_array[order]
    else:
        L_sorted = None

    # -------------------------------------------------
    # Auto-estimate max_gap_px if not provided
    # -------------------------------------------------
    if max_gap_px is None:
        if N == 1:
            # Single point -> arbitrary small gap
            est_gap = 1.0
        else:
            diffs = np.diff(pos_sorted)
            # Keep only positive diffs
            diffs = diffs[diffs > 0]

            if diffs.size == 0:
                est_gap = 1.0
            else:
                # Heuristic: small diffs correspond to within-line jitter,
                # large diffs correspond to grid spacing.
                # Pick lower quantile to capture jitter scale.
                q25 = np.percentile(diffs, 25)
                q50 = np.percentile(diffs, 50)
                jitter_scale = min(q25, q50)

                if jitter_scale <= 0:
                    jitter_scale = np.median(diffs)

                # Fall back if still degenerate
                if jitter_scale <= 0:
                    jitter_scale = max(1.0, float(np.min(diffs)))

                est_gap = jitter_scale * robust_factor

        max_gap_px = float(est_gap)

    # -------------------------------------------------
    # Run gap-based clustering in sorted order
    # -------------------------------------------------
    clusters_idx: List[np.ndarray] = []

    current_start = 0
    for i in range(1, N):
        gap = pos_sorted[i] - pos_sorted[i - 1]
        if gap > max_gap_px:
            # close current cluster [current_start, i)
            clusters_idx.append(np.arange(current_start, i, dtype=int))
            current_start = i

    # last cluster
    clusters_idx.append(np.arange(current_start, N, dtype=int))

    # -------------------------------------------------
    # Convert sorted indices back to original indices
    # and compute cluster stats
    # -------------------------------------------------
    centers_list = []
    weights_list = []
    med_lengths_list = []
    counts_list = []
    clusters_final: List[np.ndarray] = []

    for c_sorted in clusters_idx:
        if c_sorted.size < min_cluster_members:
            continue

        # original indices
        c_orig = order[c_sorted]

        # positions & weights in this cluster
        pos_c = positions[c_orig]
        w_c = weights[c_orig]

        center = float(np.median(pos_c))
        total_w = float(np.sum(w_c))
        count = int(c_orig.size)

        if L_sorted is not None:
            L_c = lengths_array[c_orig]
            med_L = float(np.median(L_c))
        else:
            med_L = float("nan")

        centers_list.append(center)
        weights_list.append(total_w)
        med_lengths_list.append(med_L)
        counts_list.append(count)
        clusters_final.append(c_orig)

    if len(centers_list) == 0:
        return {
            "centers":        np.zeros(0, dtype=float),
            "weights":        np.zeros(0, dtype=float),
            "median_lengths": np.zeros(0, dtype=float),
            "counts":         np.zeros(0, dtype=int),
            "clusters":       [],
            "max_gap_px":     max_gap_px,
        }

    centers_arr = np.array(centers_list, dtype=float)
    weights_arr = np.array(weights_list, dtype=float)
    medL_arr = np.array(med_lengths_list, dtype=float)
    counts_arr = np.array(counts_list, dtype=int)

    return {
        "centers":        centers_arr,
        "weights":        weights_arr,
        "median_lengths": medL_arr,
        "counts":         counts_arr,
        "clusters":       clusters_final,
        "max_gap_px":     max_gap_px,
    }


def periodicity_detector_1d(
    centers: np.ndarray,
    weights: np.ndarray | None = None,
    min_lines: int = 4,
) -> dict:
    """
    Estimate gridline spacing (period) from 1D cluster centers.

    Performs:
        1) Median spacing (robust baseline)
        2) Autocorrelation peak detection (best)
        3) FFT dominant frequency (sanity check)

    Parameters
    ----------
    centers : array
        1D array of cluster center coordinates along the axis.
    weights : array or None
        Optional 1D weights per cluster (from cluster_gridlines_1d).
        If None -> weight = 1.
    min_lines : int
        Require at least this many gridlines to estimate periodicity.

    Returns
    -------
    dict with:
        "n"               - number of cluster centers
        "median_spacing"  - median diff between sorted centers
        "autocorr_spacing"- spacing from autocorr peak (None if fail)
        "fft_spacing"     - spacing from FFT (None if fail)
        "best"            - chosen spacing (prefers autocorr)
        "centers"         - sorted centers
        "lags"            - autocorr lags
        "autocorr"        - autocorrelation sequence
        "freqs"           - FFT frequencies
        "fft_mag"         - FFT magnitude spectrum
    """

    centers = np.asarray(centers, float).ravel()
    n = centers.size

    if n < min_lines:
        return {
            "n": n,
            "median_spacing": None,
            "autocorr_spacing": None,
            "fft_spacing": None,
            "best": None,
            "centers": centers,
            "lags": None,
            "autocorr": None,
            "freqs": None,
            "fft_mag": None,
        }

    # ---------------------------------------------
    # Sorted positions
    # ---------------------------------------------
    centers_sorted = np.sort(centers)

    # ---------------------------------------------
    # 1. Median spacing
    # ---------------------------------------------
    diffs = np.diff(centers_sorted)
    median_spacing = float(np.median(diffs)) if diffs.size else None

    # ---------------------------------------------
    # 2. Autocorrelation-based spacing
    # ---------------------------------------------
    if weights is None:
        w = np.ones_like(centers_sorted)
    else:
        w = np.asarray(weights, float).ravel()
        w = w[np.argsort(centers)]  # align weights with sorted centers

    # Normalize weights
    w = w / np.max(w)

    # Signal for autocorrelation
    signal = (centers_sorted - centers_sorted.mean()) * w

    # Autocorrelation (positive lags only)
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2:]
    lags = np.arange(corr.size)

    # Find first peak AFTER lag 0
    peaks, _ = find_peaks(corr, distance=1)

    if len(peaks) >= 2:
        lag1 = peaks[1]         # skip lag0 (self-peak)
        autocorr_spacing = float(centers_sorted[lag1] - centers_sorted[0]) \
                           if lag1 < centers_sorted.size else None
    else:
        autocorr_spacing = None

    # ---------------------------------------------
    # 3. FFT-based spacing
    # ---------------------------------------------
    # Convert centers to histogram-like signal
    # (dense sampling improves FFT)
    N_fft = max(256, 2 ** int(np.ceil(np.log2(n * 8))))
    x_norm = (centers_sorted - centers_sorted.min()) / (centers_sorted.max() - centers_sorted.min())
    signal_dense = np.histogram(x_norm, bins=N_fft, weights=w)[0]

    F = np.fft.rfft(signal_dense - signal_dense.mean())
    freqs = np.fft.rfftfreq(signal_dense.size)

    # find dominant nonzero frequency
    if F.size > 2:
        idx = np.argmax(np.abs(F[1:])) + 1
        freq = freqs[idx]
        fft_spacing = float(1.0 / freq) if freq > 0 else None
    else:
        fft_spacing = None

    # ---------------------------------------------
    # Decide best estimate
    # ---------------------------------------------
    if autocorr_spacing is not None:
        best = autocorr_spacing
    else:
        best = median_spacing

    return {
        "n": n,
        "median_spacing": median_spacing,
        "autocorr_spacing": autocorr_spacing,
        "fft_spacing": fft_spacing,
        "best": best,
        "centers": centers_sorted,
        "lags": lags,
        "autocorr": corr,
        "freqs": freqs,
        "fft_mag": np.abs(F),
    }


def plot_periodicity_analysis(period: dict, title="Gridline Periodicity Analysis"):
    """
    Visualize periodicity estimation diagnostics:
        1) Sorted cluster centers
        2) Autocorrelation (positive lags)
        3) FFT magnitude spectrum

    Parameters
    ----------
    period : dict
        Output from periodicity_detector_1d()
    title : str
        Main title for figure
    """

    centers = period["centers"]
    autocorr = period["autocorr"]
    lags = period["lags"]
    freqs = period["freqs"]
    fft_mag = period["fft_mag"]

    median_spacing = period["median_spacing"]
    autocorr_spacing = period["autocorr_spacing"]
    fft_spacing = period["fft_spacing"]
    best = period["best"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # -------------------------------------------------------
    # 1) Sorted Centers
    # -------------------------------------------------------
    ax = axes[0]
    ax.plot(centers, np.zeros_like(centers), "o", color="black")
    ax.set_title(f"Gridline centers (n={len(centers)})")
    ax.set_xlabel("Position (px)")
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.4)

    # Median spacing overlay
    if median_spacing:
        ax.text(0.05, 0.85,
                f"Median spacing = {median_spacing:.2f}px",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7))

    # -------------------------------------------------------
    # 2) Autocorrelation
    # -------------------------------------------------------
    ax = axes[1]
    ax.plot(lags, autocorr, color="blue")
    ax.set_title("Autocorrelation of cluster-center signal")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorr")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Mark autocorr spacing
    if autocorr_spacing:
        # find closest lag index
        lag_idx = np.argmin(np.abs(centers - centers[0] - autocorr_spacing))
        ax.axvline(lag_idx, color="red", linestyle="--", label=f"AC peak -> {autocorr_spacing:.2f}px")
        ax.legend()

    # -------------------------------------------------------
    # 3) FFT magnitude
    # -------------------------------------------------------
    ax = axes[2]
    ax.plot(freqs, fft_mag, color="purple")
    ax.set_title("FFT magnitude spectrum")
    ax.set_xlabel("Frequency (1/pixels)")
    ax.set_ylabel("|FFT|")
    ax.grid(True, linestyle=":", alpha=0.4)

    if fft_spacing:
        freq = 1.0 / fft_spacing
        ax.axvline(freq, color="red", linestyle="--",
                   label=f"FFT spacing -> {fft_spacing:.2f}px")
        ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_grid_periodicity_full(
    famxy: dict,
    min_cluster_members: int = 2,
    robust_factor: float = 3.0,
):
    """
    Full periodicity analysis wrapper.

    Takes:
        famxy = compute_centerline_arrays(output of reassign_and_rotate_families_by_image_center)

    Performs:
        1) Extract x- and y-centers
        2) cluster_gridlines_1d for each family
        3) Run periodicity_detector_1d
        4) Plot diagnostics for each axis
        5) Return structured result

    Returns:
        {
            "X": {...period_x...},
            "Y": {...period_y...}
        }
    """

    from math import isnan

    # -----------------------------------------------------------
    # Extract centerline arrays
    # -----------------------------------------------------------
    xcenters = famxy["xcenters"]  # [xc, yc, length]
    ycenters = famxy["ycenters"]

    xc_x = xcenters[:, 0]
    yc_x = xcenters[:, 1]
    Lx   = xcenters[:, 2]

    xc_y = ycenters[:, 0]
    yc_y = ycenters[:, 1]
    Ly   = ycenters[:, 2]

    # -----------------------------------------------------------
    # Cluster Y-family (vertical gridlines)
    # -----------------------------------------------------------
    y_clusters = cluster_gridlines_1d(
        positions=xc_y,
        lengths=Ly,
        max_gap_px=None,
        min_cluster_members=min_cluster_members,
        robust_factor=robust_factor,
    )

    period_y = periodicity_detector_1d(
        y_clusters["centers"],
        weights=y_clusters["weights"]
    )

    # -----------------------------------------------------------
    # Cluster X-family (horizontal gridlines)
    # -----------------------------------------------------------
    x_clusters = cluster_gridlines_1d(
        positions=yc_x,
        lengths=Lx,
        max_gap_px=None,
        min_cluster_members=min_cluster_members,
        robust_factor=robust_factor,
    )

    period_x = periodicity_detector_1d(
        x_clusters["centers"],
        weights=x_clusters["weights"]
    )

    # -----------------------------------------------------------
    # Plot diagnostics
    # -----------------------------------------------------------
    plot_periodicity_analysis(period_y, title="Periodicity - Y-family (vertical lines)")
    plot_periodicity_analysis(period_x, title="Periodicity - X-family (horizontal lines)")

    return {
        "X": period_x,
        "Y": period_y,
    }


# ============================================================================
# 3) FILTERING + 2-CLUSTER ANGLE CLASSIFICATION
# ============================================================================

def _cluster_angles_two(angles: np.ndarray, iters: int = 10):
    """
    Very small, fast, stable two-cluster 1D KMeans alternative.
    Works directly on angles mod pi.
    """
    ang = np.mod(angles, np.pi)
    c0, c1 = float(np.min(ang)), float(np.max(ang))
    centers = np.array([c0, c1], np.float32)

    for _ in range(iters):
        d = np.abs(ang[:, None] - centers[None, :])
        labels = np.argmin(d, axis=1)
        for k in range(2):
            pts = ang[labels == k]
            if len(pts):
                centers[k] = pts.mean()

    return centers, labels


def filter_grid_segments(
    raw: Dict,
    angle_tol_deg: float = 20.0,
) -> Dict:
    """
    Minimal filtering for graph-paper grids.

    Strategy:
        - Compute angle for ALL raw segments
        - Cluster angles into 2 dominant orientations
        - Keep segments within +/-angle_tol_deg of each cluster center

    NO LENGTH FILTERING!
    (except rejecting segments of near-zero pixel length)

    Returns:
        {
            "lines_x": array(M1,4)
            "lines_y": array(M2,4)
            "centers": [angle_x, angle_y]
            "labels": list
            "kept_segments": (M1+M2,4)
        }
    """

    segs = raw["lines"]
    if len(segs) == 0:
        return {
            "lines_x": np.zeros((0,4)),
            "lines_y": np.zeros((0,4)),
            "centers": [None, None],
            "labels": [],
            "kept_segments": np.zeros((0,4)),
        }

    # -------------------------------------------------------
    # Compute angles for ALL segments
    # -------------------------------------------------------
    dx = segs[:,2] - segs[:,0]
    dy = segs[:,3] - segs[:,1]
    angles = np.arctan2(dy, dx)

    # -------------------------------------------------------
    # Cluster into 2 orientations
    # -------------------------------------------------------
    centers, labels = _cluster_angles_two(angles)
    c0, c1 = centers
    tol = np.deg2rad(angle_tol_deg)

    # -------------------------------------------------------
    # Filter by angle (ONLY)
    # -------------------------------------------------------
    targets = np.where(labels == 0, c0, c1)
    diff = np.abs((angles - targets + np.pi/2) % np.pi - np.pi/2)
    good_idx = diff < tol
    kept = segs[good_idx]
    labels_good = labels[good_idx]

    # -------------------------------------------------------
    # Output two angle families
    # -------------------------------------------------------
    lines_x = kept[labels_good == 0]
    lines_y = kept[labels_good == 1]

    return {
        "lines_x": lines_x,
        "lines_y": lines_y,
        "centers": centers.tolist(),
        "labels": labels_good.tolist(),
        "kept_segments": kept,
    }


# ============================================================================
# 4) SIMPLE FAMILY SPLIT
# ============================================================================

def split_segments_by_angle_circular(
    raw_lsd: Dict[str, np.ndarray],
    angle_info: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    angle_range: Tuple[float, float] = (-45.0, 135.0),
) -> Dict[str, np.ndarray]:
    """
    Split LSD segments into two angular families based on KDE-derived peak
    separation.  The function takes the full raw_lsd structure:

        {
            "lines": (N,4),
            "widths": (N,),
            "precisions": (N,),
            "nfa": (N,),
            "lengths": (N,),
            "centers": (N,2)
        }

    Returns ONLY per-family subsets (NO labels field):

        {
            "family1": {
                "lines": ...,
                "widths": ...,
                "precisions": ...,
                "nfa": ...,
                "lengths": ...,
                "centers": ...,
                "angles": ...,
            },
            "family2": { ...same fields... },
            "angles1": 1D array of angles for family1,
            "angles2": 1D array of angles for family2,
        }
    """

    # ------------------------------
    # Validate input
    # ------------------------------
    required_keys = ["lines", "widths", "precisions", "nfa", "lengths", "centers"]
    for k in required_keys:
        if k not in raw_lsd:
            raise KeyError(f"raw_lsd missing required key '{k}'")

    segments   = np.asarray(raw_lsd["lines"])
    widths     = np.asarray(raw_lsd["widths"])
    precisions = np.asarray(raw_lsd["precisions"])
    nfa        = np.asarray(raw_lsd["nfa"])
    lengths    = np.asarray(raw_lsd["lengths"])
    centers    = np.asarray(raw_lsd["centers"])

    N = segments.shape[0]
    if N == 0:
        empty = {
            "lines": np.zeros((0,4), float),
            "widths": np.zeros((0,), float),
            "precisions": np.zeros((0,), float),
            "nfa": np.zeros((0,), float),
            "lengths": np.zeros((0,), float),
            "centers": np.zeros((0,2), float),
            "angles": np.zeros((0,), float),
        }
        return {
            "family1": empty,
            "family2": empty,
            "angles1": empty["angles"],
            "angles2": empty["angles"],
        }

    # ------------------------------
    # Angles
    # ------------------------------
    if "angles_deg" not in angle_info:
        raise KeyError("angle_info must contain 'angles_deg' from compute_segment_angles()")

    angles = np.asarray(angle_info["angles_deg"], float)
    if angles.shape[0] != N:
        raise ValueError(
            f"Angle array has {angles.shape[0]} items but segments have {N}"
        )

    # ------------------------------
    # Peak split from analysis
    # ------------------------------
    p1, p2 = analysis["peaks"]
    split_deg = float(analysis["split_deg"])

    # Normalize split into domain [-45,135)
    amin, amax = angle_range
    split_norm = ((split_deg - amin) % 180.0) + amin

    # ------------------------------
    # Assign families
    # ------------------------------
    labels = (angles > split_norm).astype(int)

    # Masks
    m1 = labels == 0
    m2 = labels == 1

    # ------------------------------
    # Build family outputs
    # ------------------------------
    family1 = {
        "lines":      segments[m1],
        "widths":     widths[m1],
        "precisions": precisions[m1],
        "nfa":        nfa[m1],
        "lengths":    lengths[m1],
        "centers":    centers[m1],
        "angles":     angles[m1],
    }
    family2 = {
        "lines":      segments[m2],
        "widths":     widths[m2],
        "precisions": precisions[m2],
        "nfa":        nfa[m2],
        "lengths":    lengths[m2],
        "centers":    centers[m2],
        "angles":     angles[m2],
    }

    return {
        "family1": family1,
        "family2": family2,
        "angles1": angles[m1],
        "angles2": angles[m2],
    }


# ============================================================================
# 5) VANISHING POINT ESTIMATION (LEAST SQUARES)
# ============================================================================

def _fit_vanishing_point_least_squares(
    lines: np.ndarray,
    min_lines: int = 2,
) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    Fit a vanishing point (x,y) as the point minimizing squared distance
    to a set of lines, in a least-squares sense.

    Each line is given by segment [x1,y1,x2,y2]. We convert to the
    normalized line equation a*x + b*y + c = 0 with sqrt(a^2+b^2) = 1
    and solve for the point minimizing sum_i (a_i x + b_i y + c_i)^2.

    Returns:
        (vp_xy, rms_error) where:
            vp_xy: (x, y) in image coordinates, or None if degenerate.
            rms_error: root-mean-square distance to the lines in pixels,
                       or None if degenerate.
    """
    if lines is None or len(lines) < min_lines:
        return None, None

    # Build normal equations for least squares
    S_aa = S_ab = S_bb = 0.0
    S_ac = S_bc = 0.0

    x1 = lines[:,0]; y1 = lines[:,1]
    x2 = lines[:,2]; y2 = lines[:,3]
    
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    
    mask = length >= 1e-6
    dx = dx[mask]; dy = dy[mask]; length = length[mask]
    x1 = x1[mask]; y1 = y1[mask]
    if dx.size < min_lines:
        return None, None
    
    a = dy / length
    b = -dx / length
    c = -(a * x1 + b * y1)
    
    S_aa = np.sum(a*a)
    S_ab = np.sum(a*b)
    S_bb = np.sum(b*b)
    S_ac = np.sum(a*c)
    S_bc = np.sum(b*c)

    # 2x2 system: [S_aa S_ab][x] = -[S_ac]
    #              [S_ab S_bb][y]    [S_bc]
    det = S_aa * S_bb - S_ab * S_ab
    if abs(det) < 1e-9:
        return None, None

    inv_aa = S_bb / det
    inv_ab = -S_ab / det
    inv_bb = S_aa / det

    bx = -S_ac
    by = -S_bc

    x_vp = inv_aa * bx + inv_ab * by
    y_vp = inv_ab * bx + inv_bb * by

    # Compute RMS distance to lines as an error measure
    d = a * x_vp + b * y_vp + c
    rms = float(np.sqrt(np.mean(d*d))) if d.size > 0 else None

    return (float(x_vp), float(y_vp)), rms


def estimate_vanishing_points(
    lines_x: np.ndarray,
    lines_y: np.ndarray,
    img_shape: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Estimate vanishing points for two line families (grid axes).

    Args:
        lines_x:
            Array (N1,4) of segments [x1,y1,x2,y2] for the first family.
        lines_y:
            Array (N2,4) for the second family.
        img_shape:
            Optional (H,W). If provided, used to approximate principal point
            for a VP-based orthogonality measure.

    Returns:
        dict with keys:
            - vp_x: (x,y) or None
            - vp_y: (x,y) or None
            - rms_x: RMS distance of lines_x to vp_x, or None
            - rms_y: RMS distance of lines_y to vp_y, or None
            - angle_x: average direction angle (radians) for family x, or None
            - angle_y: same for family y, or None
            - angle_orth_error_deg: |(angle_x - angle_y) - 90deg|, or None
            - vp_orth_error_deg:    deviation from 90deg between vectors
                                    (vp_x - c) and (vp_y - c), or None
            - horizon: (a,b,c) line coefficients for the horizon (through
                       both vanishing points), or None
    """
    # --- Fit VPs in least squares sense ---
    vp_x, rms_x = _fit_vanishing_point_least_squares(lines_x)
    vp_y, rms_y = _fit_vanishing_point_least_squares(lines_y)

    # --- Angle-based orthogonality (image-space directions) ---
    angle_x = _average_direction_angle(lines_x)
    angle_y = _average_direction_angle(lines_y)
    angle_orth_error_deg: Optional[float] = None

    if angle_x is not None and angle_y is not None:
        # Normalize to [0, pi)
        ax = (angle_x + np.pi) % np.pi
        ay = (angle_y + np.pi) % np.pi
        delta = abs(ax - ay)
        # Smallest difference mod pi
        if delta > np.pi / 2:
            delta = np.pi - delta
        # Deviation from right angle
        angle_orth_error_deg = abs(np.rad2deg(delta) - 90.0)

    # --- VP-based orthogonality (requires an approximate principal point) ---
    vp_orth_error_deg: Optional[float] = None
    horizon: Optional[Tuple[float, float, float]] = None

    if vp_x is not None and vp_y is not None:
        (vx1, vy1) = vp_x
        (vx2, vy2) = vp_y

        # Horizon line through the two vanishing points
        line_h = _line_from_points((vx1, vy1), (vx2, vy2))
        if line_h is not None:
            horizon = line_h

        # If image shape is given, use center as principal point
        if img_shape is not None:
            H, W = img_shape
            cx = W * 0.5
            cy = H * 0.5

            v1 = np.array([vx1 - cx, vy1 - cy], dtype=np.float64)
            v2 = np.array([vx2 - cx, vy2 - cy], dtype=np.float64)

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                ang = np.arccos(cosang)
                vp_orth_error_deg = abs(np.rad2deg(ang) - 90.0)

    return {
        "vp_x": vp_x,
        "vp_y": vp_y,
        "rms_x": rms_x,
        "rms_y": rms_y,
        "angle_x": angle_x,
        "angle_y": angle_y,
        "angle_orth_error_deg": angle_orth_error_deg,
        "vp_orth_error_deg": vp_orth_error_deg,
        "horizon": horizon,
    }


# ============================================================================
# 6) PRINCIPAL POINT REFINEMENT
# ============================================================================

def refine_principal_point_from_vps(
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
    img_shape: Tuple[int, int],
    radius_frac: float = 0.05,
    steps: int = 20,
) -> Dict:
    """
    Refine the principal point (cx, cy) to minimize vanishing-point
    orthogonality error.

    Args:
        vp_x, vp_y: two vanishing points (x, y)
        img_shape: (H, W)
        radius_frac: search radius = radius_frac * min(H, W)
        steps: sampling resolution (e.g., 20 -> 400 grid evaluations)

    Returns:
        {
            "cx_refined": float,
            "cy_refined": float,
            "vp_orth_error_deg": float,
            "cx0": initial_cx,
            "cy0": initial_cy,
            "radius": search_radius_in_pixels
        }
    """
    if vp_x is None or vp_y is None:
        return {
            "cx_refined": cx0,
            "cy_refined": cy0,
            "vp_orth_error_deg": None,
            "cx0": cx0,
            "cy0": cy0,
            "radius": float(r),
        }

    H, W = img_shape
    cx0 = W * 0.5
    cy0 = H * 0.5

    r = radius_frac * min(H, W)

    # Generate sample grid around (cx0, cy0)
    xs = np.linspace(cx0 - r, cx0 + r, steps)
    ys = np.linspace(cy0 - r, cy0 + r, steps)

    CX, CY = np.meshgrid(xs, ys, indexing='xy')
    
    vx1, vy1 = vp_x
    vx2, vy2 = vp_y
    
    v1x = vx1 - CX
    v1y = vy1 - CY
    v2x = vx2 - CX
    v2y = vy2 - CY
    
    n1 = np.hypot(v1x, v1y)
    n2 = np.hypot(v2x, v2y)
    
    valid = (n1 >= 1e-9) & (n2 >= 1e-9)
    cosang = np.zeros_like(CX)
    cosang[valid] = (v1x[valid] * v2x[valid] + v1y[valid] * v2y[valid]) / (n1[valid] * n2[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    
    err = np.abs(np.rad2deg(np.arccos(cosang)) - 90.0)
    err[~valid] = 1e9
    
    # find minimum
    idx = np.argmin(err)
    best_cx = CX.ravel()[idx]
    best_cy = CY.ravel()[idx]
    best_err = err.ravel()[idx]
    
    return {
        "cx_refined": float(best_cx),
        "cy_refined": float(best_cy),
        "vp_orth_error_deg": float(best_err),
        "cx0": float(cx0),
        "cy0": float(cy0),
        "radius": float(r),
    }


# ============================================================================
# 7) BASIC VISUALIZATION
# ============================================================================

def mark_segments(
    img: np.ndarray,
    segments: np.ndarray,
    color=(0,255,0),
    thickness=1
) -> np.ndarray:
    """
    Draw arbitrary segments onto a copy of the image.
    segments: array(N,4)
    """
    out = img.copy()
    for (x1,y1,x2,y2) in segments:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color, thickness, cv2.LINE_AA)
    return out


def mark_segment_families(
    img: np.ndarray,
    lines_x: np.ndarray,
    lines_y: np.ndarray,
    color_x=(0,0,255),
    color_y=(255,0,0),
    thickness=1
) -> np.ndarray:
    """
    Convenience visualizer for two line families (x and y).
    """
    out = img.copy()
    for (x1,y1,x2,y2) in lines_x:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color_x, thickness, cv2.LINE_AA)
    for (x1,y1,x2,y2) in lines_y:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color_y, thickness, cv2.LINE_AA)
    return out


def plot_angle_histogram(
    hist_data: Dict[str, np.ndarray],
    ax=None,
    title: str = "Segment angle histogram"
):
    """
    Plot a precomputed angle histogram.

    Args:
        hist_data: Dict returned by compute_angle_histogram().
        ax:        Optional Matplotlib Axes. If None, a new figure is created.
        title:     Plot title.

    Returns:
        Matplotlib Axes object used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots()

    counts = hist_data["counts"]
    edges = hist_data["bin_edges"]

    ax.hist(
        bins=edges,
        x=edges[:-1],
        weights=counts,
    )

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # match your normalization interval automatically
    ax.set_xlim(hist_data["range"][0], hist_data["range"][1])

    plt.show()

    return ax


def plot_angle_histogram_with_kde(hist_info: Dict[str, np.ndarray]):
    """
    Plot histogram and KDE contained in hist_info.
    """
    import matplotlib.pyplot as plt

    x = hist_info["hist_x"]
    y = hist_info["hist_y"]
    kx = hist_info["kde_x"]
    ky = hist_info["kde_y"]

    plt.figure(figsize=(10,4))
    plt.bar(x, y, width=np.diff(x).mean(), alpha=0.4, label="Histogram")
    plt.plot(kx, ky * (y.max() / ky.max()), 'r-', label="KDE (scaled)")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_family_kdes(hist, analysis, fam_kdes, title="KDE FAMILY DIAGNOSTICS"):
    """
    Plot:
        - Global histogram
        - Combined KDE (scaled to global hist)
        - Family 1 histogram + KDE (scaled to family1 hist)
        - Family 2 histogram + KDE (scaled to family2 hist)
        - Peaks and split
    """

    import matplotlib.pyplot as plt
    import numpy as np

    lo, hi = hist["range"]
    p1, p2 = analysis["peaks"]
    split = analysis["split_deg"]

    # ---------------------------------------------------------
    # Extract histograms + KDEs
    # ---------------------------------------------------------
    g_hist_x = hist["hist_x"]
    g_hist_y = hist["hist_y"]

    f1 = fam_kdes["fam1"]
    f2 = fam_kdes["fam2"]
    comb = fam_kdes["combined"]

    # Family histograms
    f1_hist_x = f1["hist_x"]
    f1_hist_y = f1["hist_y"]
    f2_hist_x = f2["hist_x"]
    f2_hist_y = f2["hist_y"]

    # KDE x-axis
    kx = comb["kde_x"]

    # KDE y-values
    kde_comb = comb["kde_y"]
    kde_f1 = f1["kde_y"]
    kde_f2 = f2["kde_y"]

    # ---------------------------------------------------------
    # KDE -> Histogram scaling **per family**
    # ---------------------------------------------------------
    def scale_to_hist(kde_y, hist_y):
        hmax = max(hist_y.max(), 1e-12)
        kmax = max(kde_y.max(), 1e-12)
        return kde_y * (hmax / kmax)

    kde_comb_s = scale_to_hist(kde_comb, g_hist_y)
    kde_f1_s   = scale_to_hist(kde_f1, f1_hist_y)
    kde_f2_s   = scale_to_hist(kde_f2, f2_hist_y)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    w = np.diff(g_hist_x).mean()

    # Global histogram
    plt.bar(
        g_hist_x, g_hist_y,
        width=w,
        alpha=0.30,
        label="Global histogram",
        color="tab:cyan",
    )

    # ------------------ KDEs ------------------

    # Combined KDE (global scaled)
    plt.plot(
        kx, kde_comb_s,
        "k-", lw=2,
        label="Combined KDE (scaled)"
    )

    # Family-1 KDE
    plt.plot(
        f1["kde_x"], kde_f1_s,
        "r--", lw=2,
        label=f"Family 1 KDE (scaled, bw={f1['bw_deg']:.2f}deg)"
    )

    # Family-2 KDE
    plt.plot(
        f2["kde_x"], kde_f2_s,
        "b--", lw=2,
        label=f"Family 2 KDE (scaled, bw={f2['bw_deg']:.2f}deg)"
    )

    # ------------------ Peaks & split ------------------
    plt.axvline(p1, color="r", ls="-", lw=1)
    plt.axvline(p2, color="b", ls="-", lw=1)
    plt.axvline(split, color="g", ls=":", lw=2, label="Split angle")

    # ------------------ Layout ------------------
    plt.xlim(lo, hi)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Counts (hist) / KDE (scaled)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_famxy_on_image(
    img: np.ndarray,
    fam_dict: dict,
    color_x=(0, 0, 255),
    color_y=(255, 0, 0),
    draw_centers=True,
    center_radius=1,
    draw_pivot=True,
    thickness=1,
) -> np.ndarray:
    """
    Draw rotated X/Y line families on top of an image using OpenCV,
    compatible with the NEW family structure returned by
    reassign_and_rotate_families_by_image_center().

    fam_dict must contain:
        fam_dict["xfam"] = {
            "lines": (Nx,4),
            "centers": (Nx,2),
            ... other LSD fields ...
        }
        fam_dict["yfam"] = { same structure }
        fam_dict["pivot"] = (cx, cy)
    """

    out = img.copy()

    # Extract family dicts
    xfam = fam_dict.get("xfam", {})
    yfam = fam_dict.get("yfam", {})

    # Extract segments (safe fallback to empty arrays)
    x_lines = np.asarray(xfam.get("lines", np.zeros((0,4))), float)
    y_lines = np.asarray(yfam.get("lines", np.zeros((0,4))), float)

    # Extract centers if present
    x_centers = np.asarray(xfam.get("centers", np.zeros((0,2))), float)
    y_centers = np.asarray(yfam.get("centers", np.zeros((0,2))), float)

    # Extract pivot
    cx, cy = fam_dict.get("pivot", (None, None))

    # ----------------------------------------------------
    # Draw X-family (horizontal-ish) lines (default red)
    # ----------------------------------------------------
    for (x1, y1, x2, y2) in x_lines:
        cv2.line(
            out,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color_x,
            thickness,
            cv2.LINE_AA
        )

    # Optional centers
    if draw_centers and x_centers.size > 0:
        for (xc, yc) in x_centers:
            cv2.circle(out, (int(round(xc)), int(round(yc))),
                       center_radius, color_x, -1, cv2.LINE_AA)

    # ----------------------------------------------------
    # Draw Y-family (vertical-ish) lines (default blue)
    # ----------------------------------------------------
    for (x1, y1, x2, y2) in y_lines:
        cv2.line(
            out,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color_y,
            thickness,
            cv2.LINE_AA
        )

    if draw_centers and y_centers.size > 0:
        for (xc, yc) in y_centers:
            cv2.circle(out, (int(round(xc)), int(round(yc))),
                       center_radius, color_y, -1, cv2.LINE_AA)

    # ----------------------------------------------------
    # Draw pivot point
    # ----------------------------------------------------
    if draw_pivot and cx is not None and cy is not None:
        cx_i, cy_i = int(round(cx)), int(round(cy))
        s = 6
        cv2.line(out, (cx_i - s, cy_i), (cx_i + s, cy_i),
                 (0,255,0), 1, cv2.LINE_AA)
        cv2.line(out, (cx_i, cy_i - s), (cx_i, cy_i + s),
                 (0,255,0), 1, cv2.LINE_AA)

    return out


def print_angle_analysis_console(
    hist_info: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    fam_kdes: Dict[str, Any],
    title: str = "ANGLE ANALYSIS REPORT",
):
    """
    Pretty-print extended orientation family analysis to console.

    Now includes:
        - KDE bandwidth per family
        - Circular skewness / kurtosis per family
        - Effective sample size per family

    Parameters
    ----------
    hist_info : dict
        Output of compute_angle_histogram_circular_weighted()
    analysis : dict
        Output of analyze_two_orientation_families()
    fam_kdes : dict
        Output of compute_family_kdes()
    """

    lo, hi = hist_info["range"]
    p1, p2 = analysis["peaks"]
    rot = analysis.get("rotation_angle_deg")

    fam1 = analysis["family1"]
    fam2 = analysis["family2"]
    f1 = fam_kdes["fam1"]
    f2 = fam_kdes["fam2"]

    # ---------- circular skew & kurtosis ----------
    def _circular_moments(angles_deg: np.ndarray, weights: np.ndarray):
        if len(angles_deg) == 0:
            return None, None

        ang = np.deg2rad(angles_deg)
        w = weights / np.sum(weights)

        C = np.sum(w * np.cos(ang))
        S = np.sum(w * np.sin(ang))
        R = np.hypot(C, S)

        # harmonics
        C2 = np.sum(w * np.cos(2 * ang))
        S2 = np.sum(w * np.sin(2 * ang))

        # skew / kurtosis for circular
        if (1 - R) < 1e-9:
            return 0.0, 0.0

        skew = S2 / (1 - R)
        kurt = (C2 - R**2) / ((1 - R)**2)

        return float(skew), float(kurt)

    # ---------- formatting ----------
    def fmt(x):
        return f"{x:8.3f}" if isinstance(x, (float, np.floating)) else str(x)

    def print_family(name, stats, kde):
        skew, kurt = _circular_moments(kde["angles"], kde["weights"])
        bw = kde["bw_deg"]
        eff_n = (kde["weights"].sum())**2 / np.sum(kde["weights"]**2)

        print(f"{name}")
        print("-" * len(name))
        print(f"  Count                : {stats['count']}")
        print(f"  Total weight         : {stats['total_weight']:.3f}")
        print(f"  Mean angle (deg)     : {fmt(stats['mean_deg'])}")
        print(f"  Circular variance    : {fmt(stats['circ_var'])}")
        print(f"  Resultant length R   : {fmt(stats['R'])}")
        print(f"  Kappa (von Mises)    : {fmt(stats['kappa'])}")
        print(f"  KDE bandwidth (deg)  : {fmt(bw)}")
        print(f"  Skewness             : {fmt(skew)}")
        print(f"  Kurtosis             : {fmt(kurt)}")
        print(f"  Effective N          : {eff_n:8.1f}")
        print("")

    # ----------------------------------------------------------------------
    # PRINT REPORT
    # ----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)

    print(f"Angle range            : [{lo:.1f}deg, {hi:.1f}deg]")
    print(f"Total segments         : {hist_info['raw_count']}")
    print(f"Total weight           : {hist_info['raw_weight']:.3f}")
    print("")

    print("Detected peaks (deg)")
    print("-------------------")
    print(f"  Peak 1               : {p1:.3f}")
    print(f"  Peak 2               : {p2:.3f}")
    print(f"  Split angle          : {analysis['split_deg']:.3f}")
    if rot is not None:
        print(f"  Rotation (deskew)    : {rot:.3f} deg")
    print("")

    print_family("FAMILY 1", fam1, f1)
    print_family("FAMILY 2", fam2, f2)

    print("=" * 68 + "\n")


def print_family_stats_extended(
    name: str,
    stats: Dict[str, Any],
    fam_kde: Dict[str, Any],
):
    """
    Richer console output per family:
        - circular moments
        - KDE bandwidth
        - effective sample size
    """

    skew, kurt = _circular_moments(fam_kde["angles"], fam_kde["weights"])
    bw = fam_kde["bw_deg"]
    eff_n = (fam_kde["weights"].sum())**2 / np.sum(fam_kde["weights"]**2)

    def fmt(x):
        return f"{x:8.3f}" if isinstance(x, (float, np.floating)) else str(x)

    print(f"{name}")
    print("-" * len(name))
    print(f"  Count                : {stats['count']}")
    print(f"  Total weight         : {stats['total_weight']:.3f}")
    print(f"  Mean angle (deg)     : {fmt(stats['mean_deg'])}")
    print(f"  Circular variance    : {fmt(stats['circ_var'])}")
    print(f"  Resultant length R   : {fmt(stats['R'])}")
    print(f"  Kappa (von Mises)    : {fmt(stats['kappa'])}")
    print(f"  KDE bandwidth (deg)  : {fmt(bw)}")
    print(f"  Skewness             : {fmt(skew)}")
    print(f"  Kurtosis             : {fmt(kurt)}")
    print(f"  Effective N          : {eff_n:8.1f}")
    print("")


def _circular_silverman_bw(angles_rad: np.ndarray) -> float:
    """
    Silverman's rule adapted for circular domain.
    Returns bandwidth in radians.
    """
    if angles_rad.size < 2:
        return 0.2  # fallback small bw

    # unwrap to avoid artificial discontinuity
    unwrapped = np.unwrap(angles_rad)
    std = np.std(unwrapped)

    bw = 1.06 * std * (angles_rad.size ** (-1 / 5))
    return max(bw, np.deg2rad(1.0))  # avoid collapse


def _wrapped_gaussian_kde(
    angles_deg: np.ndarray,
    weights: np.ndarray,
    kde_x: np.ndarray,
    bw_rad: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Compute wrapped-Gaussian KDE on domain [lo, hi) using bandwidth bw_rad.
    """
    period = hi - lo
    out = np.zeros_like(kde_x)

    ang_rad = np.deg2rad(angles_deg)
    kx_rad = np.deg2rad(kde_x)

    for shift in (-period, 0.0, +period):
        shifted = np.deg2rad(angles_deg + shift)
        d = kx_rad[:, None] - shifted[None, :]
        out += np.sum(weights * np.exp(-(d ** 2) / (2 * bw_rad ** 2)), axis=1)

    out /= (np.sqrt(2 * np.pi) * bw_rad)
    return out / np.sum(out)  # normalize to unit mass        


def compute_family_kdes(
    hist: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    kde_samples: int = 720,
) -> Dict[str, Any]:
    """
    Compute per-family wrapped Gaussian KDEs using auto-bandwidth.
    Produce a fixed-format hist-like dict:
        fam1_hist_x, fam1_hist_y, fam1_kde_x, fam1_kde_y
        fam2_hist_x, fam2_hist_y, fam2_kde_x, fam2_kde_y
        comb_kde_x, comb_kde_y
    """

    lo, hi = hist["range"]
    period = hi - lo

    # Extract global info
    ang = hist["angles_deg"]
    w = hist["weights"]

    # Split using analysis masks
    p1, p2 = analysis["peaks"]
    d1 = np.abs(((ang - p1 + period/2) % period) - period/2)
    d2 = np.abs(((ang - p2 + period/2) % period) - period/2)
    mask1 = d1 <= d2
    mask2 = ~mask1

    a1, w1 = ang[mask1], w[mask1]
    a2, w2 = ang[mask2], w[mask2]

    # Histograms per family
    bins = len(hist["hist_y"])
    edges = np.linspace(lo, hi, bins + 1)

    h1, _ = np.histogram(a1, bins=bins, range=(lo, hi), weights=w1)
    h2, _ = np.histogram(a2, bins=bins, range=(lo, hi), weights=w2)
    hx = 0.5 * (edges[:-1] + edges[1:])

    # KDE X axis
    kde_x = np.linspace(lo, hi, kde_samples, endpoint=False)

    # Auto bandwidth per family
    bw1 = _circular_silverman_bw(np.deg2rad(a1)) if len(a1) else np.deg2rad(3)
    bw2 = _circular_silverman_bw(np.deg2rad(a2)) if len(a2) else np.deg2rad(3)

    # KDEs
    kde1 = _wrapped_gaussian_kde(a1, w1, kde_x, bw1, lo, hi) if len(a1) else np.zeros_like(kde_x)
    kde2 = _wrapped_gaussian_kde(a2, w2, kde_x, bw2, lo, hi) if len(a2) else np.zeros_like(kde_x)

    # Combined KDE
    total = w1.sum() + w2.sum()
    comb = kde1 * (w1.sum() / total) + kde2 * (w2.sum() / total)

    return {
        "fam1": {
            "angles": a1,
            "weights": w1,
            "hist_x": hx,
            "hist_y": h1,
            "kde_x": kde_x,
            "kde_y": kde1,
            "bw_deg": np.rad2deg(bw1),
        },
        "fam2": {
            "angles": a2,
            "weights": w2,
            "hist_x": hx,
            "hist_y": h2,
            "kde_x": kde_x,
            "kde_y": kde2,
            "bw_deg": np.rad2deg(bw2),
        },
        "combined": {
            "kde_x": kde_x,
            "kde_y": comb,
        }
    }


def _circular_moments(angles_deg: np.ndarray, weights: np.ndarray):
    """
    Return circular skewness and kurtosis from angles and weights.
    Fisher-style estimators for directional statistics.
    """
    if len(angles_deg) == 0:
        return None, None

    ang = np.deg2rad(angles_deg)
    w = weights / np.sum(weights)

    C = np.sum(w * np.cos(ang))
    S = np.sum(w * np.sin(ang))
    R = np.hypot(C, S)

    # 2nd and 3rd harmonics
    C2 = np.sum(w * np.cos(2 * ang))
    S2 = np.sum(w * np.sin(2 * ang))
    C3 = np.sum(w * np.cos(3 * ang))
    S3 = np.sum(w * np.sin(3 * ang))

    # Circular skewness & kurtosis
    skew = S2 / (1 - R) if (1 - R) > 1e-9 else 0.0
    kurt = (C2 - R**2) / (1 - R)**2 if (1 - R) > 1e-9 else 0.0

    return float(skew), float(kurt)


def plot_lsd_distributions(
    lsd_output: Dict[str, Any],
    title: str = "LSD Output Distributions",
    bins: int = 60,
    log_nfa: bool = True,
    fig_size=(12, 10),
    color_width="#1f77b4",
    color_precision="#ff7f0e",
    color_nfa="#2ca02c",
):
    """
    Plot stacked histograms of LSD outputs:
        - width distribution
        - precision distribution
        - NFA distribution (optionally log-scaled)

    Also prints to console:
        min / max / mean / median / std for width, precision, and NFA.
    """

    widths = np.asarray(lsd_output.get("widths", []), float)
    # Add percentile summary ONLY for width
    p95  = np.percentile(widths, 95)
    p99  = np.percentile(widths, 99)
    p997 = np.percentile(widths, 99.7)  # ~3sigma

    precisions = np.asarray(lsd_output.get("precisions", []), float)
    nfa = np.asarray(lsd_output.get("nfa", []), float)

    if widths.size == 0:
        print("plot_lsd_distributions: EMPTY LSD OUTPUT.")
        return

    # -------------------------------------------------------
    # Print statistics helper
    # -------------------------------------------------------
    def _print_stats(name: str, arr: np.ndarray):
        print(f"\n{name} statistics:")
        print(f"    count   = {arr.size}")
        print(f"    min     = {np.min(arr):.6g}")
        print(f"    max     = {np.max(arr):.6g}")
        print(f"    mean    = {np.mean(arr):.6g}")
        print(f"    median  = {np.median(arr):.6g}")
        print(f"    std     = {np.std(arr):.6g}")

    # -------------------------------------------------------
    # Console output: raw (non-log) values
    # -------------------------------------------------------
    _print_stats("Width", widths)

    print("    p95     = {:.6g}".format(p95))
    print("    p99     = {:.6g}".format(p99))
    print("    p99.7   = {:.6g}".format(p997))

    _print_stats("Precision", precisions)
    _print_stats("NFA (raw)", nfa)

    # -------------------------------------------------------
    # Prepare NFA display array
    # -------------------------------------------------------
    if log_nfa:
        eps = 1e-12
        nfa_disp = np.log10(nfa + eps)
        nfa_label = "log10(NFA)"
    else:
        nfa_disp = nfa
        nfa_label = "NFA"

    # -------------------------------------------------------
    # Create figure with 3 vertical subplots
    # -------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=fig_size)
    fig.suptitle(title, fontsize=16, weight="bold")

    # ----------------------
    # 1) WIDTH DISTRIBUTION
    # ----------------------
    ax[0].hist(widths, bins=bins, color=color_width, alpha=0.75)
    ax[0].set_title("Width Distribution")
    ax[0].set_ylabel("Count")
    ax[0].grid(True, ls=":", alpha=0.35)

    # ------------------------
    # 2) PRECISION DISTRIBUTION
    # ------------------------
    ax[1].hist(precisions, bins=bins, color=color_precision, alpha=0.75)
    ax[1].set_title("Precision Distribution")
    ax[1].set_ylabel("Count")
    ax[1].grid(True, ls=":", alpha=0.35)

    # -------------------
    # 3) NFA DISTRIBUTION
    # -------------------
    ax[2].hist(nfa_disp, bins=bins, color=color_nfa, alpha=0.75)
    ax[2].set_title(f"NFA Distribution ({nfa_label})")
    ax[2].set_xlabel("Value")
    ax[2].set_ylabel("Count")
    ax[2].grid(True, ls=":", alpha=0.35)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
