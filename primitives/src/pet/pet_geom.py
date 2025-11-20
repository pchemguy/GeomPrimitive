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
    """Create an OpenCV LSD detector with backward-compat fallbacks."""
    try:
        return cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
    except TypeError:
        try:
            return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except TypeError:
            return cv2.createLineSegmentDetector()


def detect_grid_segments(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run LSD (Line Segment Detector) and return *raw* segments only.

    Returns:
        {
            "lines": (N,4) float32
            "widths": (N,)
            "precisions": (N,)
            "nfa": (N,)
        }
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = _create_lsd()
    lines, widths, prec, nfa = lsd.detect(gray)

    if lines is None:
        return {
            "lines": np.zeros((0, 4), np.float32),
            "widths": np.zeros((0,), np.float32),
            "precisions": np.zeros((0,), np.float32),
            "nfa": np.zeros((0,), np.float32),
        }

    lines = lines.reshape(-1, 4).astype(np.float32)
    widths = widths.reshape(-1) if widths is not None else np.zeros(len(lines))
    prec = prec.reshape(-1) if prec is not None else np.zeros(len(lines))
    nfa = nfa.reshape(-1) if nfa is not None else np.zeros(len(lines))

    return {
        "lines": lines,
        "widths": widths,
        "precisions": prec,
        "nfa": nfa,
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

    dx = x2 - x1
    dy = y2 - y1

    # Raw angles in (-pi, pi]
    angles = np.arctan2(dy, dx)

    # Normalize angles to [-pi/4, 3*pi/4)  i.e. [-45deg, 135deg)
    angles_norm = (angles + np.pi / 4.0) % np.pi - np.pi / 4.0
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
    Includes:
        count, total_weight
        mean_deg, R, circular variance
        kappa (von Mises concentration)

    NOTE:
        Rayleigh test and linear-normality tests are intentionally removed.
        They are not meaningful for weighted circular data.
    """

    angles_deg = np.asarray(angles_deg, float)
    weights = np.asarray(weights, float)

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
        }

    # Normalize weights internally
    w_norm = weights / total_weight

    # Convert to radians
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

    # Kappa estimator (valid for moderate-high concentration)
    if R < 1e-6:
        kappa = 0.0
    elif R < 0.53:
        kappa = 2 * R + R**3 + 5 * (R**5) / 6
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa = 1.0 / (R**3 - 4 * R**2 + 3 * R)

    return {
        "count": count,
        "total_weight": total_weight,
        "mean_deg": mean_deg,
        "R": R,
        "circ_var": circ_var,
        "kappa": float(kappa),
    }


def analyze_two_orientation_families(
    angle_hist: Dict[str, np.ndarray],
    peak_prominence: float = 0.05,
    min_peak_distance_deg: float = 20.0,
) -> Dict[str, Any]:
    """
    Analyze a circular angle distribution that must contain exactly two
    dominant maxima. Uses KDE peaks for splitting.

    Returns an extended dict including:
        - peaks
        - split_deg
        - family1, family2 stats
        - rotation_angle_deg  (deskew angle, in [-45, +45])
    """

    # 1) Find the two dominant peaks
    p1, p2 = _find_two_kde_peaks_circular(
        angle_hist,
        peak_prominence=peak_prominence,
        min_peak_distance_deg=min_peak_distance_deg,
    )

    lo, hi = angle_hist["range"]
    period = hi - lo

    # 2) Mid-split angle (diagnostics only)
    diff_p = _ang_diff_signed_deg(p2, p1, period=period)
    split_deg = float(((p1 + 0.5 * diff_p) - lo) % period + lo)

    # 3) Assign angles to closest peak
    ang = angle_hist["angles_deg"]
    w   = angle_hist["weights"]

    d1 = _ang_dist_circular_deg(ang, p1, period=period)
    d2 = _ang_dist_circular_deg(ang, p2, period=period)

    mask1 = d1 <= d2
    mask2 = ~mask1

    ang1 = ang[mask1]
    ang2 = ang[mask2]
    w1   = w[mask1]
    w2   = w[mask2]

    # 4) Circular stats per family
    stats1 = _circular_family_stats(ang1, w1, angle_range=(lo, hi))
    stats2 = _circular_family_stats(ang2, w2, angle_range=(lo, hi))

    # ----------------------------------------------------------------------
    # 5) Compute rotation angle: smallest rotation moving dominant family
    #    mean angle into either 0° (horizontal) or 90° (vertical).
    # ----------------------------------------------------------------------

    # Determine dominant family by total weighted contribution
    if stats1["total_weight"] >= stats2["total_weight"]:
        mean_deg = stats1["mean_deg"]
    else:
        mean_deg = stats2["mean_deg"]

    # Normalize mean angle into [-45, +135)
    mean_deg_n = ((mean_deg - lo) % period) + lo

    # Candidates: align to 0° or 90°
    # rotation = target - mean
    rot_h = 0.0  - mean_deg_n      # to horizontal
    rot_v = 90.0 - mean_deg_n      # to vertical

    # Wrap both into [-90, +90] domain
    rot_h = ((rot_h + 90.0) % 180.0) - 90.0
    rot_v = ((rot_v + 90.0) % 180.0) - 90.0

    # Choose rotation with smaller absolute value
    if abs(rot_h) <= abs(rot_v):
        rotation_angle = rot_h
    else:
        rotation_angle = rot_v

    # Also clamp to [-45, +45] if preferred
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
    }


def apply_rotation_correction(
    img: np.ndarray,
    analysis: Dict[str, Any],
    border_value: Tuple[int, int, int] | int = 255,
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

    # Warp — preserve dtype
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


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
    segments: np.ndarray,
    angle_info: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    angle_range: Tuple[float, float] = (-45.0, 135.0),
) -> Dict[str, np.ndarray]:
    """
    Split raw segments into two angular families based on KDE-derived
    peak positions. Returns segments AND their corresponding angles.

    Returns:
        {
            "family1":  segments in family 1
            "family2":  segments in family 2
            "angles1":  angles belonging to family 1 (deg normalized)
            "angles2":  angles belonging to family 2 (deg normalized)
            "labels" :  array(N,) of 0/1 assignments
        }
    """

    if segments.size == 0:
        return {
            "family1": segments,
            "family2": segments,
            "angles1": np.zeros((0,), float),
            "angles2": np.zeros((0,), float),
            "labels": np.zeros(0, int),
        }

    # ----------------------------
    # Extract angles (correct field)
    # ----------------------------
    if "angles_deg" not in angle_info:
        raise KeyError("angle_info must contain 'angles_deg' from compute_segment_angles()")

    angles = np.asarray(angle_info["angles_deg"], float)

    # ----------------------------
    # Extract peak split from analysis
    # ----------------------------
    p1, p2 = analysis["peaks"]
    split_deg = float(analysis["split_deg"])

    # Normalize split into domain [-45, 135)
    amin, amax = angle_range
    split_norm = ((split_deg - amin) % 180.0) + amin

    # ----------------------------
    # Assign families based on strict split
    # ----------------------------
    labels = (angles > split_norm).astype(int)

    family1 = segments[labels == 0]
    family2 = segments[labels == 1]

    angles1 = angles[labels == 0]
    angles2 = angles[labels == 1]

    return {
        "family1": family1,
        "family2": family2,
        "angles1": angles1,
        "angles2": angles2,
        "labels": labels,
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


def print_angle_analysis_console(
    hist_info: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    title: str = "ANGLE ANALYSIS REPORT",
):
    """
    Pretty-print orientation family analysis to console.

    Expected analysis fields:
        - peaks : [p1, p2]
        - split_deg
        - family1 : stats dict
        - family2 : stats dict
        - rotation_angle_deg : deskew/rectification angle

    Expected hist_info fields:
        - raw_count
        - raw_weight
        - range
    """
    lo, hi = hist_info["range"]
    p1, p2 = analysis["peaks"]
    rot = analysis.get("rotation_angle_deg", None)

    fam1 = analysis["family1"]
    fam2 = analysis["family2"]

    def fmt(x):
        return f"{x:8.3f}" if isinstance(x, (float, np.floating)) else str(x)

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

    # ------- helper block -------
    def print_family(name, stats):
        print(f"{name}")
        print("-" * len(name))
        print(f"  Count                : {stats['count']}")
        print(f"  Total weight         : {stats['total_weight']:.3f}")
        print(f"  Mean angle (deg)     : {fmt(stats['mean_deg'])}")
        print(f"  Circular variance    : {fmt(stats['circ_var'])}")
        print(f"  Resultant length R   : {fmt(stats['R'])}")
        print(f"  K (von Mises)        : {fmt(stats['kappa'])}")
        print("")

    print_family("FAMILY 1", fam1)
    print_family("FAMILY 2", fam2)

    print("=" * 68 + "\n")
