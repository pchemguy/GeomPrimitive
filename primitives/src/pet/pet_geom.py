"""
pet_geom.py
-----------

Minimal, clean utilities for grid geometry extraction:

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

def _angle_rad(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _angle_deg(x1, y1, x2, y2) -> float:
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return ang % 180.0


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
    raw: Dict[str, np.ndarray],
    angle_tol_deg: float = 20.0,
) -> Dict[str, np.ndarray]:
    """
    Minimal grid filtering:
        - compute segment angles
        - cluster into 2 families
        - keep only segments within ±tol of each cluster

    No length filtering.
    No weighting.
    Only directional filtering.
    """
    segs = raw["lines"]
    if len(segs) == 0:
        return {
            "lines_x": np.zeros((0, 4)),
            "lines_y": np.zeros((0, 4)),
            "centers": [None, None],
            "labels": [],
            "kept_segments": np.zeros((0, 4)),
        }

    angles = np.array([_angle_rad(s) for s in segs], float)
    centers, labels = _cluster_angles_two(angles)
    c0, c1 = centers
    tol = np.deg2rad(angle_tol_deg)

    good_idx = []
    for i, (ang, lab) in enumerate(zip(angles, labels)):
        target = c0 if lab == 0 else c1
        diff = abs((ang - target + np.pi/2) % np.pi - np.pi/2)
        if diff < tol:
            good_idx.append(i)

    kept = segs[good_idx]
    labels_good = np.array(labels)[good_idx]

    return {
        "lines_x": kept[labels_good == 0],
        "lines_y": kept[labels_good == 1],
        "centers": centers.tolist(),
        "labels": labels_good.tolist(),
        "kept_segments": kept,
    }


# ============================================================================
# 4) SIMPLE KMEANS-BASED FAMILY SPLIT
# ============================================================================

def separate_line_families_kmeans(lines: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Clusters lines into two families purely by angle (in degrees).

    Returns:
        {"family1": ..., "family2": ...}
    """
    if len(lines) < 4:
        return {"family1": lines, "family2": np.zeros((0, 4))}

    angs = np.array(
        [_angle_deg(*seg) for seg in lines],
        dtype=np.float64
    ).reshape(-1, 1)

    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(angs)

    fam1 = lines[labels == 0]
    fam2 = lines[labels == 1]

    # Ensure stable ordering (family1 = smaller median angle)
    if np.median(angs[labels == 0]) > np.median(angs[labels == 1]):
        fam1, fam2 = fam2, fam1

    return {"family1": fam1, "family2": fam2}


# ============================================================================
# 5) VANISHING POINT ESTIMATION (LEAST SQUARES)
# ============================================================================

def _fit_vp(lines: np.ndarray, min_lines=2):
    """Least-squares vanishing point for one line family."""
    if lines is None or len(lines) < min_lines:
        return None, None

    S_aa = S_ab = S_bb = S_ac = S_bc = 0.0

    for (x1, y1, x2, y2) in lines:
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 1e-6:
            continue

        a = dy / L
        b = -dx / L
        c = -(a * x1 + b * y1)

        S_aa += a*a
        S_ab += a*b
        S_bb += b*b
        S_ac += a*c
        S_bc += b*c

    det = S_aa * S_bb - S_ab * S_ab
    if abs(det) < 1e-9:
        return None, None

    inv_aa = S_bb / det
    inv_ab = -S_ab / det
    inv_bb = S_aa / det

    bx, by = -S_ac, -S_bc
    x_vp = inv_aa * bx + inv_ab * by
    y_vp = inv_ab * bx + inv_bb * by

    # RMS
    d2 = []
    for (x1, y1, x2, y2) in lines:
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 1e-6:
            continue
        a = dy / L
        b = -dx / L
        c = -(a * x1 + b * y1)
        d = a * x_vp + b * y_vp + c
        d2.append(d*d)

    rms = float(np.sqrt(np.mean(d2))) if d2 else None
    return (float(x_vp), float(y_vp)), rms


def estimate_vanishing_points(
    lines_x: np.ndarray,
    lines_y: np.ndarray,
    img_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Optional[float]]:
    """
    Estimate two vanishing points and basic orthogonality metrics.
    """
    vp_x, rms_x = _fit_vp(lines_x)
    vp_y, rms_y = _fit_vp(lines_y)

    ang_x = _median_direction(lines_x)
    ang_y = _median_direction(lines_y)

    angle_orth = None
    if ang_x is not None and ang_y is not None:
        ax = (ang_x + np.pi) % np.pi
        ay = (ang_y + np.pi) % np.pi
        d = abs(ax - ay)
        if d > np.pi/2:
            d = np.pi - d
        angle_orth = abs(np.rad2deg(d) - 90.0)

    vp_orth = None
    if vp_x and vp_y and img_shape:
        H, W = img_shape
        cx, cy = W/2, H/2
        v1 = np.array([vp_x[0]-cx, vp_x[1]-cy])
        v2 = np.array([vp_y[0]-cx, vp_y[1]-cy])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1, 1)
            vp_orth = abs(np.rad2deg(np.arccos(cosang)) - 90.0)

    return {
        "vp_x": vp_x,
        "vp_y": vp_y,
        "rms_x": rms_x,
        "rms_y": rms_y,
        "angle_x": ang_x,
        "angle_y": ang_y,
        "angle_orth_error_deg": angle_orth,
        "vp_orth_error_deg": vp_orth,
    }


def _median_direction(lines: np.ndarray) -> Optional[float]:
    if lines is None or len(lines) == 0:
        return None
    ang = np.array([_angle_rad(s) for s in lines], float)
    return float(np.median(np.mod(ang, np.pi)))


# ============================================================================
# 6) PRINCIPAL POINT REFINEMENT
# ============================================================================

def refine_principal_point_from_vps(
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
    img_shape: Tuple[int, int],
    radius_frac: float = 0.05,
    steps: int = 20,
) -> Dict[str, float]:
    """
    Small brute-force search around image center to minimize VP orthogonality.
    """
    H, W = img_shape
    cx0, cy0 = W/2, H/2
    r = radius_frac * min(H, W)

    xs = np.linspace(cx0 - r, cx0 + r, steps)
    ys = np.linspace(cy0 - r, cy0 + r, steps)

    best = 1e9
    best_cx = cx0
    best_cy = cy0

    for cx in xs:
        for cy in ys:
            err = _vp_orth_error(vp_x, vp_y, cx, cy)
            if err < best:
                best = err
                best_cx, best_cy = cx, cy

    return {
        "cx_refined": float(best_cx),
        "cy_refined": float(best_cy),
        "vp_orth_error_deg": float(best),
        "cx0": float(cx0),
        "cy0": float(cy0),
        "radius": float(r),
    }


def _vp_orth_error(vp_x, vp_y, cx, cy) -> float:
    v1 = np.array([vp_x[0]-cx, vp_x[1]-cy], float)
    v2 = np.array([vp_y[0]-cx, vp_y[1]-cy], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1e9
    cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1, 1)
    return abs(np.rad2deg(np.arccos(cosang)) - 90.0)


# ============================================================================
# 7) BASIC VISUALIZATION
# ============================================================================

def mark_segments(img, segments, color=(0,255,0), thickness=1):
    """Draw arbitrary segments (N,4) onto a copy of the BGR image."""
    out = img.copy()
    for (x1, y1, x2, y2) in segments:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color, thickness, cv2.LINE_AA)
    return out


def mark_segment_families(
    img, lines_x, lines_y,
    color_x=(0,0,255),    # X-family in red
    color_y=(255,0,0),    # Y-family in blue
    thickness=1
):
    """Overlay two segment families with different colors."""
    out = img.copy()
    for (x1,y1,x2,y2) in lines_x:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color_x, thickness, cv2.LINE_AA)
    for (x1,y1,x2,y2) in lines_y:
        cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)),
                 color_y, thickness, cv2.LINE_AA)
    return out
