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
# 4) SIMPLE KMEANS-BASED FAMILY SPLIT
# ============================================================================

def separate_line_families_kmeans(lines):
    """
    Separate lines into two families using k-means on their angles.
    Returns dict with 'family1', 'family2' arrays of shape (N,4).
    """
    if len(lines) < 4:
        return {"family1": lines, "family2": np.empty((0,4))}

    # Compute angles
    dx = lines[:,2] - lines[:,0]
    dy = lines[:,3] - lines[:,1]
    angles = (np.degrees(np.arctan2(dy, dx)) % 180.0).reshape(-1,1)

    # Run k-means
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = kmeans.fit_predict(angles)

    fam1 = lines[labels == 0]
    fam2 = lines[labels == 1]

    # Optional: normalize family angles around +/-90/0
    # to ensure consistency (smaller median first)
    med1 = np.median(angles[labels==0])
    med2 = np.median(angles[labels==1])
    if med1 > med2:
        fam1, fam2 = fam2, fam1

    return {"family1": fam1, "family2": fam2}


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


def _vp_orth_error_deg(
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
    cx: float,
    cy: float
) -> float:
    """
    Compute VP orthogonality error (in degrees) for a given principal point (cx,cy).
    """
    (vx1, vy1) = vp_x
    (vx2, vy2) = vp_y

    v1 = np.array([vx1 - cx, vy1 - cy], dtype=np.float64)
    v2 = np.array([vx2 - cx, vy2 - cy], dtype=np.float64)

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1e9

    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    return abs(np.rad2deg(np.arccos(cosang)) - 90.0)


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
