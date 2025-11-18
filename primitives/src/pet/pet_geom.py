"""
pet_geom.py
-----------

Grid geometry utilities.

Functions:
    detect_grid_segments(img)          -> raw LSD segments
    filter_grid_segments(raw_segments) -> clustered, filtered line families
    mark_segments(img, segments, ...)  -> generic overlay
    mark_segment_families(img, ...)    -> overlay x/y families
"""

from __future__ import annotations
import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple


# ======================================================================
# 1) LSD DETECTOR (RAW, UNFILTERED)
# ======================================================================

def _create_lsd():
    try:
        return cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
    except TypeError:
        try:
            return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except TypeError:
            return cv2.createLineSegmentDetector()


def detect_grid_segments(img: np.ndarray) -> Dict:
    """
    Run LSD and return *raw segments only*.
    No filtering. No clustering.

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
            "lines": np.zeros((0,4), np.float32),
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


def _line_angle_deg(x1, y1, x2, y2):
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Map to [0,180)
    ang = ang % 180.0
    return ang


# ======================================================================
# 2) SEGMENT FILTERING + ANGLE CLUSTERING
# ======================================================================

def _line_angle(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _cluster_angles_two(angles: np.ndarray, iters: int = 12):
    ang = np.mod(angles, np.pi)
    c1, c2 = np.min(ang), np.max(ang)
    centers = np.array([c1, c2], dtype=np.float32)
    for _ in range(iters):
        d = np.abs(ang[:,None] - centers[None,:])
        labels = np.argmin(d, axis=1)
        for k in range(2):
            pts = ang[labels == k]
            if len(pts) > 0:
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
    angles = np.array([_line_angle(s) for s in segs])

    # -------------------------------------------------------
    # Cluster into 2 orientations
    # -------------------------------------------------------
    centers, labels = _cluster_angles_two(angles)
    c0, c1 = centers
    tol = np.deg2rad(angle_tol_deg)

    # -------------------------------------------------------
    # Filter by angle (ONLY)
    # -------------------------------------------------------
    good_idx = []

    for i, (ang, lab) in enumerate(zip(angles, labels)):
        target = c0 if lab == 0 else c1
        # Compute circular distance mod pi
        diff = abs((ang - target + np.pi/2) % np.pi - np.pi/2)
        if diff < tol:
            good_idx.append(i)

    kept = segs[good_idx]
    labels_good = np.array(labels)[good_idx]

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


def separate_line_families_kmeans(lines):
    """
    Separate lines into two families using k-means on their angles.
    Returns dict with 'family1', 'family2' arrays of shape (N,4).
    """
    if len(lines) < 4:
        return {"family1": lines, "family2": np.empty((0,4))}

    # Compute angles
    angles = np.array([
        _line_angle_deg(x1,y1,x2,y2) 
        for (x1,y1,x2,y2) in lines
    ], dtype=np.float64).reshape(-1,1)

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


# ======================================================================
# 3) VANISHING POINT ESTIMATION
# ======================================================================

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

    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue

        # Normal vector (a,b) perpendicular to direction (dx,dy)
        # Normalize so that sqrt(a^2 + b^2) == 1
        a = dy / length
        b = -dx / length
        c = -(a * x1 + b * y1)

        S_aa += a * a
        S_ab += a * b
        S_bb += b * b
        S_ac += a * c
        S_bc += b * c

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
    dists_sq = []
    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        a = dy / length
        b = -dx / length
        c = -(a * x1 + b * y1)
        d = a * x_vp + b * y_vp + c   # signed distance
        dists_sq.append(d * d)

    if not dists_sq:
        return (float(x_vp), float(y_vp)), None

    rms = float(np.sqrt(np.mean(dists_sq)))
    return (float(x_vp), float(y_vp)), rms


def _average_direction_angle(lines: np.ndarray) -> Optional[float]:
    """
    Compute a robust average direction angle (in radians) for a family
    of line segments, using the median of individual segment angles.
    """
    if lines is None or len(lines) == 0:
        return None
    angles = []
    for (x1, y1, x2, y2) in lines:
        angles.append(np.arctan2(y2 - y1, x2 - x1))
    angles = np.array(angles, dtype=np.float64)

    # Map to [0, pi) to keep sign-free direction
    ang_mod = np.mod(angles, np.pi)
    return float(np.median(ang_mod))


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


def _vp_orth_error_deg_for_center(
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

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1e9

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = float(np.clip(cosang, -1.0, 1.0))

    ang = np.rad2deg(np.arccos(cosang))
    return abs(ang - 90.0)


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

    best_err = 1e9
    best_cx = cx0
    best_cy = cy0

    # Coarse brute-force grid search
    for cx in xs:
        for cy in ys:
            err = _vp_orth_error_deg_for_center(vp_x, vp_y, cx, cy)
            if err < best_err:
                best_err = err
                best_cx = cx
                best_cy = cy

    return {
        "cx_refined": float(best_cx),
        "cy_refined": float(best_cy),
        "vp_orth_error_deg": float(best_err),
        "cx0": float(cx0),
        "cy0": float(cy0),
        "radius": float(r),
    }


def _family_direction_vector(lines: np.ndarray) -> np.ndarray:
    """
    Compute a robust average direction vector (2D) for a line family.

    Returns a unit vector [dx, dy] (float64). If no lines, returns [1, 0].
    """
    if lines is None or len(lines) == 0:
        return np.array([1.0, 0.0], dtype=np.float64)

    dirs = []
    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        n = np.hypot(dx, dy)
        if n < 1e-6:
            continue
        # Normalize each segment direction, ignore sign (we only care about axis)
        ux = dx / n
        uy = dy / n
        # Make direction sign-invariant: flip to keep angles in [-90, 90)
        if ux < 0:
            ux = -ux
            uy = -uy
        dirs.append([ux, uy])

    if not dirs:
        return np.array([1.0, 0.0], dtype=np.float64)

    dirs = np.array(dirs, dtype=np.float64)

    # Average and renormalize
    mean = dirs.mean(axis=0)
    nmean = np.hypot(mean[0], mean[1])
    if nmean < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)

    return mean / nmean


# ======================================================================
# 4) Metric Rectification from Two Orthogonal Vanishing Points
# ======================================================================

def compute_rectifying_homography(
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
    cx: float,
    cy: float,
    return_affine=False,
) -> Dict:
    """
    Compute a metric rectification homography from two orthogonal
    vanishing points and a refined principal point.

    Args:
        vp_x, vp_y: (x,y) vanishing points for the two orthogonal axes.
        cx, cy: refined principal point.
        return_affine: if True, return the intermediate affine step.

    Returns:
        {
            'H': full 3x3 rectifying homography (metric rectification)
            'H_m': metric part of H
            'H_a': affine part (if return_affine=True else None)
        }
    """

    # Homogeneous coords w.r.t. principal point
    px = np.array([vp_x[0] - cx, vp_x[1] - cy, 1.0])
    py = np.array([vp_y[0] - cx, vp_y[1] - cy, 1.0])

    # Normalize directions (not strictly necessary)
    px = px / np.linalg.norm(px[:2])
    py = py / np.linalg.norm(py[:2])

    #
    # Step 1: Solve for the Image of the Absolute Conic (IAC) entries
    #
    # Orthogonality constraint:
    #   px^T * w * py = 0
    #
    # For square pixels with unknown scale:
    #   w = [[a, 0, 0],
    #        [0, a, 0],
    #        [0, 0, 1]]
    #
    # For orthogonality:
    #   a*(px_x * py_x + px_y * py_y) + 1 * (px_z * py_z) = 0
    #
    # Solve for 'a' (unique scale).
    #

    dot_xy = px[0]*py[0] + px[1]*py[1]
    dot_z  = px[2]*py[2]

    if abs(dot_xy) < 1e-9:
        raise RuntimeError("Degenerate case: cannot solve IAC for metric homography.")

    a = -dot_z / dot_xy   # IAC scale factor

    # Construct IAC matrix
    w = np.array([
        [a,   0., 0.],
        [0.,  a,  0.],
        [0.,  0., 1.],
    ])

    #
    # Step 2: Retrieve rectifying transform H_m from IAC
    #
    # Find Cholesky factor L such that:
    #   w = L^T * L
    #
    # Then H_m = L^{-1}
    #
    # This transforms the image into a metric plane where IAC = identity.
    #

    # Ensure symmetric
    w = 0.5 * (w + w.T)

    # Cholesky or SVD fallback
    try:
        L = np.linalg.cholesky(w)
    except np.linalg.LinAlgError:
        # fallback: w = U S U^T, take L = sqrt(S) U^T
        U, S, _ = np.linalg.svd(w)
        L = np.sqrt(np.diag(S)) @ U.T

    H_m = np.linalg.inv(L)

    #
    # Step 3: Add translation to account for principal point offset
    #
    H_center = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0,  1 ],
    ])

    # Full homography (metric correction AFTER centering)
    H_full = H_m @ H_center

    result = {
        "H": H_full,
        "H_m": H_m,
    }

    if return_affine:
        result["H_a"] = None  # placeholder for future affine stage

    return result


def _build_affine_metric_homography_from_lines(
    lines_x: np.ndarray,
    lines_y: np.ndarray,
) -> np.ndarray:
    """
    Build a purely affine+metric homography H (3x3, last row [0,0,1]) that:

        - makes the two line families orthogonal
        - aligns them with the coordinate axes
        - does NOT care which family becomes horizontal/vertical
        - implicitly uses a small rotation (because original dirs are already near orthogonal)

    This is numerically very stable and does not cause huge image blow-ups.
    """
    # Average directions for both families in image coords
    v1 = _family_direction_vector(lines_x)
    v2 = _family_direction_vector(lines_y)

    # Make v2 orthogonal to v1 via Gram-Schmidt, to get a clean basis
    # v1 is unit-length
    proj = np.dot(v2, v1) * v1
    v2_orth = v2 - proj
    n2 = np.hypot(v2_orth[0], v2_orth[1])
    if n2 < 1e-9:
        # Degenerate (families nearly collinear) -> fall back to identity
        return np.eye(3, dtype=np.float64)
    v2_orth /= n2

    # Construct basis matrix B whose columns are v1, v2_orth
    # B maps the [grid basis] to image basis. We want the inverse:
    # H_lin: (x,y) -> B^{-1} * [x,y], i.e., make grid basis align to (1,0) and (0,1).
    B = np.column_stack([v1, v2_orth])  # shape (2,2)

    det = np.linalg.det(B)
    if abs(det) < 1e-9:
        # Degenerate -> identity
        return np.eye(3, dtype=np.float64)

    M2 = np.linalg.inv(B)

    # Embed into 3x3 homography
    H_lin = np.array([
            [M2[0, 0], M2[0, 1], 0.0],
            [M2[1, 0], M2[1, 1], 0.0],
            [0.0,      0.0,      1.0],
    ], dtype=np.float64)

    return H_lin


# ======================================================================
# 5) Optional: Warp image using rectifying H
# ======================================================================

def warp_with_homography(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H to image using cv2.warpPerspective.
    Result is auto-sized to include full rectified grid.
    """
    H_inv = np.linalg.inv(H)

    # Estimate output bounds by transforming corners
    h, w = img.shape[:2]
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    ones = np.ones((4,1), dtype=np.float32)
    c_h = np.hstack((corners, ones))

    warped = (H @ c_h.T).T
    xs = warped[:,0] / warped[:,2]
    ys = warped[:,1] / warped[:,2]

    xmin, xmax = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    ymin, ymax = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    W_out = xmax - xmin
    H_out = ymax - ymin

    # Translation to keep all corners positive
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1     ]
    ], dtype=np.float64)

    return cv2.warpPerspective(img, T @ H, (W_out, H_out), flags=cv2.INTER_LINEAR)


def warp_with_homography_clamped(
    img: np.ndarray,
    H: np.ndarray,
    scale_limit: float = 2.0,
) -> np.ndarray:
    """
    Warp image with homography H, but clamp output size to avoid huge images.
    """
    h, w = img.shape[:2]

    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    ones = np.ones((4, 1), dtype=np.float32)
    c_h = np.hstack((corners, ones))
    warped = (H @ c_h.T).T
    xs = warped[:, 0] / warped[:, 2]
    ys = warped[:, 1] / warped[:, 2]

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    W_out = xmax - xmin
    H_out = ymax - ymin

    max_w = int(min(W_out, w * scale_limit))
    max_h = int(min(H_out, h * scale_limit))

    if max_w <= 0 or max_h <= 0:
        return img.copy()

    tx = -xmin
    ty = -ymin
    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_total = T @ H

    rect = cv2.warpPerspective(img, H_total, (int(max_w), int(max_h)),
                               flags=cv2.INTER_LINEAR)
    return rect


def rectify_grid_affine(
    img: np.ndarray,
    lines_x: np.ndarray,
    lines_y: np.ndarray,
    scale_limit: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level affine+metric rectification:

      - Uses the two grid line families to build a stable linear homography
      - Makes the families orthogonal and axis-aligned
      - Does not care which family becomes horizontal/vertical
      - Implicitly uses the small rotation induced by line directions
      - Clamps output size to avoid huge images

    Args:
      img: original BGR image.
      lines_x, lines_y: (N1,4) and (N2,4) segment arrays from filter_grid_segments.

    Returns:
      rectified_img, H_lin
    """
    H_lin = _build_affine_metric_homography_from_lines(lines_x, lines_y)
    rect = warp_with_homography_clamped(img, H_lin, scale_limit=scale_limit)
    return rect, H_lin


def rectify_grid_projective(
    img: np.ndarray,
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
    lines_x: np.ndarray,
    lines_y: np.ndarray,
    scale_limit: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full projective rectification of the graph-paper plane:

      1) Projective -> affine rectification using vanishing points
      2) Metric + axis alignment using line families
      3) Warp original image with combined homography

    Args:
        img: original BGR image.
        vp_x, vp_y: vanishing points for the two grid directions.
        lines_x, lines_y: (N1,4) and (N2,4) segment arrays in original coords.
        scale_limit: clamp for output size vs original size.

    Returns:
        rectified_img, H_total  (3x3 homography in original image coords)
    """
    # 1) Projective -> affine
    H_A = affine_rectification_from_vps(vp_x, vp_y)

    # Transform segments into affine-rectified coordinates
    lines_x_aff = _transform_segments(H_A, lines_x)
    lines_y_aff = _transform_segments(H_A, lines_y)

    # 2) Metric + axis alignment in affine space
    H_M = _build_metric_axis_homography_from_affine_lines(
        lines_x_aff, lines_y_aff
    )

    # Combined homography (projective + metric)
    H_total = H_M @ H_A

    # 3) Warp original image
    rectified = warp_with_homography_clamped(
        img, H_total, scale_limit=scale_limit
    )

    return rectified, H_total


def _transform_segments(H: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Apply homography H to an array of segments of shape (N, 4):
    [x1, y1, x2, y2] -> transformed in the same format.
    """
    if segments is None or len(segments) == 0:
        return np.zeros((0, 4), dtype=np.float64)

    seg = segments.astype(np.float64)
    p1 = np.c_[seg[:, 0:2], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2:4], np.ones(len(seg))]

    tp1 = (H @ p1.T).T
    tp2 = (H @ p2.T).T

    tp1_xy = tp1[:, :2] / tp1[:, 2:3]
    tp2_xy = tp2[:, :2] / tp2[:, 2:3]

    return np.c_[tp1_xy, tp2_xy]


def _family_direction_vector(lines: np.ndarray) -> np.ndarray:
    """
    Robust average direction vector (unit 2D) for a line family.
    """
    if lines is None or len(lines) == 0:
        return np.array([1.0, 0.0], dtype=np.float64)

    dirs = []
    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        n = np.hypot(dx, dy)
        if n < 1e-6:
            continue
        ux, uy = dx / n, dy / n

        # make sign-invariant (we only care about axis, not direction)
        if ux < 0:
            ux, uy = -ux, -uy
        dirs.append([ux, uy])

    if not dirs:
        return np.array([1.0, 0.0], dtype=np.float64)

    dirs = np.array(dirs, dtype=np.float64)
    mean = dirs.mean(axis=0)
    nmean = np.hypot(mean[0], mean[1])
    if nmean < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return mean / nmean


def affine_rectification_from_vps(
    vp_x: Tuple[float, float],
    vp_y: Tuple[float, float],
) -> np.ndarray:
    """
    Compute a projective homography H_A that rectifies the grid plane
    *affinely* by sending the horizon (vanishing line) to infinity.

    After applying H_A, parallel grid lines become globally parallel
    (no perspective convergence), but may still not be orthogonal.
    """
    # homogeneous vanishing points
    v1 = np.array([vp_x[0], vp_x[1], 1.0], dtype=np.float64)
    v2 = np.array([vp_y[0], vp_y[1], 1.0], dtype=np.float64)

    # horizon line l_inf = v1 x v2 = [a, b, c]
    l = np.cross(v1, v2)
    a, b, c = l

    if abs(c) < 1e-9:
        # Degenerate: horizon at infinity already - return identity
        return np.eye(3, dtype=np.float64)

    # H_A of the form [[1,0,0],[0,1,0],[p,q,1]]
    # We want H_A^{-T} * l ~ [0,0,1]^T -> p=a/c, q=b/c
    p = a / c
    q = b / c

    H_A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [p,   q,   1.0],
    ], dtype=np.float64)

    return H_A


def _build_metric_axis_homography_from_affine_lines(
    lines_x_aff: np.ndarray,
    lines_y_aff: np.ndarray,
) -> np.ndarray:
    """
    Given two line families in an *affine-rectified* coordinate system,
    build a 3x3 homography H_M (linear, last row [0,0,1]) that:

      - makes the families orthogonal
      - aligns them with coordinate axes
      - implicitly picks the nearest orientation (no arbitrary 180deg flips)
    """
    v1 = _family_direction_vector(lines_x_aff)
    v2 = _family_direction_vector(lines_y_aff)

    # Gram-Schmidt: make v2 orthogonal to v1
    proj = np.dot(v2, v1) * v1
    v2_orth = v2 - proj
    n2 = np.hypot(v2_orth[0], v2_orth[1])
    if n2 < 1e-9:
        # Degenerate; fall back to identity
        return np.eye(3, dtype=np.float64)
    v2_orth /= n2

    # Columns of B are the current (affine) basis vectors in image coords.
    B = np.column_stack([v1, v2_orth])  # shape (2,2)

    det = np.linalg.det(B)
    if abs(det) < 1e-9:
        return np.eye(3, dtype=np.float64)

    M2 = np.linalg.inv(B)  # sends v1->(1,0), v2_orth->(0,1)

    # Embed as homography
    H_M = np.array([
        [M2[0, 0], M2[0, 1], 0.0],
        [M2[1, 0], M2[1, 1], 0.0],
        [0.0,      0.0,      1.0],
    ], dtype=np.float64)

    return H_M


def warp_homography_optimal_size(img, H):
    h, w = img.shape[:2]

    # corners in homogeneous coords
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float64)

    warped = (H @ corners.T).T
    xs = warped[:, 0] / warped[:, 2]
    ys = warped[:, 1] / warped[:, 2]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    width  = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))

    # translation to shift into positive region
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0,     1]
    ], dtype=np.float64)

    H2 = T @ H

    rect = cv2.warpPerspective(img, H2, (width, height),
                               flags=cv2.INTER_LINEAR)
    return rect, H2


# ======================================================================
# VISUALIZATION
# ======================================================================

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


def debug_draw_raw_segments(img, segments, path="debug_raw_lsd.jpg"):
    vis = img.copy()
    for (x1, y1, x2, y2) in segments.astype(int):
        cv2.line(vis, (x1,y1), (x2,y2), (0,0,255), 1)
    cv2.imwrite(path, vis)
    return vis


def debug_draw_families(img, fam1, fam2, path="debug_families.jpg"):
    vis = img.copy()

    # family 1: red
    for (x1,y1,x2,y2) in fam1.astype(int):
        cv2.line(vis, (x1,y1), (x2,y2), (0,0,255), 1)

    # family 2: green
    for (x1,y1,x2,y2) in fam2.astype(int):
        cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 1)

    cv2.imwrite(path, vis)
    return vis


def print_family_angles(fams):
    ang1 = []
    for (x1,y1,x2,y2) in fams["family1"]:
        ang1.append(_line_angle_deg(x1,y1,x2,y2))
    ang2 = []
    for (x1,y1,x2,y2) in fams["family2"]:
        ang2.append(_line_angle_deg(x1,y1,x2,y2))

    print("Family 1 median:", np.median(ang1))
    print("Family 1 mean:  ", np.mean(ang1))
    print("Family 1 std:   ", np.std(ang1))

    print("Family 2 median:", np.median(ang2))
    print("Family 2 mean:  ", np.mean(ang2))
    print("Family 2 std:   ", np.std(ang2))

    print("Angle difference (median):", abs(np.median(ang1) - np.median(ang2)))

