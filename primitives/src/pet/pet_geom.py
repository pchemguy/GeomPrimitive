"""
pet_geom.py
-----------

Geometry and grid analysis utilities for PET.

This module begins with:
    detect_grid_lines(img, mark_grid=False)

which detects graph-paper grid lines using LSD (Line Segment Detector),
clusters them into two principal directions, and returns them for later
vanishing-point and homography estimation.

Dependencies:
    - OpenCV (cv2)
    - numpy (np)
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List, Tuple


# ======================================================================
# 1) HELPER: Convert raw LSD output into a clean list of line segments
# ======================================================================

def _lsd_detect(img_gray: np.ndarray):
    """
    Run OpenCV's LSD (Line Segment Detector) on a grayscale image,
    supporting different OpenCV versions.

    Returns:
        lines_raw: array of shape (N, 4) in format [x1, y1, x2, y2]
    """

    # OpenCV 4.7+ uses "refine="
    # Some older versions accept no keyword arguments at all.
    try:
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
    except TypeError:
        try:
            # Older OpenCV using positional refine parameter
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except TypeError:
            # Very old OpenCV versions require no refine mode
            lsd = cv2.createLineSegmentDetector()

    # Run LSD
    lines, widths, prec, nfa = lsd.detect(img_gray)

    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)

    # Flatten (N,1,4) -> (N,4)
    lines = lines.reshape(-1, 4).astype(np.float32)
    return lines


# ======================================================================
# 2) HELPER: Compute angle of a line segment in radians
# ======================================================================

def _line_angle(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    return float(np.arctan2(dy, dx))


# ======================================================================
# 3) HELPER: K-means (1D) for angles into 2 directions
# ======================================================================

def _cluster_angles_into_two(angles: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Simple 1D k-means on angles (radians), k=2.
    Angles mod pi => grid directions are separated by ~90 degrees.
    """
    # Map angles into [0, pi)
    ang = np.mod(angles, np.pi)

    # Initialize 2 cluster centers by picking min and max
    c1, c2 = np.min(ang), np.max(ang)
    centers = np.array([c1, c2], dtype=np.float32)

    for _ in range(12):  # small fixed-iteration kmeans
        # Assign
        dists = np.abs(ang[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)

        # Update
        for k in range(2):
            pts = ang[labels == k]
            if len(pts) > 0:
                centers[k] = np.mean(pts)

    return centers, labels.tolist()


# ======================================================================
# 4) MAIN FUNCTION: detect_grid_lines
# ======================================================================

def detect_grid_lines(
    img: np.ndarray,
    mark_grid: bool = False,
    min_length_px: float = 40.0,
    angle_tol_deg: float = 8.0,
) -> Tuple[Dict, np.ndarray]:
    """
    Detect graph-paper grid lines with robust filtering.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) LSD detection
    lines_raw = _lsd_detect(gray)
    if len(lines_raw) == 0:
        return {"lines_all": [], "lines_x": [], "lines_y": []}, img.copy()

    # 2) Filter by length
    lens = np.hypot(lines_raw[:, 2] - lines_raw[:, 0],
                    lines_raw[:, 3] - lines_raw[:, 1])
    keep = lens >= min_length_px
    lines = lines_raw[keep]

    if len(lines) < 4:
        return {"lines_all": lines, "lines_x": lines, "lines_y": []}, img.copy()

    # 3) Compute angles & cluster
    angles = np.array([_line_angle(l) for l in lines])
    centers, labels = _cluster_angles_into_two(angles)
    center0, center1 = centers

    # 4) Angle refinement (remove off-angle lines)
    angle_tol = np.deg2rad(angle_tol_deg)
    keep_mask = np.zeros(len(lines), dtype=bool)

    for i, (ang, lab) in enumerate(zip(angles, labels)):
        target = center0 if lab == 0 else center1
        if abs((ang - target + np.pi/2) % np.pi - np.pi/2) < angle_tol:
            keep_mask[i] = True

    lines = lines[keep_mask]
    labels = np.array(labels)[keep_mask]

    # Split into two families
    lines_x = lines[labels == 0]
    lines_y = lines[labels == 1]

    # 5) Sort by line intercepts (rho in normal form)
    #    rho = x*cos theta + y*sin theta
    def compute_rho(line, theta):
        x1, y1, x2, y2 = line
        return (x1*np.cos(theta) + y1*np.sin(theta) +
                x2*np.cos(theta) + y2*np.sin(theta)) * 0.5

    lines_x_sorted = sorted(
        lines_x, key=lambda l: compute_rho(l, center0)
    )
    lines_y_sorted = sorted(
        lines_y, key=lambda l: compute_rho(l, center1)
    )

    # Prepare marked output
    out = img.copy()
    if mark_grid:
        for (x1, y1, x2, y2) in lines_x_sorted:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 1, cv2.LINE_AA)
        for (x1, y1, x2, y2) in lines_y_sorted:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (int(x2), int(y2)), (255, 0, 0), 1, cv2.LINE_AA)

    meta = {
        "lines_all": lines,
        "lines_x": np.array(lines_x_sorted),
        "lines_y": np.array(lines_y_sorted),
        "centers": centers.tolist(),
        "num_x": len(lines_x_sorted),
        "num_y": len(lines_y_sorted),
    }

    return meta, out
