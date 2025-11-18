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
    min_length_px: float = 20.0,
) -> Tuple[Dict, np.ndarray]:
    """
    Detect grid lines in a graph-paper image using LSD + angle clustering.

    Args:
        img:
            BGR uint8 source image (manually adjusted or raw).
        mark_grid:
            If True, overlays detected lines:
                - major-direction lines in RED
                - minor-direction lines in BLUE
        min_length_px:
            Minimum line length in pixels to keep (filters noise / short segments).

    Returns:
        meta: dict with:
            - lines_all: (N,4) array of all LSD lines
            - lines_x:   lines in cluster 0
            - lines_y:   lines in cluster 1
            - centers:   angle cluster centers
            - labels:    cluster label for each line

        out_img:
            BGR image with optional overlay if mark_grid=True,
            otherwise a copy of the input image.
    """
    # ---------------------------------------------------
    # PREP
    # ---------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LSD line detection
    lines_raw = _lsd_detect(gray)
    if len(lines_raw) == 0:
        return {
            "lines_all": np.zeros((0, 4)),
            "lines_x": np.zeros((0, 4)),
            "lines_y": np.zeros((0, 4)),
            "centers": None,
            "labels": [],
        }, img.copy()

    # ---------------------------------------------------
    # FILTER by length
    # ---------------------------------------------------
    def line_length(line):
        return np.hypot(line[2] - line[0], line[3] - line[1])

    lengths = np.array([line_length(l) for l in lines_raw])
    keep = lengths >= min_length_px
    lines = lines_raw[keep]

    if len(lines) < 2:
        return {
            "lines_all": lines,
            "lines_x": lines,
            "lines_y": np.zeros((0, 4)),
            "centers": None,
            "labels": [0] * len(lines),
        }, img.copy()

    # ---------------------------------------------------
    # Angle clustering into two directions
    # ---------------------------------------------------
    angles = np.array([_line_angle(l) for l in lines])
    centers, labels = _cluster_angles_into_two(angles)

    # cluster 0 & cluster 1
    lines_x = lines[np.array(labels) == 0]
    lines_y = lines[np.array(labels) == 1]

    # ---------------------------------------------------
    # Prepare output
    # ---------------------------------------------------
    out = img.copy()
    if mark_grid:
        # Draw major grid (cluster 0) in RED
        for (x1, y1, x2, y2) in lines_x:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 1, cv2.LINE_AA)
        # Draw minor grid (cluster 1) in BLUE
        for (x1, y1, x2, y2) in lines_y:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 0, 0), 1, cv2.LINE_AA)

    # ---------------------------------------------------
    # Metadata dictionary
    # ---------------------------------------------------
    meta = {
        "lines_all": lines,
        "lines_x": lines_x,
        "lines_y": lines_y,
        "centers": centers.tolist(),
        "labels": labels,
        "num_all": len(lines),
        "num_x": len(lines_x),
        "num_y": len(lines_y),
    }

    return meta, out
