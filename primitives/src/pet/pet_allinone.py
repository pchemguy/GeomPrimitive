"""
pet_allinone.py
----------------

Prototype orchestration module for Paper Enhancement & Transformation (PET).

This version:
- Defines module-level logger name.
- Defines sample image path.
- Implements image_loader() that loads image, logs metadata.
- All components use the same logger.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional, Tuple

# FORCE-DISABLE joblib multiprocessing
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_START_METHOD"] = "threading"

# Also disable MKL / BLAS threading (optional but recommended)
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

from pet_utils import image_loader, save_image, LOGGER_NAME

from pet_geom import (
    detect_grid_segments,
    compute_segment_angles, compute_angle_histogram,
    plot_angle_histogram, plot_angle_histogram_with_kde, apply_rotation_correction,
    compute_angle_histogram_circular_weighted, compute_segment_lengths,
    analyze_two_orientation_families, print_angle_analysis_console,
    compute_family_kdes, plot_family_kdes,
    filter_grid_segments, estimate_vanishing_points,
    refine_principal_point_from_vps, split_segments_by_angle_circular,
    mark_segments, mark_segment_families, 
)

from pet_utils import image_loader, save_image


# ======================================================================
# MODULE CONSTANTS
# ======================================================================


SAMPLE_IMAGE = "photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00.jpg"   # relative to script location


# ======================================================================
# 1) LOGGING SETUP
# ======================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Initialize the PET logger if it has no handlers yet.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        logger.setLevel(level)

        fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(level)

        logger.addHandler(handler)
        logger.debug("Logging initialized (fresh setup).")
    else:
        logger.debug("Logging already initialized; skipping setup.")

    return logger


# ======================================================================
# 4) MAIN ORCHESTRATOR
# ======================================================================

def main(image_path: Optional[str] = None) -> None:
    # --------------------------------------------------------------
    # 0. Init
    # --------------------------------------------------------------
    log = setup_logging()
    log.info("Starting PET prototype pipeline...")

    img, img_meta = image_loader(image_path or SAMPLE_IMAGE)
    H, W = img.shape[:2]

    # --------------------------------------------------------------
    # 1. Detect raw line segments
    # --------------------------------------------------------------
    raw = detect_grid_segments(img)                   # raw["lines"]
    raw_lines = raw["lines"]

    # Generate angle histograms
    # -------------------------
    angle_info = compute_segment_angles(raw)
    hist = compute_angle_histogram_circular_weighted(angle_info, bins=72)
    angle_info = compute_segment_lengths(angle_info)
    hist_weighted = compute_angle_histogram_circular_weighted(angle_info)    
    
    # Analyze angle stats
    # -------------------
    analysis = analyze_two_orientation_families(hist)
    analysis_weighted = analyze_two_orientation_families(hist_weighted)
    fam_kdes = compute_family_kdes(hist_weighted, analysis_weighted)
    print_angle_analysis_console(hist_weighted, analysis_weighted, fam_kdes)
    plot_family_kdes(hist_weighted, analysis_weighted, fam_kdes)

    # Plot histograms
    # ---------------
    plot_angle_histogram_with_kde(hist)
    plot_angle_histogram_with_kde(hist_weighted)

    # Debug image rotation
    # --------------------
    img_fixed = apply_rotation_correction(img, analysis)
    save_image(img_fixed, "rotated.jpg")

    # Split into two rough direction families (unsupervised)
    fam = split_segments_by_angle_circular(raw["lines"], angle_info, analysis)

    raw_x = fam["family1"]
    raw_y = fam["family2"]

    # Debug: mark raw lines (green)
    dbg_raw = mark_segments(img, raw_lines, color=(0, 255, 0))
    save_image(dbg_raw, "debug_raw_segments.jpg")

    # --------------------------------------------------------------
    # 2. Filter + refine the line families
    # --------------------------------------------------------------
    flt = filter_grid_segments(
        raw,
        angle_tol_deg=20
    )
    flt_x = flt["lines_x"]
    flt_y = flt["lines_y"]

    # Debug: filtered families (x=red, y=blue)
    dbg_flt = mark_segment_families(img, flt_x, flt_y)
    save_image(dbg_flt, "debug_filtered_segments.jpg")

    # --------------------------------------------------------------
    # 3. Vanishing point estimation (projective geometry)
    # --------------------------------------------------------------
    vp_info = estimate_vanishing_points(
        flt_x,
        flt_y,
        img_shape=(H, W),
    )

    vp_x = vp_info["vp_x"]
    vp_y = vp_info["vp_y"]

    print("--- Vanishing Point Diagnostics ---")
    print("VP X:", vp_x, " RMS:", vp_info["rms_x"])
    print("VP Y:", vp_y, " RMS:", vp_info["rms_y"])
    print("Orthogonality angle error:", vp_info["angle_orth_error_deg"], "deg")
    print("VP orth error:", vp_info["vp_orth_error_deg"], "deg")
    print("Horizon line coefficients:", vp_info["horizon"])
    print("-----------------------------------")

    refined = refine_principal_point_from_vps(
        vp_info["vp_x"],
        vp_info["vp_y"],
        (H, W),
        radius_frac=0.1,    # 10% search radius
        steps=30            # 30x30 grid
    )
    
    print("Original center:", refined["cx0"], refined["cy0"])
    print("Refined center :", refined["cx_refined"], refined["cy_refined"])
    print("Improved VP orth error:", refined["vp_orth_error_deg"])

    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
