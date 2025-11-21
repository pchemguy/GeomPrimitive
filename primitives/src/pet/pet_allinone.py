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
import matplotlib.pyplot as plt

from pet_utils import image_loader, save_image, LOGGER_NAME

from pet_geom import (
    detect_grid_segments, clamp_segment_length,
    compute_segment_angles, compute_angle_histogram,
    plot_angle_histogram, plot_angle_histogram_with_kde, apply_rotation_correction,
    compute_angle_histogram_circular_weighted, compute_segment_lengths,
    analyze_two_orientation_families, print_angle_analysis_console,
    analyze_grid_periodicity_full, plot_periodicity_analysis,
    reassign_and_rotate_families_by_image_center, draw_famxy_on_image,
    compute_centerline_arrays, draw_centerline_arrays, yc_hist, plot_yc_hist,
    plot_rotated_family_length_histograms, periodicity_detector_1d,
    xc_hist_from_clusters, yc_hist_from_clusters, cluster_gridlines_1d,
    compute_family_kdes, plot_family_kdes,
    filter_grid_segments, estimate_vanishing_points,
    refine_principal_point_from_vps, split_segments_by_angle_circular,
    mark_segments, mark_segment_families, 
    plot_lsd_distributions,
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
    plot_lsd_distributions(raw)

    # Drop extremely short segments.
    # ------------------------------
    flt = clamp_segment_length(raw, min_len=5, max_len=1000)

    # Generate angle histograms
    # -------------------------
    angle_info = compute_segment_angles(flt)
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
    img_rotated = apply_rotation_correction(img, analysis_weighted)
    save_image(img_rotated, "rotated.jpg")

    # Split into two rough direction families (unsupervised)
    fam = split_segments_by_angle_circular(flt["lines"], angle_info, analysis_weighted)

    raw_x = fam["family1"]
    raw_y = fam["family2"]
    # Debug: mark raw lines (green)
    dbg_raw = mark_segments(img, raw_lines, color=(0, 255, 0))
    save_image(dbg_raw, "debug_raw_segments.jpg")


    famxy = reassign_and_rotate_families_by_image_center(fam, analysis_weighted, img)
    img_rotated_lines = draw_famxy_on_image(img_rotated, famxy)

    plot_rotated_family_length_histograms(famxy, bin_size=2)    
    save_image(img_rotated_lines, "rotated_lines.jpg")

    famxcyc = compute_centerline_arrays(famxy)
    xcenters = famxcyc["xcenters"]   # shape (Nx, 3): [xc, yc, length]
    ycenters = famxcyc["ycenters"]   # shape (Ny, 3): [xc, yc, length]

    # -------------------------------------
    # Y-family: vertical gridlines -> X-axis positions
    # -------------------------------------
    xc = ycenters[:, 0]       # X-midpoints
    Ly = ycenters[:, 2]       # segment lengths

    y_clusters = cluster_gridlines_1d(
        positions=xc,
        lengths=Ly,
        max_gap_px=None,
        min_cluster_members=2,
        robust_factor=3.0,
    )

    # histogram from cluster centers
    yc_hist_data = yc_hist_from_clusters(
        y_clusters["centers"],       # FIXED
        bin_size=10,
        gap_size=0,
        offset=0,
        weights=y_clusters["weights"]
    )

    # Plot Y-family histogram
    plot_yc_hist(yc_hist_data, title="Y-family X-Histogram", color="blue")

    period_y = periodicity_detector_1d(
        y_clusters["centers"],
        weights=y_clusters["weights"]
    )
    print("Y-family spacing ->", period_y["best"])
    plot_periodicity_analysis(period_y)

    # -------------------------------------
    # X-family: horizontal gridlines -> Y-axis positions
    # -------------------------------------
    yc = xcenters[:, 1]        # Y-midpoints
    Lx = xcenters[:, 2]        # segment lengths

    x_clusters = cluster_gridlines_1d(
        positions=yc,
        lengths=Lx,
        max_gap_px=None,
        min_cluster_members=2,
        robust_factor=3.0,
    )

    # histogram from cluster centers
    xc_hist_data = xc_hist_from_clusters(
        x_clusters["centers"],       # FIXED
        bin_size=10,
        gap_size=0,
        offset=0,
        weights=x_clusters["weights"]
    )

    # Plot X-family histogram
    plot_yc_hist(xc_hist_data, title="X-family Y-Histogram", color="red")

    period_x = periodicity_detector_1d(
        x_clusters["centers"],
        weights=x_clusters["weights"]
    )    
    print("X-family spacing ->", period_x["best"])

    img_rotated_centerline = draw_centerline_arrays(img_rotated, famxcyc)
    save_image(img_rotated_centerline, "rotated_lines_centers.jpg")

    #yc_hist_data = yc_hist(famxy, bin_size=10, gap_size=90, offset=5)
    
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

    
    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
