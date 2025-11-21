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
    reassign_and_rotate_families_by_image_center, draw_famxy_on_image,
    draw_centerline_arrays,
    plot_rotated_family_length_histograms, 
    compute_family_kdes, plot_family_kdes,
    filter_grid_segments,
    refine_principal_point_from_vps, split_segments_by_angle_circular,
    mark_segments, mark_segment_families, mark_segments_w, mark_segment_families_w,
    plot_lsd_distributions,
)

from pet_lsd_width_analysis import (
    analyze_lsd_widths, 
    cluster_line_thickness,
    merge_lsd_dicts,
    split_widths_hist,
)

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
    lsd_dist_bins = 50
    plot_lsd_distributions(raw, bins=lsd_dist_bins)

    # Drop extremely short segments.
    # ------------------------------
    flt = clamp_segment_length(raw, min_len=5, max_len=1000, width_percentile=95)
    flt_lines = flt["lines"]
    plot_lsd_distributions(flt, bins=lsd_dist_bins)

    # Diagnostics (optional)
    width_analysis = analyze_lsd_widths(flt, max_components=3, plot=True)
    
    # Hard split by thickness (minor / major / outliers)
    thickness_groups = cluster_line_thickness(flt, analysis=width_analysis, robust_sigma=3.0,)
    
    lsd_minor       = thickness_groups["minor"]
    lsd_major       = thickness_groups["major"]
    lsd_outliers_lo = thickness_groups["outliers_lo"]
    lsd_outliers_hi = thickness_groups["outliers_hi"]
    split_widths_hist(thickness_groups)

    # lsd_outliers_hi actually belong to major grid
    lsd_major = merge_lsd_dicts(lsd_major, lsd_outliers_hi)

    dbg = mark_segments_w(img, lsd_minor)
    save_image(dbg, "debug_flt_minor.jpg")
    dbg = mark_segments_w(img, lsd_major)
    save_image(dbg, "debug_flt_major.jpg")
    dbg = mark_segments_w(img, lsd_outliers_lo)
    save_image(dbg, "debug_flt_outliers_lo.jpg")
    dbg = mark_segments_w(img, lsd_outliers_hi)
    save_image(dbg, "debug_flt_outliers_hi.jpg")

    flt = lsd_major
    
    # Debug: mark flt lines (green)
    # -----------------------------
    dbg_flt = mark_segments(img, flt_lines, color=(0, 255, 0))
    save_image(dbg_flt, "debug_flt_segments.jpg")

    dbg_flt_w = mark_segments_w(img, flt)
    save_image(dbg_flt_w, "debug_flt_segments_w.jpg")

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
    fam = split_segments_by_angle_circular(flt, angle_info, analysis_weighted)

    raw_x = fam["family1"]["lines"]
    raw_y = fam["family2"]["lines"]

    famxy = reassign_and_rotate_families_by_image_center(fam, analysis_weighted, img)
    img_rotated_lines = draw_famxy_on_image(img_rotated, famxy)

    plot_rotated_family_length_histograms(famxy, bin_size=2)    
    save_image(img_rotated_lines, "rotated_lines.jpg")

    xcenters = np.column_stack((famxy["xfam"]["centers"], famxy["xfam"]["lengths"]))   # shape (Nx, 3): [xc, yc, length]
    ycenters = np.column_stack((famxy["yfam"]["centers"], famxy["yfam"]["lengths"]))   # shape (Ny, 3): [xc, yc, length]
    famxcyc = {"xcenters": xcenters, "ycenters": ycenters}

    img_rotated_centerline = draw_centerline_arrays(img_rotated, famxcyc)
    save_image(img_rotated_centerline, "rotated_lines_centers.jpg")

    #yc_hist_data = yc_hist(famxy, bin_size=10, gap_size=90, offset=5)
    
    # --------------------------------------------------------------
    # 2. Filter + refine the line families
    # --------------------------------------------------------------
    flt_ang = filter_grid_segments(
        raw,
        angle_tol_deg=20
    )
    flt_x = flt_ang["lines_x"]
    flt_y = flt_ang["lines_y"]

    # Debug: filtered families (x=red, y=blue)
    dbg_flt_ang = mark_segment_families(img, flt_x, flt_y)
    save_image(dbg_flt_ang, "debug_split_segments_filter_angle.jpg")

    
    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
