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
    detect_grid_segments, normalize_missing_metas, clamp_segment_length,
    clamp_segment_bbox, plot_lsd_distributions, xy_scatter_from_centers,
)

from pet_geom import (
    compute_segment_angles, compute_angle_histogram,
    plot_angle_histogram, plot_angle_histogram_with_kde,
    compute_angle_histogram_circular_weighted, compute_segment_lengths,
    analyze_two_orientation_families, print_angle_analysis_console,
    compute_family_kdes, plot_family_kdes,
    split_segments_by_angle_circular,
)

from pet_lsd_width_analysis import (
    analyze_lsd_widths, 
    cluster_line_thickness,
    merge_lsd_dicts,
    split_widths_hist,
    print_thickness_summary,
)

from pet_grid_auto_crop import detect_grid_area_density
from pet_grid_node_detector import find_grid_nodes
from pet_kde_interactive import plot_kde_interactive


# ======================================================================
# MODULE CONSTANTS
# ======================================================================

# BUG: -----------------------------------------------------------------
#      Note there is a bug in the current pipeline most likely related to angle normalization.
#      If this image version, rotated by 90 deg, is used, the pipeline fails.
#      With original image it appears to be working.
#      The core difference is probably due to algorithm primarily focusing initially
#      on subgrid family ("vertical" vs. "horizontal") with more detected segments.
#      The orientation is selected as smallest magnitude angle that aligns the dominant
#      family with either X or Y axis.
# SAMPLE_IMAGE = "photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00_90.jpg"

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

    # Load image
    # ----------
    img, img_meta = image_loader(image_path or SAMPLE_IMAGE)              
    H, W = img.shape[:2]

    # Detect grid bounding box (presently not used further)
    # Saves debug image with detected bounding box.
    # This routine is based on detecting grid edges, and applying any
    # of Photoshop Auto- Contrast/Tone/Color/Curves may improve quality.
    # -------------------------------------------------------------------
    bbox = detect_grid_area_density(img)

    # Detect grid nodes (presently not used further)
    # Saves debug image with detected nodes.
    # Note: Application of any of Photoshop Auto- Contrast/Tone/Color/Curves
    #       (following normalization of local contrast) may considerably
    #       increase node detection rate. Visiually, these option result
    #       only in limited noise level increase, though more nodes are labled
    #       on major grid edges. Whether actual covearage quality is improved
    #       needs to be assessed.
    # ----------------------------------------------
    nodes = find_grid_nodes(img, output_dir="output")

    # --------------------------------------------------------------
    # 1. Detect raw line segments
    # --------------------------------------------------------------
    # LSD OpenCV segment detector
    # Note applying any of Photoshop Auto- Contrast/Tone/Color/Curves
    # may improve quality.
    # ---------------------------
    raw = detect_grid_segments(img)
    raw_lines = raw["lines"]
    raw_centers = raw["centers"]
    xy_scatter_from_centers(raw_centers, title="Raw LSD Segments Centers", size_scale=6)

    # Histogram of detected LSD segments: width, precision, and NFA
    # Note, if opencv-contrib version with extras included in the build 
    # is not installed, only the most basic segment detction is performed,
    # with only width information populated, but not precision/nfa.
    # For millimeter graph paper expect bimodal distribution:
    # thinner minor and thicker major lines
    # -------------------------------------------------------------------
    lsd_dist_bins = 50
    plot_lsd_distributions(raw, bins=lsd_dist_bins)

    # Replace missig metas with dummies.
    #-----------------------------------
    raw = normalize_missing_metas(raw)

    # Drop extremely short segments and excessively thick (top 5%)
    # -------------------------------------------------------------
    flt = clamp_segment_length(raw, min_len=5, max_len=1000, width_percentile=95)
    flt_lines = flt["lines"]
    flt_centers = flt["centers"]
    xy_scatter_from_centers(flt_centers, bbox=bbox, title="Pre-filtered LSD Segments Centers", size_scale=6)
    plot_lsd_distributions(flt, bins=lsd_dist_bins)

    flt_bbox = clamp_segment_bbox(flt, bbox)
    flt_bbox_centers = flt_bbox["centers"]
    xy_scatter_from_centers(flt_bbox_centers, bbox=bbox, title="Pre-filtered LSD Segments Centers", size_scale=6)
    
    # Statistical analysis of segment width distribution - bimodal distribution.
    # --------------------------------------------------------------------------
    width_analysis = analyze_lsd_widths(flt_bbox, max_components=3, plot=True)
    
    # Hard split by thickness (minor / major / outliers)
    # Split segments based on bimodal distribution. Place high and low outliers
    # in separate groups.
    # -------------------------------------------------------------------------
    thickness_groups = cluster_line_thickness(flt_bbox, analysis=width_analysis, robust_sigma=3.0)
    lsd_minor       = thickness_groups["minor"]
    lsd_major       = thickness_groups["major"]
    lsd_outliers_lo = thickness_groups["outliers_lo"]
    lsd_outliers_hi = thickness_groups["outliers_hi"]
    print_thickness_summary(thickness_groups)

    # Debug display of LSD segment width distribution split.
    # ------------------------------------------------------
    split_widths_hist(thickness_groups)
    
    # Merge high-range outliers group into the "major" segment group
    # (based on preliminary inspection)
    # --------------------------------------------------------------
    lsd_major = merge_lsd_dicts(lsd_major, lsd_outliers_hi)
    flt = lsd_major

    # Compute segment angles and length and generate angle histogram
    # --------------------------------------------------------------
    angle_info = compute_segment_angles(flt)
    angle_info = compute_segment_lengths(angle_info)
    hist = compute_angle_histogram_circular_weighted(angle_info, bins=72)    
    
    # Plot angle histograms
    # ---------------------
    plot_angle_histogram_with_kde(hist)

    # Analyze angle histogram distribution and split into to families (X/Y)
    # ---------------------------------------------------------------------
    analysis = analyze_two_orientation_families(hist)
    fam_kdes = compute_family_kdes(hist, analysis)
    print_angle_analysis_console(hist, analysis, fam_kdes)
    plot_family_kdes(hist, analysis, fam_kdes)

    # Split into two direction families
    # ---------------------------------
    fam = split_segments_by_angle_circular(flt, angle_info, analysis)

    landscape_centers = fam["family1"]["centers"]
    portait_centers = fam["family2"]["centers"]
    
    plot_kde_interactive(landscape_centers, bw=2)
    plot_kde_interactive(portait_centers, bw=2)


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
