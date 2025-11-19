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
from PIL import Image, ExifTags


from pet_utils import image_loader, save_image, LOGGER_NAME
from pet_whitepoint import (
    whitepoint_pipeline, whitepoint_correct, estimate_paper_mask, estimate_whitepoint,
    apply_white_balance, auto_levels
)
from pet_exposure import exposure_pipeline_graphpaper

from pet_geom import (
    detect_grid_segments, filter_grid_segments, estimate_vanishing_points,
    refine_principal_point_from_vps, compute_rectifying_homography,
    warp_with_homography, rectify_grid_affine, separate_line_families_kmeans,
    mark_segments, mark_segment_families, debug_draw_raw_segments,
    rectify_grid_projective, warp_homography_optimal_size,
    debug_draw_families, print_family_angles,
)



from pet_utils import image_loader, save_image
from pet_pipeline import run_pet_pipeline
from pet_config import PETConfig


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

    cfg = PETConfig(
        preset_name="default_graphpaper",
        debug_enabled=True,
        debug_outdir="debug_run_01"
    )

    img, img_meta = image_loader(image_path or SAMPLE_IMAGE)
    H, W = img.shape[:2]

    # --------------------------------------------------------------
    # 1. Detect raw line segments
    # --------------------------------------------------------------
    raw = detect_grid_segments(img)                   # raw["lines"]
    raw_lines = raw["lines"]

    # Split into two rough direction families (unsupervised)
    fam = separate_line_families_kmeans(raw_lines)
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

    # --------------------------------------------------------------
    # 4. Affine + metric rectification (full projective correction)
    # --------------------------------------------------------------
    rectified, H_projective = rectify_grid_projective(
        img,
        vp_x,
        vp_y,
        raw_x,     # original X-lines (not filtered)
        raw_y,     # original Y-lines
        scale_limit=6.0,
    )

    save_image(rectified, "rectified_affine.jpg")
    log.info("Affine+metric rectification completed.")

    # --------------------------------------------------------------
    # Optionally: return results
    # --------------------------------------------------------------
    return rectified, {
        "vp": vp_info,
        "H": H_projective,
        "raw": raw_lines,
        "filtered_x": flt_x,
        "filtered_y": flt_y,
    }

   # debug_draw_raw_segments(img, raw["lines"], "debug_raw_lsd.jpg")
   # families = separate_line_families_kmeans(raw["lines"])
   # print_family_angles(families)
   # lines_x = families["family1"]
   # lines_y = families["family2"]    
   # debug_draw_families(img, lines_x, lines_y)
    
   # rectified, H_lin = rectify_grid_affine(img, lines_x, lines_y)
   # save_image(rectified, "debug_rectified_affine.jpg")
   # log.info("Affine+metric rectification completed.")

    
   # refined = refine_principal_point_from_vps(
   #     vp_info["vp_x"],
   #     vp_info["vp_y"],
   #     (H, W),
   #     radius_frac=0.1,    # 10% search radius
   #     steps=30            # 30x30 grid
   # )
   # 
   # print("Original center:", refined["cx0"], refined["cy0"])
   # print("Refined center :", refined["cx_refined"], refined["cy_refined"])
   # print("Improved VP orth error:", refined["vp_orth_error_deg"])


    
   #    
   # Hinfo = compute_rectifying_homography(
   #     vp_info["vp_x"],
   #     vp_info["vp_y"],
   #     refined["cx_refined"],
   #     refined["cy_refined"]
   # )
   # 
   # rectified = warp_with_homography(img, Hinfo["H"])
   # 
   # save_image(rectified, "rectified.jpg")   
   #   
   # img, meta0 = image_loader(image_path)
   #
   # img_w, meta_w = whitepoint_pipeline(img)
   # log.info(f"Whitepoint stage complete: {meta_w}")
   # 
   # 
   # # Perform corrective processing and write result
   # save_path = SAMPLE_IMAGE[:-4] + "_whitepoint.jpg"
   # 
   # # 1) White-balance only (gentle)
   # mask = estimate_paper_mask(img)
   # white = estimate_whitepoint(img, mask)
   # balanced = apply_white_balance(img, white)
   # 
   # # 2) Auto-levels for brightness+contrast (Photoshop-style)
   # corrected = auto_levels(balanced)
   # save_image(corrected, save_path)
   # meta = {"white_bgr": white}    
   # 
   # save_path = SAMPLE_IMAGE[:-4] + "_exposure.jpg"
   # corrected, meta_exp = exposure_pipeline_graphpaper(img)
   #
   # save_image(corrected, save_path)
   # log.info(f"Exposure meta: {meta_exp}")
   # log.info("PET pipeline completed.")
   #
   #
    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
