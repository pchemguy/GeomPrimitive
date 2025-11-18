"""
pet_pipeline.py
----------------

High-level orchestrator for the PET graph-paper enhancement pipeline.
Uses PETConfig for all tunables and pet_debug for visualization.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Tuple

import numpy as np
import cv2

os.environ["LOKY_EXECUTABLE"] = sys.executable
os.environ["LOKY_WORKER"]     = sys.executable
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_PICKLER"] = "pickle"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pet_config import PETConfig
from pet_utils import save_image
from pet_exposure import (
    detect_white_regions_gmm,
    refine_paper_mask,
    whitebalance_auto_graphpaper,
    auto_levels_masked,
    exposure_detail_enhance,
)
from pet_debug import (
    save_pipeline_debug,
    overlay_mask,
)


LOGGER_NAME = "pet"


# ======================================================================
# PIPELINE ORCHESTRATOR
# ======================================================================

def run_pet_pipeline(
    img: np.ndarray,
    cfg: PETConfig,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full PET graph-paper exposure pipeline using settings from cfg.

    Stages:
        1) GMM-based paper detection
        2) Mask refinement
        3) White balance
        4) Auto-levels (with guardrails)
        5) Optional local detail enhancement
        6) Optional debug visualization

    Args:
        img: BGR uint8 numpy array.
        cfg: PETConfig with all tunables.

    Returns:
        (final_img, metadata_dict)
    """
    log = logging.getLogger(LOGGER_NAME)

    # ==============================================================
    # 1) WHITE REGION DETECTION
    # ==============================================================
    raw_mask, probs, gmm_info = detect_white_regions_gmm(
        img,
        n_components=cfg.gmm_n_components,
        sample_fraction=cfg.gmm_sample_fraction,
        prob_threshold=cfg.gmm_prob_threshold,
        highlight_clip=cfg.gmm_highlight_clip,
        random_state=cfg.gmm_random_state,
    )
    
    # ==============================================================
    # 2) MASK REFINEMENT
    # ==============================================================
    mask = refine_paper_mask(
        raw_mask,
        min_area_frac=cfg.mask_min_area_frac,
        close_ksize=cfg.mask_close_ksize,
    )

    # ==============================================================
    # 3) WHITE BALANCE
    # ==============================================================
    wb_img, wb_info = whitebalance_auto_graphpaper(
        img,
        mask,
        gain_min=cfg.wb_gain_min,
        gain_max=cfg.wb_gain_max,
    )

    # ==============================================================
    # 4) AUTO LEVELS
    # ==============================================================
    lvl_img, lvl_info = auto_levels_masked(
        wb_img,
        mask=mask,
        low_clip=cfg.levels_low_clip,
        high_clip=cfg.levels_high_clip,
        lo_max=cfg.levels_lo_max,
        min_range=cfg.levels_min_range,
    )

    # ==============================================================
    # 5) DETAIL ENHANCE
    # ==============================================================
    if cfg.enable_detail:
        final = exposure_detail_enhance(
            lvl_img,
            sigma_s=cfg.detail_sigma_s,
            sigma_r=cfg.detail_sigma_r,
        )
    else:
        final = lvl_img

    # ==============================================================
    # DEBUG OUTPUT
    # ==============================================================
    if cfg.debug_enabled:
        mask_overlay_img = overlay_mask(img, mask)

        stages = {
            "original": img,
            "mask_raw": raw_mask,
            "mask_refined": mask,
            "mask_overlay": mask_overlay_img,
            "wb": wb_img,
            "levels": lvl_img,
            "detail": final,
        }

        save_pipeline_debug(
            stages,
            out_dir=cfg.debug_outdir,
            composite_name=cfg.debug_composite_name,
            histogram=cfg.debug_histograms,
        )

    # ==============================================================
    # METADATA
    # ==============================================================
    meta = {
        "gmm": gmm_info,
        "wb": wb_info,
        "levels": lvl_info,
        "used_detail": cfg.enable_detail,
        "config_preset": cfg.preset_name,
    }

    return final, meta
