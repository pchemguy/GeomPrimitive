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

import cv2
import numpy as np
from PIL import Image, ExifTags


from pet_utils import image_loader, save_image, LOGGER_NAME, SAMPLE_IMAGE
from pet_whitepoint import (
    whitepoint_pipeline, whitepoint_correct, estimate_paper_mask, estimate_whitepoint,
    apply_white_balance, auto_levels
)


# ======================================================================
# MODULE CONSTANTS
# ======================================================================



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
# 3) OTHER PIPELINE STEPS (placeholders)
# ======================================================================

def step_example_grid_analysis() -> None:
    log = logging.getLogger(LOGGER_NAME)
    log.info("Running grid-based geometric analysis...")
    # TODO: implement


# ======================================================================
# 4) MAIN ORCHESTRATOR
# ======================================================================

def main(image_path: Optional[str] = None) -> None:
    log = setup_logging()
    log.info("Starting PET prototype pipeline...")

    img, meta0 = image_loader(image_path)

    img_w, meta_w = whitepoint_pipeline(img)
    log.info(f"Whitepoint stage complete: {meta_w}")


    # Perform corrective processing and write result
    save_path = SAMPLE_IMAGE[:-4] + "whitepoint.jpg"
    
    # 1) White-balance only (gentle)
    mask = estimate_paper_mask(img)
    white = estimate_whitepoint(img, mask)
    balanced = apply_white_balance(img, white)
    
    # 2) Auto-levels for brightness+contrast (Photoshop-style)
    corrected = auto_levels(balanced)
    save_image(corrected, save_path)
    meta = {"white_bgr": white}    
    
    step_example_grid_analysis()

    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
