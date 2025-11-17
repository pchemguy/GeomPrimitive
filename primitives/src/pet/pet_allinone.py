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
from pet_exposure import exposure_pipeline_graphpaper


from pet_utils import image_loader, save_image
from pet_pipeline import run_pet_pipeline
from pet_config import PETConfig




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
    log = setup_logging()
    log.info("Starting PET prototype pipeline...")

    cfg = PETConfig(
        preset_name="default_graphpaper",
        debug_enabled=True,
        debug_outdir="debug_run_01"
    )
    

    img, meta0 = image_loader()
    final, meta = run_pet_pipeline(img, cfg)
    log.info(f"Exposure meta: {meta}")

    save_image(final, "output/final_corrected.jpg")

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
