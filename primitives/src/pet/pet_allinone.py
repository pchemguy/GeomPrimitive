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


# ======================================================================
# MODULE CONSTANTS
# ======================================================================

LOGGER_NAME = "pet"
SAMPLE_IMAGE = "photo_2025-11-17_23-50-02.jpg"   # relative to script location


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
# 2) IMAGE LOADING ROUTINE
# ======================================================================

def _resolve_sample_path() -> str:
    """
    Resolve SAMPLE_IMAGE relative to the script location (__file__).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, SAMPLE_IMAGE)
    return full_path


def _extract_exif_metadata_pillow(path: str) -> dict:
    """
    Extract EXIF metadata using Pillow, if present.
    Returns a dict of key -> value.
    """
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if not exif:
                return {}

            # Convert numeric EXIF tags to readable names
            label_map = {v: k for k, v in ExifTags.TAGS.items()}

            readable = {}
            for key, val in exif.items():
                name = ExifTags.TAGS.get(key, key)
                readable[name] = val

            return readable
    except Exception:
        return {}


def image_loader(path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """
    Load an image file and extract metadata.

    Args:
        path: Optional explicit path. If None, uses SAMPLE_IMAGE.

    Returns:
        (image_np, metadata_dict)
    """
    log = logging.getLogger(LOGGER_NAME)

    if path is None:
        path = _resolve_sample_path()
        log.info(f"No input path provided - using sample: '{path}'")

    if not os.path.exists(path):
        log.error(f"Image not found: {path}")
        raise FileNotFoundError(path)

    # Load using OpenCV (BGR)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        log.error(f"Failed to load image via OpenCV: {path}")
        raise RuntimeError(f"Could not load: {path}")

    log.info(f"Loaded image: {path}")

    # Basic metadata
    h, w = img.shape[:2]
    meta = {
        "path": path,
        "width": w,
        "height": h,
        "channels": img.shape[2] if img.ndim == 3 else 1,
        "dtype": str(img.dtype),
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
    }

    # Extract EXIF with Pillow
    exif = _extract_exif_metadata_pillow(path)
    if exif:
        meta["exif"] = exif
        log.info(f"EXIF tags detected ({len(exif)} entries).")
    else:
        log.info("No EXIF metadata found.")

    # Log summary
    log.info(
        f"Image metadata: {w}x{h}, "
        f"{meta['channels']}ch, dtype={meta['dtype']}, "
        f"min={meta['min']:.1f}, max={meta['max']:.1f}, mean={meta['mean']:.1f}"
    )

    return img, meta


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

    img, meta = image_loader(image_path)
    step_example_grid_analysis()

    log.info("PET pipeline completed.")


# ======================================================================
# 5) ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
