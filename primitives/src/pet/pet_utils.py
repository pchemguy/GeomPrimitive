"""
pet_utils.py
------------
"""

from __future__ import annotations

import os
import logging
import sys
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
    log = logging.getLogger(LOGGER_NAME)
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if not exif:
                return {}

            readable = {}
            for key, val in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                readable[name] = val

            return readable
    except Exception as e:
        log.debug(f"EXIF extraction failed for {path}: {e}")
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
# IMAGE SAVE HELPER
# ======================================================================

def _resolve_output_path(path: str) -> str:
    """
    Resolve a file path:
    - absolute paths returned unchanged
    - relative paths interpreted relative to this module's directory
    """
    if os.path.isabs(path):
        return path

    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, path)


def save_image(img: np.ndarray, path: str) -> str:
    """
    Save an image to a JPEG file.

    Args:
        img: BGR uint8 numpy array.
        path: Output path (absolute or relative to this module).

    Returns:
        Resolved absolute path as string.
    """
    log = logging.getLogger(LOGGER_NAME)

    full_path = _resolve_output_path(path)
    directory = os.path.dirname(full_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        log.debug(f"Created directory: {directory}")

    ok = cv2.imwrite(full_path, img)

    if not ok:
        log.error(f"Failed to save image to: {full_path}")
        raise IOError(f"Could not write: {full_path}")

    log.info(f"Image saved: {full_path}")
    return full_path
