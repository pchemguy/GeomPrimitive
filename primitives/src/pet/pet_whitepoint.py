"""
pet_whitepoint.py
-----------------

White region detection, white-point estimation, and illumination correction
for PET (Paper Enhancement & Transformation) pipeline.

Routines:
    estimate_paper_mask
    estimate_whitepoint
    apply_white_balance
    correct_illumination
    whitepoint_pipeline
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ======================================================================
# 1) WHITE REGION DETECTION
# ======================================================================

def estimate_paper_mask(
    img: np.ndarray,
    blur_radius: int = 41,
    percentile: float = 97.5,
) -> np.ndarray:
    """
    Estimate the binary mask of the "white paper" background.
    Assumes the paper is the brightest large-area region.

    Steps:
      - Convert to grayscale
      - Heavy Gaussian blur to suppress the grid
      - Threshold using a top percentile

    Args:
        img: BGR uint8 image.
        blur_radius: Gaussian kernel size (must be odd).
        percentile: Percentile of blurred-brightness to use as threshold.

    Returns:
        mask: uint8 binary mask (0/255).
    """
    log = logging.getLogger("pet")

    if img.dtype != np.uint8:
        raise TypeError("estimate_paper_mask expects uint8 BGR input.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

    t = np.percentile(blur, percentile)
    mask = (blur >= t).astype(np.uint8) * 255

    log.info(
        f"Paper mask: blur_radius={blur_radius}, percentile={percentile}, "
        f"threshold={t:.1f}, white_frac={mask.mean()/255:.3f}"
    )

    return mask


# ======================================================================
# 2) WHITE-POINT ESTIMATION
# ======================================================================

def estimate_whitepoint(
    img: np.ndarray,
    mask: np.ndarray,
    clip_frac: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Estimate paper whitepoint from masked pixels.

    Args:
        img: BGR uint8 image.
        mask: Binary mask of paper region.
        clip_frac: Fraction of pixels to clip on both extremes to avoid outliers.

    Returns:
        (B_white, G_white, R_white) floating-point estimates in [0,255].
    """
    log = logging.getLogger("pet")

    if img.dtype != np.uint8:
        raise TypeError("estimate_whitepoint expects uint8 BGR input.")

    idx = mask > 0
    if idx.sum() < 100:
        log.warning("Paper mask too small. Whitepoint may be inaccurate.")

    pixels = img[idx]

    # Trim extremes to robustify
    N = len(pixels)
    k = int(N * clip_frac)
    if k > 0:
        pixels_sorted = np.sort(pixels, axis=0)
        pixels = pixels_sorted[k:-k]

    white = pixels.mean(axis=0).astype(np.float32)
    B, G, R = white

    log.info(
        f"Estimated whitepoint (BGR): "
        f"{B:.1f}, {G:.1f}, {R:.1f}"
    )

    return float(B), float(G), float(R)


# ======================================================================
# 3) WHITE BALANCE APPLICATION
# ======================================================================

def apply_white_balance(
    img: np.ndarray,
    white_bgr: Tuple[float, float, float],
    target: float = 255.0,
) -> np.ndarray:
    """
    Apply simple channel scaling based on estimated whitepoint.

    Args:
        img: BGR uint8 image.
        white_bgr: Estimated whitepoint (floats).
        target: Desired white value (usually 255).

    Returns:
        balanced: BGR uint8.
    """
    log = logging.getLogger("pet")
    B, G, R = white_bgr

    # Compute gains
    gainB = target / max(B, 1e-6)
    gainG = target / max(G, 1e-6)
    gainR = target / max(R, 1e-6)
    gains = np.array([gainB, gainG, gainR])

    log.info(f"White-balance gains: B={gainB:.3f}, G={gainG:.3f}, R={gainR:.3f}")

    balanced = img.astype(np.float32) * gains
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


# ======================================================================
# 4) ILLUMINATION / SHADING CORRECTION
# ======================================================================

def correct_illumination(
    img: np.ndarray,
    sigma: float = 101.0,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Retinex-like coarse shading removal.

    Steps:
        - Convert to float
        - Estimate illumination via strong Gaussian blur
        - Divide image by illumination
        - Normalize back to uint8

    Args:
        img: BGR uint8 image.
        sigma: Gaussian sigma for illumination estimation.
        epsilon: Avoid division by zero.

    Returns:
        corrected: BGR uint8.
    """
    log = logging.getLogger("pet")

    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    illum = gaussian_filter(gray, sigma=sigma)
    illum_expanded = np.stack([illum, illum, illum], axis=-1)

    corrected = img_f / (illum_expanded + epsilon)
    corrected = corrected / corrected.max()  # normalize
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    log.info(f"Illumination corrected (sigma={sigma}).")

    return corrected


# ======================================================================
# 5) FULL WHITEPOINT PIPELINE
# ======================================================================

def whitepoint_pipeline(
    img: np.ndarray,
    do_illumination: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Full whitepoint -> white balance -> optional illumination correction pipeline.

    Returns:
        corrected_image, metadata_dict
    """
    log = logging.getLogger("pet")
    log.info("Whitepoint pipeline started.")

    mask = estimate_paper_mask(img)
    white = estimate_whitepoint(img, mask)
    balanced = apply_white_balance(img, white)

    if do_illumination:
        corrected = correct_illumination(balanced)
    else:
        corrected = balanced

    meta = {
        "white_bgr": white,
        "illumination_corrected": do_illumination,
    }

    log.info("Whitepoint pipeline complete.")
    return corrected, meta


# ======================================================================
# 6) HIGH-LEVEL WHITEPOINT CORRECTION WRAPPER
# ======================================================================

def _resolve_output_path(path: str) -> str:
    """
    Resolve path for saving corrected output.
    - If absolute, return as-is.
    - If relative, interpret relative to this module's directory (__file__).
    """
    if os.path.isabs(path):
        return path

    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, path)


def whitepoint_correct(
    img: np.ndarray,
    save_path: Optional[str] = None,
    do_illumination: bool = True,
) -> np.ndarray:
    """
    Apply whitepoint pipeline and optionally save result.

    Args:
        img: Loaded image (BGR uint8).
        save_path: Optional output path. Absolute or relative to script directory.
        do_illumination: Whether to run illumination correction step.

    Returns:
        corrected_img: BGR uint8 array.
    """
    log = logging.getLogger("pet")
    log.info("Whitepoint correction routine started.")

    corrected, meta = whitepoint_pipeline(img, do_illumination=do_illumination)
    log.info(f"Whitepoint correction complete. Meta: {meta}")

    if save_path:
        full = _resolve_output_path(save_path)
        out_dir = os.path.dirname(full)

        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        ok = cv2.imwrite(full, corrected)
        if ok:
            log.info(f"Corrected image saved to: {full}")
        else:
            log.error(f"Failed to save corrected image to: {full}")

    return corrected


# ======================================================================
# AUTO BRIGHTNESS / AUTO CONTRAST (Photoshop-like)
# ======================================================================

def auto_levels(
    img: np.ndarray,
    low_clip: float = 0.005,
    high_clip: float = 0.995
) -> np.ndarray:
    """
    Photoshop-like Auto Levels (Auto Brightness/Contrast).

    Steps:
        - Compute grayscale
        - Determine input black/white points via percentiles
        - Linearly stretch all channels to [0,255] accordingly

    Args:
        img: BGR uint8 image.
        low_clip: Lower percentile (0.005 = 0.5%).
        high_clip: Upper percentile (0.995 = 99.5%).

    Returns:
        Corrected BGR uint8 image.
    """
    log = logging.getLogger("pet")

    if img.dtype != np.uint8:
        raise TypeError("auto_levels expects uint8 input.")

    # Convert to grayscale for histogram analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lo = np.percentile(gray, low_clip * 100)
    hi = np.percentile(gray, high_clip * 100)

    log.info(f"Auto-levels: low={lo:.1f}, high={hi:.1f}")

    # Avoid division by zero
    if hi - lo < 1e-6:
        log.warning("Auto-levels degenerate range; returning original image.")
        return img.copy()

    # Vectorized linear stretch
    img_f = img.astype(np.float32)
    img_norm = (img_f - lo) * (255.0 / (hi - lo))
    img_out = np.clip(img_norm, 0, 255).astype(np.uint8)

    return img_out
