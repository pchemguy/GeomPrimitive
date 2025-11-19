"""
pet_debug.py
------------

Debugging utilities for visualizing intermediate states of the PET
graph-paper exposure pipeline.

Produces:
    - Side-by-side composite of each processing stage
    - Mask overlays
    - Histogram diagnostics
    - File-based saving for each intermediate stage

All helpers work with OpenCV BGR uint8 images.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Optional, Dict, List

LOGGER_NAME = "pet"


# ======================================================================
# 1) UTILITY: RESOLVE OUTPUT PATHS
# ======================================================================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ======================================================================
# 2) MASK OVERLAY
# ======================================================================

def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay a binary mask onto an image with translucent color.
    mask should be 0/255.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask_bool = (mask > 0).astype(np.uint8)
    overlay = img.copy()

    color_arr = np.zeros_like(img, dtype=np.uint8)
    color_arr[:, :, 0] = color[0]
    color_arr[:, :, 1] = color[1]
    color_arr[:, :, 2] = color[2]

    blended = img.copy()
    blended = cv2.addWeighted(blended, 1.0, color_arr, alpha, 0)

    # Only replace where mask is true
    out = img.copy()
    out[mask_bool == 1] = blended[mask_bool == 1]
    return out


# ======================================================================
# 3) IMAGE GRID COMPOSITOR
# ======================================================================

def grid_composite(
    images: List[np.ndarray],
    labels: List[str],
    cols: int = 3,
    scale: float = 0.4,
    pad: int = 15,
    font_scale: float = 0.6,
) -> Optional[np.ndarray]:
    """
    Create a side-by-side composite grid with labels.
    Returns a BGR image suitable for saving via cv2.imwrite.

    images: list of BGR uint8 images
    labels: list of strings matching images length
    """
    assert len(images) == len(labels), "images and labels length mismatch"
    if len(images) == 0:
        return None

    # Resize all to first image size * scale
    H0, W0 = images[0].shape[:2]
    H = int(H0 * scale)
    W = int(W0 * scale)
    resized = [cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR) for im in images]

    # Add labels on top
    labeled = []
    for im, text in zip(resized, labels):
        canvas = im.copy()
        cv2.putText(
            canvas,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        labeled.append(canvas)

    rows = int(np.ceil(len(images) / cols))
    cols = min(cols, len(images))

    # Build grid
    grid_rows = []
    for r in range(rows):
        start = r * cols
        end = min((r + 1) * cols, len(images))
        row_imgs = labeled[start:end]
        if len(row_imgs) < cols:
            # pad with black
            for _ in range(cols - len(row_imgs)):
                row_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row_imgs))

    out = np.vstack(grid_rows)

    # Padding
    pad_img = np.full((out.shape[0] + pad * 2, out.shape[1] + pad * 2, 3),
                      0, dtype=np.uint8)
    pad_img[pad:-pad, pad:-pad] = out
    return pad_img


# ======================================================================
# 4) HISTOGRAM DIAGNOSTICS
# ======================================================================

def save_histogram(
    img: np.ndarray,
    out_path: str,
    title: str = "",
) -> None:
    """
    Save histogram (gray and per-channel) using matplotlib.
    """
    log = logging.getLogger(LOGGER_NAME)
    _ensure_dir(out_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(6, 4))
    plt.hist(gray.flatten(), bins=256, range=(0, 255), alpha=0.5, label="gray")
    plt.hist(img_rgb[..., 0].flatten(), bins=256, range=(0, 255), alpha=0.5, label="R")
    plt.hist(img_rgb[..., 1].flatten(), bins=256, range=(0, 255), alpha=0.5, label="G")
    plt.hist(img_rgb[..., 2].flatten(), bins=256, range=(0, 255), alpha=0.5, label="B")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    log.info(f"Saved histogram: {out_path}")


# ======================================================================
# 5) MAIN DEBUGGING HOOK
# ======================================================================

def save_pipeline_debug(
    stages: Dict[str, np.ndarray],
    out_dir: str,
    composite_name: str = "pipeline_debug.jpg",
    histogram: bool = True,
) -> str:
    """
    Save a visual debugging report for the pipeline.

    Args:
        stages:
            Ordered dict-like mapping:
                "original": img0,
                "mask_raw": mask,
                "mask_refined": mask2,
                "wb": wb_img,
                "levels": lvl_img,
                "detail": detail_img
            Any stage may be None; it will be skipped.

        out_dir:
            Directory to store outputs.

        composite_name:
            Name of the composite image file.

        histogram:
            Whether to generate histograms for each stage.

    Returns:
        Full path to composite image.
    """
    log = logging.getLogger(LOGGER_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Save individual stages
    image_list = []
    label_list = []

    for name, obj in stages.items():
        if obj is None:
            continue

        if obj.ndim == 2:  # mask
            bgr = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
        else:
            bgr = obj

        out_path = os.path.join(out_dir, f"{name}.jpg")
        cv2.imwrite(out_path, bgr)
        log.info(f"Saved stage: {out_path}")

        image_list.append(bgr)
        label_list.append(name)

        if histogram and obj.ndim == 3:
            h_path = os.path.join(out_dir, f"{name}_hist.png")
            save_histogram(bgr, h_path, title=name)

    # Build composite
    comp = grid_composite(image_list, label_list, cols=3, scale=0.45)
    comp_path = os.path.join(out_dir, composite_name)
    cv2.imwrite(comp_path, comp)
    log.info(f"Saved pipeline composite: {comp_path}")

    return comp_path
