"""
pet_visualize.py
----------------

Quick matplotlib visualization utilities for interactive debugging of
the PET graph-paper enhancement pipeline.

These helpers are intentionally lightweight and do not write files.
They display images (converted from BGR to RGB) and masks inline.

Typical usage in interactive session:

    from pet_visualize import show_images, show_pipeline
    show_images([img, wb, levels], ["raw", "wb", "levels"])
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


# ======================================================================
# BASIC HELPERS
# ======================================================================

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to RGB for matplotlib display."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ======================================================================
# 1) SHOW MULTIPLE IMAGES IN GRID
# ======================================================================

def show_images(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 3,
    figsize: Tuple[int, int] = (14, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Show a list of BGR/gray images as a matplotlib grid.
    Titles are optional.
    """

    if titles is None:
        titles = [f"img {i}" for i in range(len(images))]

    assert len(images) == len(titles), "images and titles length mismatch"

    rows = int(np.ceil(len(images) / cols))

    plt.figure(figsize=figsize)

    for i, (im, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if im.ndim == 2:
            plt.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            plt.imshow(_bgr_to_rgb(im), vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================================
# 2) MASK OVERLAY VISUALIZATION
# ======================================================================

def show_mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3,
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """
    Display a mask overlay on BGR image using matplotlib.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask > 0

    overlay = img.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)[None, None, :]

    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color_arr * alpha
    )

    plt.figure(figsize=figsize)
    plt.imshow(_bgr_to_rgb(overlay.astype(np.uint8)))
    plt.title("Mask Overlay")
    plt.axis("off")
    plt.show()


# ======================================================================
# 3) HISTOGRAM DIAGNOSTICS
# ======================================================================

def show_hist(
    img: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Display grayscale + per-channel histograms.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb = _bgr_to_rgb(img)

    plt.figure(figsize=figsize)
    plt.hist(gray.flatten(), bins=256, alpha=0.5, label="gray")
    plt.hist(rgb[..., 0].flatten(), bins=256, alpha=0.5, label="R")
    plt.hist(rgb[..., 1].flatten(), bins=256, alpha=0.5, label="G")
    plt.hist(rgb[..., 2].flatten(), bins=256, alpha=0.5, label="B")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ======================================================================
# 4) LAB PLANE VISUALIZATION
# ======================================================================

def show_lab_planes(
    img: np.ndarray,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Display L*, a*, b* planes for quick color evaluation.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(L, cmap="gray")
    plt.title("L*")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(a, cmap="gray")
    plt.title("a*")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap="gray")
    plt.title("b*")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================================
# 5) ONE-CALL PIPELINE VISUAL DEBUG
# ======================================================================

def show_pipeline(
    stages: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (14, 10),
    cols: int = 3,
) -> None:
    """
    Quick interactive visualization of pipeline stages.

    Example:
        show_pipeline({
            "orig": img,
            "mask_raw": raw_mask,
            "mask_refined": mask,
            "wb": wb_img,
            "levels": lvl_img,
            "detail": final_img,
        })
    """
    images = []
    titles = []
    for name, obj in stages.items():
        if obj is None:
            continue
        if obj.ndim == 2:      # mask
            rgb = cv2.cvtColor(obj, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        images.append(rgb)
        titles.append(name)

    show_images(
        images=images,
        titles=titles,
        cols=cols,
        figsize=figsize,
    )
