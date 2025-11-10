"""
mpl_utils.py
-----------
"""

from __future__ import annotations

__all__ = [
    "bgr_from_rgba", "rgb_from_bgr",
    "show_RGBx_grid", "render_scene",
    "ImageBGR", "ImageRGB", "ImageRGBA", "ImageRGBx",
    "PAPER_COLORS", "DEFAULT_LINEWIDTHS",
]

import os
import sys
import time
import random
import math
from typing import TypeAlias, Sequence, Union
import numpy as np
from numpy.typing import NDArray
from skimage import util, exposure

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ImageBGR:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) BGR order
ImageRGB:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) RGB order
ImageRGBA: TypeAlias = NDArray[np.uint8]  # (H, W, 4) RGBA order
ImageRGBx: TypeAlias = Union[ImageRGB, ImageRGBA] # Either RGB or RGBA

DEFAULT_LINEWIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
PAPER_COLORS = [
    "none", "white", "cornsilk", "ivory", "oldlace", "floralwhite", "whitesmoke"
] # X11/CSS4

# Plot background color settings
#
# Global settings:
#   plt.rcParams["figure.facecolor"] = "white" # Canvas
#   plt.rcParams["axes.facecolor"] = "white"   # Plot area
#
# Object:
#   fig.patch.set_facecolor("white")           # Canvas
#   ax.set_facecolor("white")                  # Plot area
#
# Object - Transparent:                        # Canvas   
#   fig.patch.set_alpha(0.0)                   # Plot area
#   ax.set_facecolor("none")

# Plot padding trimming (numbers are fractions of canvas size)
#
# Global:
#   fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
#   plt.tight_layout() # Reduces margins to reasonable minimum.
#       IMPORTANT: must be executed AFTER ax.axis("off")
#
# Object:
#   ax.set_position([0, 0, 1, 1])
#   ax.margins(x=0, y=0)


def bgr_from_rgba(rgba: ImageRGBA) -> ImageBGR:
    """Convert RGBA (Matplotlib) to BGR (OpenCV)."""
    return rgba[..., :3][..., ::-1]


def rgb_from_bgr(bgr: ImageBGR) -> ImageRGB:
    """Convert BGR (OpenCV) to RGB (Matplotlib)."""
    return bgr[..., ::-1]


def show_RGBx_grid(images: dict[str, ImageRGBx], title_style: dict = None, 
                   n_columns: int = None, figsize_scale: float = 5) -> None:
    """
    Display multiple images in an automatically balanced rectangular grid.

    Layout rule:
        cols = ceil(sqrt(N))
        rows = ceil(N / cols)
    (Keeps the layout close to square, with longer side horizontal.)

    Args:
        images: Dictionary of <Title>:<Image>; Image - NumPy array (RGB or RGBA).
        title_style: Optional dict for Matplotlib title styling.
        figsize_scale: Multiplier for overall figure size.
    """
    images_n = len(images)
    if images_n == 0:
        raise ValueError("No images to display.")

    # --- Compute balanced grid ---
    
    cols = n_columns or math.ceil(math.sqrt(images_n))
    rows = math.ceil(images_n / cols)

    fig_w = cols * figsize_scale
    fig_h = rows * figsize_scale * 0.9
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)  # flatten axes array

    # --- Title style defaults ---
    style = dict(fontsize=14, fontweight="bold", color="green")
    if title_style:
        style.update(title_style)

    # --- Draw each image ---
    for (title, img), ax in zip(images.items(), axes):
        ax.imshow(img)
        ax.set_title(title, **style)
        ax.axis("off")

    for i in range(images_n, rows * cols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def add_grid(ax, width_mm=100, height_mm=80) -> None:
    """Add fine (1 mm) and major (10 mm) gridlines using LineCollection."""
    # Fine grid every 1 mm
    x_fine = np.arange(0, width_mm + 1, 1)
    y_fine = np.arange(0, height_mm + 1, 1)
    lines_fine = (
        [((x, 0), (x, height_mm)) for x in x_fine] +
        [((0, y), (width_mm, y)) for y in y_fine]
    )
    lc_fine = LineCollection(lines_fine, colors="gray", linewidths=0.2, alpha=0.5)
    ax.add_collection(lc_fine)

    # Major grid every 10 mm
    x_major = np.arange(0, width_mm + 1, 10)
    y_major = np.arange(0, height_mm + 1, 10)
    lines_major = (
        [((x, 0), (x, height_mm)) for x in x_major] +
        [((0, y), (width_mm, y)) for y in y_major]
    )
    lc_major = LineCollection(lines_major, colors="gray", linewidths=0.6, alpha=0.7)
    ax.add_collection(lc_major)


def render_scene(width_mm: float = 100, 
                 height_mm: float = 80,
                 dpi: int = 200,
                 canvas_bg_idx: int = None,
                 plot_bg_idx: int = None) -> ImageRGBA:
    """Render an ideal grid + primitives scene via Matplotlib.

    Returns:
        RGBA numpy array (H x W x 4), to be passed into SyntheticPhotoProcessor.
    """
    rng = random.Random(os.getpid() ^ int(time.time()))

    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)

    bg_n = len(PAPER_COLORS)

    if not canvas_bg_idx is None:
        if (not isinstance(canvas_bg_idx, int)
            or canvas_bg_idx >= bg_n or canvas_bg_idx < 0 ):
            canvas_bg_idx = rng.randrange(len(PAPER_COLORS))
        if canvas_bg_idx == 0:
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_facecolor(PAPER_COLORS[canvas_bg_idx])

    if not plot_bg_idx is None:
        if (not isinstance(plot_bg_idx, int)
            or plot_bg_idx >= bg_n or plot_bg_idx < 0):
            plot_bg_idx = rng.randrange(len(PAPER_COLORS))
        ax.set_facecolor(PAPER_COLORS[plot_bg_idx])

    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    #ax.set_position([0, 0, 1, 1])
    #ax.margins(x=0, y=0)
    
    # Grid (1mm + 10mm thicker lines)

    add_grid(ax, width_mm=100, height_mm=80)

    # Primitives: square, circle, triangle

    ax.add_patch(plt.Rectangle((30, 30), 20, 20, edgecolor="red", fill=False, lw=2))
    ax.add_patch(plt.Rectangle((10, 50), 20, 20, edgecolor="cyan", fill=False, lw=2))
    ax.add_patch(plt.Circle((70, 40), 10, edgecolor="blue", fill=False, lw=2))
    tri = np.array([[10, 10], [20, 10], [15, 25]])
    ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return rgba


def main():
    # ----------------------------------------------------------------------
    rng = random.Random(os.getpid() ^ int(time.time()))

    rgba: ImageRGBA = render_scene()
    bgr:  ImageBGR  = bgr_from_rgba(rgba)
    rgb:  ImageRGB  = rgb_from_bgr(bgr)
    
    bgr_f = util.img_as_float(bgr)
    bgr_via_f = util.img_as_ubyte(exposure.rescale_intensity(bgr_f))
    rgb_via_f = rgb_from_bgr(bgr_via_f)

    demos = {
        "Matplotlib RGBA":               rgba,
        "Roundtrip: RGBA -> BGR -> RGB": rgb,
        "Roundtrip: SKIMAGE AS UBYTE": rgb_via_f,
    }
    
    canvas_bg_idx = rng.randrange(len(PAPER_COLORS))
    plot_bg_idx = rng.randrange(len(PAPER_COLORS))
    demos[
        f"\nRandom: "
        f"Canvas: '{PAPER_COLORS[canvas_bg_idx]}'. Plot - '{PAPER_COLORS[plot_bg_idx]}'"
    ] = render_scene(canvas_bg_idx=canvas_bg_idx, plot_bg_idx=plot_bg_idx)

    for color_idx in range(len(PAPER_COLORS)):
        demos[
            f"Canvas: '{PAPER_COLORS[color_idx]}'. Plot - '{PAPER_COLORS[color_idx]}'"
        ] = render_scene(canvas_bg_idx=color_idx, plot_bg_idx=color_idx)

    show_RGBx_grid(demos)

if __name__ == "__main__":
    main()

