"""
spt_base.py
-----------
"""

from __future__ import annotations

from typing import TypeAlias
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ImageBGR:  TypeAlias = np.typing.NDArray[np.uint8]  # (H, W, 3) BGR order
ImageRGBA: TypeAlias = np.typing.NDArray[np.uint8]  # (H, W, 4) RGBA order


def rgb2rgba(rgba: ImageRGBA) -> ImageBGR:
    """Convert RGBA (Matplotlib) to BGR (OpenCV)."""
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


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
                 dpi: int = 200) -> ImageRGBA:
    """Render an ideal grid + primitives scene via Matplotlib.

    Returns:
        RGBA numpy array (H x W x 4), to be passed into SyntheticPhotoProcessor.
    """
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")

    # Grid (1mm + 10mm thicker lines)

    add_grid(ax, width_mm=100, height_mm=80)

    # Primitives: square, circle, triangle

    ax.add_patch(plt.Rectangle((30, 30), 20, 20, edgecolor="red", fill=False, lw=2))
    ax.add_patch(plt.Circle((70, 40), 10, edgecolor="blue", fill=False, lw=2))
    tri = np.array([[10, 10], [20, 10], [15, 25]])
    ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return rgba


def main():
    rgba: np.ndarray = render_scene()

    plt.imshow(rgba)
    plt.axis("off")
    plt.title("Matplotlib RGBA Display")
    plt.show()


if __name__ == "__main__":
    main()

