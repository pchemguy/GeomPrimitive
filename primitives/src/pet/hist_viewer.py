"""
hist_viewer.py
--------------

Display image histograms:
- Left: grayscale intensity (L mode)
- Right: R, G, B histograms stacked vertically

Usage:
    python hist_viewer.py path/to/image.jpg
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path: str) -> Image.Image:
    """Load an image and return a Pillow Image object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return Image.open(p).convert("RGB")


def compute_histograms(img: Image.Image):
    """
    Return:
        r, g, b: histograms for channels
        l: grayscale (L-mode) histogram
    """
    arr = np.asarray(img).astype(np.uint8)
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    # L = perceptual luminance (Pillow's "L")
    L = np.asarray(img.convert("L")).ravel()

    return r, g, b, L


def plot_histograms(r, g, b, L, title: str = None):
    """Create the 2x2 layout: grayscale on left, R/G/B stacked on right."""
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.4)

    # -------- LEFT: Grayscale histogram --------
    axL = fig.add_subplot(gs[:, 0])
    axL.hist(L, bins=256, color="black", alpha=0.8)
    axL.set_title("Grayscale (L) Histogram")
    axL.set_xlim(0, 255)
    axL.set_xlabel("Intensity")
    axL.set_ylabel("Count")

    # -------- RIGHT: R, G, B stacked --------
    axR = fig.add_subplot(gs[0, 1])
    axG = fig.add_subplot(gs[1, 1])
    axB = fig.add_subplot(gs[2, 1])

    axR.hist(r, bins=256, color="red", alpha=0.8)
    axG.hist(g, bins=256, color="green", alpha=0.8)
    axB.hist(b, bins=256, color="blue", alpha=0.8)

    for ax, name in [(axR, "Red"), (axG, "Green"), (axB, "Blue")]:
        ax.set_xlim(0, 255)
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Channel")
        ax.set_xticks([])

    axB.set_xlabel("Intensity")  # only bottom histogram shows x-axis

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def show_histograms(path: str):
    """High-level wrapper: load -> compute -> plot."""
    img = load_image(path)
    r, g, b, L = compute_histograms(img)
    plot_histograms(r, g, b, L, title=path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hist_viewer.py <image_path>")
        sys.exit(1)

    show_histograms(sys.argv[1])
