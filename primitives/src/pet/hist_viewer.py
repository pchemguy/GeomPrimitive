"""
Histogram viewer with percentile markers and full tick labels.

- Left: grayscale (L) histogram
- Right: stacked R, G, B histograms
- All histograms show:
    - x-axis ticks and numeric labels
    - y-axis ticks (percent)
    - dashed percentile lines at 5th and 95th percentiles
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_image(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return Image.open(p).convert("RGB")


# ------------------------------------------------------------
# Histogram + percentiles
# ------------------------------------------------------------

def compute_channel_data(img):
    arr = np.asarray(img)

    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    L = np.asarray(img.convert("L")).ravel()

    total = len(r)

    def hist(x):
        h, _ = np.histogram(x, bins=256, range=(0, 255))
        return h / total * 100

    hr = hist(r)
    hg = hist(g)
    hb = hist(b)
    hL = hist(L)

    def p(x):
        return np.percentile(x, 5), np.percentile(x, 95)

    r5, r95 = p(r)
    g5, g95 = p(g)
    b5, b95 = p(b)
    L5, L95 = p(L)

    return (hr, (r5, r95)), (hg, (g5, g95)), (hb, (b5, b95)), (hL, (L5, L95))


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def draw_percentile_lines(ax, p05, p95):
    ax.axvline(p05, color="gray", linestyle="--", linewidth=1)
    ax.axvline(p95, color="gray", linestyle="--", linewidth=1)


def format_axes(ax):
    """Apply standard axis formatting for all histograms."""
    ax.set_xlim(0, 255)
    ax.set_xticks(np.linspace(0, 255, 9))  # 0,32,...,255
    ax.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 255, 9)])

    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Percent (%)")


def plot_histograms(R, G, B, L, title=None):
    (hr, (r5, r95)) = R
    (hg, (g5, g95)) = G
    (hb, (b5, b95)) = B
    (hL, (L5, L95)) = L

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.4)

    # ---------------- Left: Grayscale ----------------
    axL = fig.add_subplot(gs[:, 0])
    axL.bar(range(256), hL, width=1.0, color="black", alpha=0.8)
    format_axes(axL)
    axL.set_ylim(0, max(hL) * 1.1)
    axL.set_title("Grayscale (L)")
    draw_percentile_lines(axL, L5, L95)

    # ---------------- Right: RGB ----------------
    axR = fig.add_subplot(gs[0, 1])
    axG = fig.add_subplot(gs[1, 1])
    axB = fig.add_subplot(gs[2, 1])

    for ax, h, color, title, p05, p95 in [
        (axR, hr, "red",   "Red",   r5, r95),
        (axG, hg, "green", "Green", g5, g95),
        (axB, hb, "blue",  "Blue",  b5, b95),
    ]:
        ax.bar(range(256), h, width=1.0, color=color, alpha=0.8)
        format_axes(ax)
        ax.set_ylim(0, max(h) * 1.1)
        ax.set_title(title)
        draw_percentile_lines(ax, p05, p95)

    axB.set_xlabel("Intensity")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def show_histograms(path):
    img = load_image(path)
    R, G, B, L = compute_channel_data(img)
    plot_histograms(R, G, B, L, title=path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hist_viewer.py <image_path>")
        sys.exit(1)

    show_histograms(sys.argv[1])
