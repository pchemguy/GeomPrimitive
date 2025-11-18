"""
Histogram viewer with full x/y ticks, numeric labels,
and percentile markers.

- Left: grayscale (L) histogram
- Right: R, G, B histograms stacked vertically
- Dashed lines: 5th and 95th percentile for each histogram
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
# Computation
# ------------------------------------------------------------

def compute_channel_data(img):
    arr = np.asarray(img)

    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    L = np.asarray(img.convert("L")).ravel()

    N = len(r)

    def hist_percent(x):
        h, _ = np.histogram(x, bins=256, range=(0, 255))
        return h / N * 100

    hr = hist_percent(r)
    hg = hist_percent(g)
    hb = hist_percent(b)
    hL = hist_percent(L)

    def p05_p95(x):
        return float(np.percentile(x, 5)), float(np.percentile(x, 95))

    return (
        (hr, p05_p95(r)),
        (hg, p05_p95(g)),
        (hb, p05_p95(b)),
        (hL, p05_p95(L)),
    )


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def draw_percentile_lines(ax, p05, p95):
    ax.axvline(p05, color="gray", linestyle="--", linewidth=1)
    ax.axvline(p95, color="gray", linestyle="--", linewidth=1)


def set_axis_ticks(ax, ymax):
    """
    Apply consistent tick formatting to both axes.
    """

    # ----- X-axis -----
    ax.set_xlim(0, 255)
    xticks = np.linspace(0, 255, 9)  # 0,32,...,255
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}" for x in xticks])

    # ----- Y-axis -----
    ax.set_ylim(0, ymax)
    yticks = np.arange(0, 101, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}" for y in yticks])
    ax.set_ylabel("Percent (%)")


# ------------------------------------------------------------
# Main plot function
# ------------------------------------------------------------

def plot_histograms(R, G, B, L, title=None):
    (hr, (r5, r95)) = R
    (hg, (g5, g95)) = G
    (hb, (b5, b95)) = B
    (hL, (L5, L95)) = L

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], hspace=0.45)

    # --------------------------------------------------------
    # Left column: Grayscale
    # --------------------------------------------------------
    axL = fig.add_subplot(gs[:, 0])
    ymax_L = max(hL) * 1.12
    axL.bar(range(256), hL, width=1.0, color="black", alpha=0.82)
    set_axis_ticks(axL, ymax_L)
    draw_percentile_lines(axL, L5, L95)
    axL.set_title("Grayscale (L)")

    # --------------------------------------------------------
    # Right column: R, G, B stacked
    # --------------------------------------------------------
    for ax, h, color, name, (p05, p95) in [
        (fig.add_subplot(gs[0, 1]), hr, "red",   "Red",   (r5, r95)),
        (fig.add_subplot(gs[1, 1]), hg, "green", "Green", (g5, g95)),
        (fig.add_subplot(gs[2, 1]), hb, "blue",  "Blue",  (b5, b95)),
    ]:
        ymax = max(h) * 1.12
        ax.bar(range(256), h, width=1.0, color=color, alpha=0.82)
        set_axis_ticks(ax, ymax)
        draw_percentile_lines(ax, p05, p95)
        ax.set_title(name)

    # Only the bottom histogram gets the x-axis label
    fig.axes[-1].set_xlabel("Intensity")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def show_histograms(path: str):
    img = load_image(path)
    R, G, B, L = compute_channel_data(img)
    plot_histograms(R, G, B, L, title=path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hist_viewer.py <image_path>")
        sys.exit(1)

    show_histograms(sys.argv[1])
