"""
hist_viewer.py

Correct version:
- Histogram shown in PERCENT (%), not density
- Main Y-axis = percent (%), scaled to 1.05 * max histogram
- Secondary Y-axis = cumulative (%), 0-100
- Savitzky-Golay smoothed cumulative
- p05/p95 percentile markers
- Outline histogram (no bars)
- Stacked R/G/B + grayscale L
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# Load image
# ============================================================

def load_image(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return Image.open(p).convert("RGB")


# ============================================================
# Histogram + percentiles
# ============================================================

def compute_channel_data(img):
    arr = np.asarray(img)

    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    L = np.asarray(img.convert("L")).ravel()

    total = len(r)

    def hist_percent(x):
        """
        Return histogram where each bin is PERCENT of total pixels
        so that sum(h) = 100.
        """
        h, _ = np.histogram(x, bins=256, range=(0, 255))
        return h / total * 100

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


# ============================================================
# Helpers
# ============================================================

def draw_percentile_lines(ax, p05, p95):
    ax.axvline(p05, color="gray", linestyle="--", linewidth=1)
    ax.axvline(p95, color="gray", linestyle="--", linewidth=1)


def set_percent_axis(ax, ymax):
    """
    Main Y-axis - percent scale.
    """

    # --- X-axis ticks ---
    ax.set_xlim(0, 255)
    xt = np.linspace(0, 255, 9)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(x)}" for x in xt])

    # --- Y-axis ticks (percent scale) ---
    ax.set_ylim(0, ymax)

    # smart step
    if ymax < 5:
        step = 0.5
    elif ymax < 15:
        step = 1
    elif ymax < 40:
        step = 2
    else:
        step = 5

    yticks = np.arange(0, ymax + step, step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax.set_ylabel("Percent (%)")


def cumulative_savgol(h):
    """
    Compute cumulative percent curve (0-100%) and smooth it.
    """
    cum = np.cumsum(h)
    cum = cum / cum[-1] * 100  # now 0->100%
    sm = savgol_filter(cum, window_length=21, polyorder=3)
    return np.clip(sm, 0, 100)


# ============================================================
# Plotting
# ============================================================

def plot_histograms(R, G, B, L, title=None):
    (hr, (r5, r95)) = R
    (hg, (g5, g95)) = G
    (hb, (b5, b95)) = B
    (hL, (L5, L95)) = L

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], hspace=0.45)

    # ========================================================
    # Grayscale L
    # ========================================================
    axL = fig.add_subplot(gs[:, 0])
    ymax_L = max(hL) * 1.05

    # Outline histogram (percent)
    axL.plot(range(256), hL, color="black", linewidth=1.8, alpha=0.9)

    draw_percentile_lines(axL, L5, L95)
    set_percent_axis(axL, ymax_L)
    axL.set_title("Grayscale (L)")

    # Cumulative on secondary axis
    axL_cum = axL.twinx()
    cumL = cumulative_savgol(hL)
    axL_cum.plot(range(256), cumL, color="gray", linewidth=1.4, alpha=0.85)
    axL_cum.set_ylim(0, 100)
    axL_cum.set_yticks(np.arange(0, 101, 20))
    axL_cum.set_ylabel("Cumulative (%)")

    # ========================================================
    # RGB channels
    # ========================================================
    for ax, h, color, name, (p05, p95) in [
        (fig.add_subplot(gs[0, 1]), hr, "red",   "Red",   (r5, r95)),
        (fig.add_subplot(gs[1, 1]), hg, "green", "Green", (g5, g95)),
        (fig.add_subplot(gs[2, 1]), hb, "blue",  "Blue",  (b5, b95)),
    ]:
        ymax = max(h) * 1.05

        # outline histogram (%)
        ax.plot(range(256), h, color=color, linewidth=1.8, alpha=0.9)

        draw_percentile_lines(ax, p05, p95)
        set_percent_axis(ax, ymax)
        ax.set_title(name)

        # cumulative curve on secondary axis
        cum = cumulative_savgol(h)
        axc = ax.twinx()
        axc.plot(range(256), cum, color="black", linewidth=1.4, alpha=0.85)
        axc.set_ylim(0, 100)
        axc.set_yticks(np.arange(0, 101, 20))
        axc.set_ylabel("Cumulative (%)")

    # bottom x-label
    fig.axes[-1].set_xlabel("Intensity")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# ============================================================
# Entry point
# ============================================================

def show_histograms(path: str):
    img = load_image(path)
    R, G, B, L = compute_channel_data(img)
    plot_histograms(R, G, B, L, title=path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hist_viewer.py <image_path>")
        sys.exit(1)
    show_histograms(sys.argv[1])
