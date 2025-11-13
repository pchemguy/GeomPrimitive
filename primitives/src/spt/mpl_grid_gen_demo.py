"""
mpl_grid_gen_demo.py
--------------------

Demonstration of jittered oblique grids using presets.

Includes visual gallery of multiple jitter presets from mpl_grid_utils.py.
Produces an 8-panel (2x4) comparison of different grid distortion styles.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from utils.rng import RNGBackend, RNG, get_rng

from mpl_grid_gen import (
    generate_grid_collections,
    debug_dump_grid_info,
    GridJitterConfig,
)


# ---------------------------------------------------------------------------
# Helper function for drawing a single grid into an Axes
# ---------------------------------------------------------------------------
def draw_grid(ax, preset_name, bbox=(-10, -10, 10, 10)):

    # Create jitter config from preset
    jitter = GridJitterConfig.preset(preset_name)

    # Grid geometry
    obliquity_deg = 60.0
    rotation_deg = 20.0

    x_major_step = 3.0
    x_minor_step = 1.0
    y_major_step = 3.0
    y_minor_step = 1.0

    xM, xm, yM, ym = generate_grid_collections(
        bbox=bbox,
        obliquity_deg=obliquity_deg,
        rotation_deg=rotation_deg,
        x_major_step=x_major_step,
        x_minor_step=x_minor_step,
        y_major_step=y_major_step,
        y_minor_step=y_minor_step,
        jitter=jitter,
    )

    # Styling
    for lc in (xM, yM):
        lc.set_linewidth(1.4)
        lc.set_color((0.15, 0.15, 0.15, 0.8))
    for lc in (xm, ym):
        lc.set_linewidth(0.7)
        lc.set_color((0.15, 0.15, 0.15, 0.4))

    # Add to axes
    ax.add_collection(xM)
    ax.add_collection(xm)
    ax.add_collection(yM)
    ax.add_collection(ym)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(preset_name.replace("_", " ").title(), fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def demo0():

    debug_dump_grid_info(
        bbox=(-10,-10,10,10),
        obliquity_deg=90,
        rotation_deg=0,
        x_major_step=5,
        x_minor_step=2.5,
        y_major_step=5,
        y_minor_step=2.5,
        jitter = GridJitterConfig()
    )    
        
    
    presets = [
        "handwriting_synthetic",
        "engineering_paper",
        "architectural_drift",
        "printlike_subtle",
        "sketchy",
        "technical",
        "messy",
        "blueprint",
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=4, figsize=(10, 16),
        constrained_layout=True
    )

    for ax, preset in zip(axes.ravel(), presets):
        draw_grid(ax, preset)

    fig.suptitle("Grid Jitter Preset Gallery (2x4)", fontsize=16, y=0.995)
    plt.show()



def demo1():
    # Plot area (world coordinates)
    bbox = (-10.0, -10.0, 10.0, 10.0)

    # Oblique grid: 60deg between axes, rotated 20deg CCW
    obliquity_deg = 60.0
    rotation_deg = 20.0

    # Steps: fairly fine minor grid, more spaced major grid
    x_major = 3.0
    x_minor = 1.0
    y_major = 3.0
    y_minor = 1.0

    # Use the hand-drawn preset
    jitter = GridJitterConfig.hand_drawn()

    x_major_lc, x_minor_lc, y_major_lc, y_minor_lc = generate_grid_collections(
        bbox=bbox,
        obliquity_deg=obliquity_deg,
        rotation_deg=rotation_deg,
        x_major_step=x_major,
        x_minor_step=x_minor,
        y_major_step=y_major,
        y_minor_step=y_minor,
        jitter=jitter,
    )

    # Styling
    for lc in (x_major_lc, y_major_lc):
        lc.set_linewidth(1.5)
        lc.set_color((0.1, 0.1, 0.1, 0.8))  # dark grey, almost black

    for lc in (x_minor_lc, y_minor_lc):
        lc.set_linewidth(0.8)
        lc.set_color((0.1, 0.1, 0.1, 0.4))  # lighter grey

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(x_major_lc)
    ax.add_collection(x_minor_lc)
    ax.add_collection(y_major_lc)
    ax.add_collection(y_minor_lc)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Heavily Jittered Oblique Grid (Hand-drawn Preset)")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def demo2():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    bbox = ((-10.0, -10.0), (10.0, 10.0))
    
    x_major_lc, x_minor_lc, y_major_lc, y_minor_lc = generate_grid_collections(
        bbox=bbox,
        obliquity_deg=90.0,      # orthogonal
        rotation_deg=15.0,      # rotate grid 15deg CCW
        x_major_step=2.0,
        x_minor_step=0.5,
        y_major_step=2.0,
        y_minor_step=0.5,
        jitter=GridJitterConfig(),  # enable jitter with defaults
    )
    
    # Style the collections however you like
    x_major_lc.set_linewidth(1.2)
    x_major_lc.set_alpha(0.7)
    
    x_minor_lc.set_linewidth(0.6)
    x_minor_lc.set_alpha(0.4)
    
    y_major_lc.set_linewidth(1.2)
    y_major_lc.set_alpha(0.7)
    
    y_minor_lc.set_linewidth(0.6)
    y_minor_lc.set_alpha(0.4)
    
    ax.add_collection(x_major_lc)
    ax.add_collection(x_minor_lc)
    ax.add_collection(y_major_lc)
    ax.add_collection(y_minor_lc)
    
    ax.set_title("Default Jittered Oblique Grid")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    
    plt.show()



def demo3():
    pass
    """
    lc.set(**{
        "linewidths": 1.2,
        "colors": "tab:blue",
        "alpha": 0.6,
        "linestyles": "--",
        "zorder": 0,
    })
    """

def main():
    demo0()
    demo1()
    demo2()


if __name__ == "__main__":
    main()
