"""
mpl_artist_preview.py
---------------------
Render any Matplotlib artist with automatic bounding-box sizing (5% margin),
hidden axes, tight layout, and immediate display.
"""

from __future__ import annotations

__all__ = ["preview_mpl_artist",]


import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, PathCollection, PatchCollection
from matplotlib.patches import PathPatch, Patch, Circle, Rectangle
import numpy as np
from matplotlib.transforms import Bbox


def ax_autofit(ax, margin: float = 0.05) -> None:
    """
    Automatically adjust axis limits to tightly fit all artists currently attached to `ax`.

    Inspects the geometric extents of all patches, lines, collections, and generic
    artists in the given Axes, computes a combined bounding box, and expands the
    view limits by a fixed fractional margin (default 5%).

    Args:
        ax: Matplotlib Axes instance whose artists are to be fitted.
        margin: Fractional padding around the computed bounding box (default = 0.05).

    Behavior:
        - Collects artists from `ax.patches`, `ax.lines`, `ax.collections`, and `ax.artists`.
        - Falls back to direct vertex inspection if no extent methods exist.
        - Ignores zero-area objects and degenerate extents.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    bboxes: list[Bbox] = []

    # Collect all artists already added to the axes
    all_artists = list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.artists)

    for a in all_artists:
        try:
            if hasattr(a, "get_extents"):
                bb = a.get_extents()
            elif hasattr(a, "get_datalim"):
                bb = a.get_datalim(ax.transData)
            elif hasattr(a, "get_window_extent"):
                bb = a.get_window_extent(renderer=renderer)
            else:
                # Manual vertex inspection fallback
                if hasattr(a, "get_paths"):
                    verts = np.vstack([p.vertices for p in a.get_paths()])
                elif hasattr(a, "get_data"):
                    x, y = a.get_data()
                    verts = np.column_stack([x, y])
                elif hasattr(a, "vertices"):
                    verts = a.vertices
                else:
                    continue
                bb = Bbox.from_extents(
                    np.min(verts[:, 0]), np.min(verts[:, 1]),
                    np.max(verts[:, 0]), np.max(verts[:, 1]),
                )
        except Exception:
            continue

        if bb.width > 0 and bb.height > 0:
            bboxes.append(bb)

    if not bboxes:
        # Fallback default
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return

    # --- Combine bounding boxes and apply margin ---
    combined = Bbox.union(bboxes)
    x0, y0, x1, y1 = combined.x0, combined.y0, combined.x1, combined.y1
    dx, dy = x1 - x0, y1 - y0
    pad_x, pad_y = dx * margin, dy * margin

    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal", adjustable="box")


def preview_mpl_artist(artist, title: str = None):
    """
    Display a Matplotlib artist with automatic axis limits (5% margin),
    hidden axes, and tight layout. Calls plt.show() automatically.

    Args:
        artist: Any of Path, Line2D, LineCollection, PathCollection,
                Patch, or PatchCollection, or a list/tuple of them.
        title:  Optional title for display.
    """
    artists = [artist] if not isinstance(artist, (list, tuple)) else list(artist)
    fig, ax = plt.subplots()

    # Add all artists to axes
    for a in artists:
        if isinstance(a, mplPath):
            ax.add_patch(PathPatch(a, facecolor="none", edgecolor="black", lw=1.5))
        elif isinstance(a, (Line2D, LineCollection, PathCollection, Patch, PatchCollection)):
            ax.add_artist(a)
        else:
            raise TypeError(f"Unsupported artist type: {type(a).__name__}")

    ax_autofit(ax)

    # Final layout
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0)

    plt.show()


def main():
    patches = [Circle((0, 0), 1), Rectangle((-1, -0.5), 2, 1)]
    pc = PatchCollection(patches, facecolor='lightgray', edgecolor='black')

    preview_mpl_artist(pc, title="Auto Preview Example")


if __name__ == "__main__":
    main()
