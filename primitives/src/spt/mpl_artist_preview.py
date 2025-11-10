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


def preview_mpl_artist(artist, title: str | None = None):
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

    # --- Determine overall bounding box ---
    bboxes: list[Bbox] = []

    for a in artists:
        try:
            # Some artists have get_extents(), others get_datalim() or get_window_extent()
            if hasattr(a, "get_extents"):
                bb = a.get_extents()
            elif hasattr(a, "get_datalim"):
                bb = a.get_datalim(ax.transData)
            elif hasattr(a, "get_window_extent"):
                bb = a.get_window_extent(renderer=fig.canvas.get_renderer())
            else:
                # fallback to manual vertex inspection
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

    # Combine into one global bounding box
    if bboxes:
        combined = Bbox.union(bboxes)
        x0, y0, x1, y1 = combined.x0, combined.y0, combined.x1, combined.y1
        dx, dy = x1 - x0, y1 - y0
        pad_x, pad_y = dx * 0.05, dy * 0.05
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

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
