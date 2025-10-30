"""
worker.py - Synthetic image generator worker.

Each worker owns its RNG seed, Matplotlib figure, and image-level metadata.
It uses primitives.Line (and future primitives) for geometry generation.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Union, Tuple

import matplotlib
matplotlib.use("Agg")  # safe for multiprocessing
import matplotlib.pyplot as plt

sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from primitives.line import Line

PathLike = Union[str, os.PathLike]
DPI = 100


class ThreadWorker:
    line: Line = Line()

    """Encapsulates the drawing state for one process."""

    def __init__(self, img_size: Tuple[int, int] = (1920, 1080)):
        self.img_size = img_size
        self.logger = logging.getLogger(f"worker-{os.getpid()}")
        self.fig, self.ax = self._create_canvas()
        self._meta: dict = {}
        self.line = Line()  # reusable primitive instance
        self._init_rng()
        self.logger.debug("ThreadWorker initialized.")

    # --------------------------------------------------------------------------
    # Initialization / teardown
    # --------------------------------------------------------------------------
    def _init_rng(self) -> None:
        """Seed RNG uniquely per process."""
        pid = os.getpid()
        seed = pid ^ int(time.time())
        Line.reseed(seed)
        self._meta = {"pid": pid, "seed": seed, "draw_ops": []}
        self.logger.debug(f"RNG seeded with {seed}")

    def _create_canvas(self):
        """Create a clean Matplotlib canvas."""
        w, h = (self.img_size[0] / DPI, self.img_size[1] / DPI)
        fig, ax = plt.subplots(figsize=(w, h), frameon=False)
        ax.set_xlim(0, self.img_size[0])
        ax.set_ylim(0, self.img_size[1])
        ax.invert_yaxis()
        ax.axis("off")
        return fig, ax

    def plot_reset(self) -> None:
        """Clear axes and reset draw_ops."""
        self.ax.cla()
        self.ax.set_xlim(0, self.img_size[0])
        self.ax.set_ylim(0, self.img_size[1])
        self.ax.invert_yaxis()
        self.ax.axis("off")
        self._meta["draw_ops"] = []

    def close(self) -> None:
        """Release figure resources."""
        plt.close(self.fig)
        self.logger.debug("Canvas closed.")

    # --------------------------------------------------------------------------
    # Drawing operations
    # --------------------------------------------------------------------------
    def draw_line(self, **kwargs) -> None:
        """Generate and draw a single line primitive."""
        self.line.make_geometry(self.ax, **kwargs).draw(self.ax)
        self._append_meta("Line", self.line.meta)

    def _append_meta(self, kind: str, data: dict) -> None:
        self._meta["draw_ops"].append({kind: data})

    # --------------------------------------------------------------------------
    # I/O
    # --------------------------------------------------------------------------
    def save_image(self, output_path: PathLike) -> Tuple[Path, str]:
        """Save current figure and return JSON metadata string."""
        out = Path(output_path)
        self.fig.savefig(out, dpi=DPI, format="jpg", bbox_inches="tight", pad_inches=0)
        return out, json.dumps(self._meta, indent=2, ensure_ascii=False)
