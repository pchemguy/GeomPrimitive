"""
worker.py
---------

Per-process drawing worker used by orchestration.py.

Responsibilities:
- create and own a Matplotlib figure/axes for this process
- seed RNG deterministically per PID and time
- hold a reusable primitives.line.Line instance
- draw N primitives per job
- collect per-image JSON metadata
- save the image to disk and return (path, json)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Union, Tuple

import matplotlib
matplotlib.use("Agg")  # safe for multiprocessing workers
import matplotlib.pyplot as plt

# add project root so we can import primitives when run as a script
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

from primitives.line import Line  # uses your class-based primitive


PathLike = Union[str, os.PathLike]
DPI = 100  # fixed global DPI


class ThreadWorker:
    """
    Per-process synthetic image worker.

    Each process:
    - has its own figure and axes
    - has its own image-level metadata dict
    """

    def __init__(self, img_size: Tuple[int, int] = (1920, 1080)) -> None:
        self.img_size = img_size
        self.logger = logging.getLogger(f"worker-{os.getpid()}")
        self.fig, self.ax = self._create_canvas()
        self._meta: dict = {}
        self.line = Line()  # reusable primitive
        self._seed_rng()
        self.logger.info("Initialized ThreadWorker")

    # -------------------------------------------------------------------------
    # init helpers
    # -------------------------------------------------------------------------
    def _seed_rng(self) -> None:
        """
        Seed the Line-level RNG so every worker process gets its own stream.
        Note: this controls primitive randomness, not Python's global random.
        """
        pid = os.getpid()
        seed = pid ^ int(time.time())
        Line.reseed(seed)
        # image-level metadata root
        self._meta = {
            "pid": pid,
            "seed": seed,
            "draw_ops": [],
        }
        self.logger.debug(f"Random seed set: {seed}")

    def _create_canvas(self) -> Tuple[plt.Figure, plt.Axes]:
        width_in = self.img_size[0] / DPI
        height_in = self.img_size[1] / DPI
        fig, ax = plt.subplots(figsize=(width_in, height_in), frameon=False)
        ax.set_xlim(0, self.img_size[0])
        ax.set_ylim(0, self.img_size[1])
        ax.invert_yaxis()
        ax.axis("off")
        return fig, ax

    # -------------------------------------------------------------------------
    # per-job lifecycle
    # -------------------------------------------------------------------------
    def plot_reset(self) -> None:
        """
        Reset axes to a blank image while keeping pid/seed.
        Called once per job by orchestration.main_worker.
        """
        self.ax.cla()
        self.ax.set_xlim(0, self.img_size[0])
        self.ax.set_ylim(0, self.img_size[1])
        self.ax.invert_yaxis()
        self.ax.axis("off")

        # keep pid and seed, reset draw_ops
        pid = self._meta.get("pid")
        seed = self._meta.get("seed")
        self._meta = {
            "pid": pid,
            "seed": seed,
            "draw_ops": [],
        }

    # -------------------------------------------------------------------------
    # draw ops
    # -------------------------------------------------------------------------
    def draw_line(self, **kwargs) -> None:
        """
        Draw a single line using the reusable Line primitive.
        Extra kwargs are forwarded to line.make_geometry.
        """
        self.line.make_geometry(self.ax, **kwargs).draw(self.ax)
        self._append_meta("line", self.line.meta)

    def _append_meta(self, shape_type: str, data: dict) -> None:
        # store primitive-level metadata in per-image metadata
        self._meta["draw_ops"].append({shape_type: data})

    # -------------------------------------------------------------------------
    # output
    # -------------------------------------------------------------------------
    def save_image(self, output_path: PathLike) -> Tuple[Path, str]:
        """
        Save current figure as JPG and return (path, json_str).
        Orchestration layer will collect this JSON into batch file.
        """
        out = Path(output_path)
        self.fig.savefig(
            out,
            dpi=DPI,
            format="jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        return out, json.dumps(self._meta, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # teardown
    # -------------------------------------------------------------------------
    def close(self) -> None:
        plt.close(self.fig)
        self.logger.info("Figure closed and resources released.")
