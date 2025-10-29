import os
import json
import random
import time
from pathlib import Path
from typing import Union, Tuple
import logging

import numpy as np
import matplotlib
# Use a non-interactive backend for multiprocessing workers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import splprep, splev


PathLike = Union[str, os.PathLike]
DPI = 100  # fixed global DPI


class SyntheticImageWorker:
    """
    Per-process image generator exposing primitive drawing operations.
    No flow orchestration is handled here.
    """

    def __init__(self, img_size: Tuple[int, int] = (1920, 1080)):
        self.img_size = img_size
        self.logger = logging.getLogger(f"worker-{os.getpid()}")
        self.fig, self.ax = self._create_canvas()
        self._meta: str = "{}"  # JSON string for image-level metadata
        self._seed_rng()
        self.logger.info("Initialized SyntheticImageWorker")

    def _seed_rng(self):
        pid = os.getpid()
        seed = pid ^ int(time.time())
        random.seed(seed)
        np.random.seed((pid * 2654435761) % 2**32)
        self._meta = json.dumps({"pid": pid, "seed": seed, "draw_ops": []})
        self.logger.debug(f"Random seed set: {seed}")

    def _create_canvas(self):
        width_in = self.img_size[0] / DPI
        height_in = self.img_size[1] / DPI
        fig, ax = plt.subplots(figsize=(width_in, height_in), frameon=False)
        return fig, ax

    def plot_reset(self):
        # Preserve existing metadata (pid, seed, etc.), but reset draw_ops
        m = json.loads(self._meta)
        m["draw_ops"] = []
        self._meta = json.dumps(m)
        self.ax.cla()
        self.ax.set_xlim(0, self.img_size[0])
        self.ax.set_ylim(0, self.img_size[1])
        self.ax.invert_yaxis()
        self.ax.axis("off")

    # ---- Drawing primitives -------------------------------------------------
    def draw_spline(self):
        pts = np.array([
            [random.randint(0, 400), random.randint(0, 400)],
            [random.randint(400, 800), random.randint(500, 900)],
            [random.randint(900, 1800), random.randint(100, 500)],
        ])
        m, k = len(pts), min(3, len(pts) - 1)
        if m > k:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
            x_new, y_new = splev(np.linspace(0, 1, 100), tck)
        else:
            x_new, y_new = pts[:, 0], pts[:, 1]
        self.ax.plot(x_new, y_new, "b-")
        self._append_meta("spline", {"points": pts.tolist()})

    def draw_ellipse(self):
        cx, cy = random.randint(800, 1200), random.randint(600, 900)
        w, h = random.randint(100, 500), random.randint(100, 500)
        ang = random.uniform(0, 90)
        oval = Ellipse(
            xy=(cx, cy),
            width=w,
            height=h,
            angle=ang,
            edgecolor="r",
            facecolor="none",
        )
        self.ax.add_patch(oval)
        self._append_meta("ellipse", {"center": [cx, cy], "w": w, "h": h, "angle": ang})

    def draw_line(self):
        x1, x2 = 10, 1800
        y1, y2 = random.randint(10, 1000), 1000
        lw = random.uniform(1, 3)
        self.ax.plot([x1, x2], [y1, y2], "g--", linewidth=lw)
        self._append_meta("line", {"x": [x1, x2], "y": [y1, y2], "lw": lw})

    # ---- Helpers ------------------------------------------------------------
    def _append_meta(self, shape_type: str, data: dict):
        m = json.loads(self._meta)
        m["draw_ops"].append({shape_type: data})
        self._meta = json.dumps(m)

    def save_image(self, output_path: PathLike) -> Tuple[Path, str]:
        out = Path(output_path)
        self.fig.savefig(out, dpi=DPI, format="jpg", bbox_inches="tight", pad_inches=0)
        return out, self._meta

    def close(self):
        plt.close(self.fig)
        self.logger.info("Figure closed and resources released.")
