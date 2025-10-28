import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import splprep, splev


PathLike = Union[str, os.PathLike]
_worker = None  # Per-process global worker instance


class SyntheticImageWorker:
    """Encapsulates Matplotlib figure, axes, and drawing logic per worker."""

    def __init__(self, img_size: Tuple[int, int] = (1920, 1080), dpi: int = 100):
        self.img_size = img_size
        self.dpi = dpi
        self.fig, self.ax = self._create_canvas()

        # Per-process RNG
        pid = os.getpid()
        seed = pid ^ int(time.time())
        random.seed(seed)
        np.random.seed((pid * 2654435761) % 2**32)
        print(f"[PID {pid}] Initialized with seed {seed}")

    def _create_canvas(self):
        width_in, height_in = (self.img_size[0] / 100, self.img_size[1] / 100)
        fig, ax = plt.subplots(figsize=(width_in, height_in), frameon=False)
        ax.set_xlim(0, self.img_size[0])
        ax.set_ylim(0, self.img_size[1])
        ax.invert_yaxis()
        ax.axis("off")
        return fig, ax

    def _draw_spline(self):
        pts = np.array([
            [random.randint(0, 400), random.randint(0, 400)],
            [random.randint(400, 800), random.randint(500, 900)],
            [random.randint(900, 1800), random.randint(100, 500)],
        ])
        m = len(pts)
        k = min(3, m - 1)
        if m > k:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
            x_new, y_new = splev(np.linspace(0, 1, 100), tck)
        else:
            x_new, y_new = pts[:, 0], pts[:, 1]
        self.ax.plot(x_new, y_new, "b-")

    def _draw_ellipse(self):
        oval = Ellipse(
            xy=(random.randint(800, 1200), random.randint(600, 900)),
            width=random.randint(100, 500),
            height=random.randint(100, 500),
            angle=random.uniform(0, 90),
            edgecolor="r",
            facecolor="none",
        )
        self.ax.add_patch(oval)

    def _draw_line(self):
        self.ax.plot(
            [10, 1800],
            [random.randint(10, 1000), 1000],
            "g--",
            linewidth=random.uniform(1, 3),
        )

    def render(self, output_path: PathLike) -> Tuple[Optional[Path], Optional[Exception]]:
        """Clear, redraw, and save the image."""
        try:
            self.ax.cla()
            self.ax.set_xlim(0, self.img_size[0])
            self.ax.set_ylim(0, self.img_size[1])
            self.ax.invert_yaxis()
            self.ax.axis("off")

            with plt.xkcd():
                self._draw_spline()
                self._draw_ellipse()
                self._draw_line()

            out = Path(output_path)
            self.fig.savefig(out, dpi=self.dpi, format="jpg", bbox_inches="tight", pad_inches=0)
            return out, None
        except Exception as e:
            return None, e

    def __del__(self):
        plt.close(self.fig)
        del self.ax
        del self.fig


def worker_init():
    global _worker
    _worker = SyntheticImageWorker()


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[Exception]]:
    global _worker
    return _worker.render(output_path)


def main():
    BATCH_SIZE = 90
    OUTPUT_DIR = Path("./synth_images_refactored")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_cores = max(1, int(os.cpu_count() * 0.75))

    print(f"Using {num_cores} workers...")
    job_paths = [OUTPUT_DIR / f"synthetic_{i:06d}.jpg" for i in range(BATCH_SIZE)]

    start = time.time()
    with Pool(processes=num_cores, initializer=worker_init) as pool:
        for i, (path, err) in enumerate(pool.map(main_worker, job_paths, chunksize=10)):
            if err:
                print(f"[Job {i}] Failed: {err}")
    elapsed = time.time() - start
    print(f"Generated {BATCH_SIZE} images in {elapsed:.2f}s "
          f"({BATCH_SIZE / elapsed:.2f} img/s)")


if __name__ == "__main__":
    main()
