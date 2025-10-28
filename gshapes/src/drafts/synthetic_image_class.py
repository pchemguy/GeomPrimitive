import os
import random
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple, Union
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import splprep, splev
from colorama import Fore, Style, init as colorama_init


# =============================================================================
# Types / globals
# =============================================================================
PathLike = Union[str, os.PathLike]
_worker = None
DPI = 100  # fixed global DPI


# =============================================================================
# Logging
# =============================================================================
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL
        record.process_str = f"{record.process:5d}"
        record.level_str = f"{record.levelname:<5s}"
        return (
            f"[{self.formatTime(record, self.datefmt)}] "
            f"[{record.process_str}] "
            f"[{color}{record.level_str}{reset}] "
            f"{record.getMessage()}"
        )


def configure_logging(
    level: int = logging.INFO,
    log_dir: Union[str, Path] = "logs",
    run_prefix: str = "run",
) -> Path:
    """Configure colorized console + rotating file logging."""
    colorama_init()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = log_dir / f"{run_prefix}_{ts}.log"

    mono_fmt = "[%(asctime)s] [%(process)5d] [%(levelname)-5s] %(message)s"
    datefmt = "%H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter(datefmt=datefmt))

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(logging.Formatter(mono_fmt, datefmt))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging initialized - {log_path}")
    return log_path


# =============================================================================
# Run summary footer
# =============================================================================
class RunSummary:
    def __init__(self, total_jobs: int, log_path: Path):
        self.start_time = time.time()
        self.total_jobs = total_jobs
        self.failed_jobs = 0
        self.completed_jobs = 0
        self.log_path = log_path

    def record_result(self, success: bool):
        self.completed_jobs += 1
        if not success:
            self.failed_jobs += 1

    def finalize(self):
        end_time = time.time()
        duration = end_time - self.start_time
        ok = self.total_jobs - self.failed_jobs
        throughput = (ok / duration) if duration > 0 else 0.0

        sep_color = Style.BRIGHT + Fore.WHITE
        title_color = Style.BRIGHT + Fore.CYAN
        thr_color = Fore.GREEN if self.failed_jobs == 0 else Fore.RED
        reset = Style.RESET_ALL

        console_lines = [
            "",
            f"{sep_color}{'=' * 78}{reset}",
            f"{title_color}RUN SUMMARY{reset}",
            f"{sep_color}{'=' * 78}{reset}",
            f"Start Time : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}",
            f"End Time   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
            f"Duration   : {duration:.2f} seconds",
            f"Jobs Total : {self.total_jobs}",
            f"Jobs OK    : {ok}",
            f"Jobs Failed: {self.failed_jobs}",
            f"Throughput : {thr_color}{throughput:.2f} images/sec{reset}",
            f"Log File   : {self.log_path.resolve()}",
            f"{sep_color}{'=' * 78}{reset}",
            "",
        ]
        for line in console_lines:
            print(line)


# =============================================================================
# Worker class (no flow logic)
# =============================================================================
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


# =============================================================================
# Worker glue
# =============================================================================
def worker_init():
    global _worker
    _worker = SyntheticImageWorker()


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]:
    """
    Flow logic per job:
      - reset
      - draw shapes
      - save
    """
    global _worker
    try:
        _worker.plot_reset()
        with plt.xkcd():
            _worker.draw_spline()
            _worker.draw_ellipse()
            _worker.draw_line()
        out, meta = _worker.save_image(output_path)
        if not _worker._meta:
            logging.getLogger("worker").warning(f"No metadata collected for {output_path}")
        return out, meta, None
    except Exception as e:
        logging.getLogger("worker").error(f"Failed to process {output_path}: {e}")
        return None, None, e


# =============================================================================
# Main orchestration
# =============================================================================
def main():
    log_path = configure_logging()
    logger = logging.getLogger("main")

    BATCH_SIZE = 90
    OUTPUT_DIR = Path("./synth_images_final")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_cores = max(1, int(os.cpu_count() * 0.75))

    summary = RunSummary(total_jobs=BATCH_SIZE, log_path=log_path)
    logger.info(f"Using {num_cores} workers for {BATCH_SIZE} images...")
    logger.info(f"Logs written to: {log_path}")

    job_paths = [OUTPUT_DIR / f"synthetic_{i:06d}.jpg" for i in range(BATCH_SIZE)]

    results_meta = {}

    try:
        with Pool(processes=num_cores, initializer=worker_init) as pool:
            for i, (path, meta, err) in enumerate(pool.map(main_worker, job_paths, chunksize=10)):
                summary.record_result(success=(err is None))
                if err:
                    logger.error(f"Job {i} failed: {err}")
                elif path and meta:
                    results_meta[str(path)] = json.loads(meta)
    finally:
        summary.finalize()

    # Serialize batch metadata
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    batch_file = OUTPUT_DIR / f"batch_{ts}_{ms:03d}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(results_meta, f, indent=4, ensure_ascii=True)

    logger.info(f"Batch metadata written - {batch_file}")


if __name__ == "__main__":
    main()
