import os
import random
import time
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


# -----------------------------------------------------------------------------
# Type aliases and globals
# -----------------------------------------------------------------------------
PathLike = Union[str, os.PathLike]
_worker = None  # Global per-process worker instance


# -----------------------------------------------------------------------------
# Colorized logging formatter
# -----------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
  """Adds ANSI colors to log level names for console output."""

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
    # Keep PID and level aligned
    record.process_str = f"{record.process:5d}"
    record.level_str = f"{record.levelname:<5s}"
    msg = f"[{self.formatTime(record, self.datefmt)}] [{record.process_str}] [{color}{record.level_str}{reset}] {record.getMessage()}"
    return msg


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def configure_logging(
    level: int = logging.INFO,
    log_dir: Union[str, Path] = "logs",
    run_prefix: str = "run",
) -> Path:
  """
  Configure colorized console + rotating file logging.
  Returns the path to the log file for this run.
  """
  colorama_init()  # Safe on all platforms

  log_dir = Path(log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  ts = time.strftime("%Y-%m-%d_%H%M%S")
  log_path = log_dir / f"{run_prefix}_{ts}.log"

  # Monochrome format (for file handler)
  mono_fmt = "[%(asctime)s] [%(process)5d] [%(levelname)-5s] %(message)s"
  datefmt = "%H:%M:%S"

  logger = logging.getLogger()
  logger.setLevel(level)
  for h in list(logger.handlers):
    logger.removeHandler(h)

  # --- Console handler with color ---
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(ColorFormatter(datefmt=datefmt))

  # --- File handler (no color, aligned) ---
  file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
  file_handler.setFormatter(logging.Formatter(mono_fmt, datefmt))

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  logger.info(f"Logging initialized - {log_path}")
  return log_path


# -----------------------------------------------------------------------------
# SyntheticImageWorker class
# -----------------------------------------------------------------------------
class SyntheticImageWorker:
  """Encapsulates a per-process Matplotlib figure, drawing logic, and saving."""

  def __init__(self, img_size: Tuple[int, int] = (1920, 1080), dpi: int = 100):
    self.img_size = img_size
    self.dpi = dpi
    self.logger = logging.getLogger(f"worker-{os.getpid()}")
    self.fig, self.ax = self._create_canvas()
    self._seed_rng()
    self.logger.info("Initialized SyntheticImageWorker")

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def _seed_rng(self):
    pid = os.getpid()
    seed = pid ^ int(time.time())
    random.seed(seed)
    np.random.seed((pid * 2654435761) % 2**32)
    self.logger.debug(f"Random seed set: {seed}")

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
    m, k = len(pts), min(3, len(pts) - 1)
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
    """Clear, redraw, and save the image to file."""
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
      self.fig.savefig(
          out, dpi=self.dpi, format="jpg", bbox_inches="tight", pad_inches=0
      )
      self.logger.debug(f"Saved image: {out}")
      return out, None
    except Exception as e:
      self.logger.error(f"Render failed: {e}")
      return None, e

  def close(self):
    """Release resources deterministically."""
    plt.close(self.fig)
    self.logger.info("Figure closed and resources released.")


# -----------------------------------------------------------------------------
# Multiprocessing glue
# -----------------------------------------------------------------------------
def worker_init():
  global _worker
  # Workers inherit logging handlers
  _worker = SyntheticImageWorker()


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[Exception]]:
  global _worker
  return _worker.render(output_path)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main():
  log_path = configure_logging()
  logger = logging.getLogger("main")

  BATCH_SIZE = 90
  OUTPUT_DIR = Path("./synth_images_final")
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  num_cores = max(1, int(os.cpu_count() * 0.75))

  logger.info(f"Using {num_cores} workers for {BATCH_SIZE} images...")
  logger.info(f"Logs written to: {log_path}")

  job_paths = [OUTPUT_DIR / f"synthetic_{i:06d}.jpg" for i in range(BATCH_SIZE)]

  start = time.time()
  with Pool(processes=num_cores, initializer=worker_init) as pool:
    for i, (path, err) in enumerate(pool.map(main_worker, job_paths, chunksize=10)):
      if err:
        logger.error(f"Job {i} failed: {err}")
  elapsed = time.time() - start
  logger.info(
      f"Completed {BATCH_SIZE} images in {elapsed:.2f}s "
      f"({BATCH_SIZE / elapsed:.2f} img/s)"
  )


if __name__ == "__main__":
  main()
