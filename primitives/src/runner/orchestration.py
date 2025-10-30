"""
orchestration.py - Worker orchestration logic for multiprocessing harness.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import ThreadWorker
    from config import WorkerConfig
else:
    from .worker import ThreadWorker
    from .config import WorkerConfig


PathLike = Union[str, Path]
_worker: Optional[ThreadWorker] = None


def worker_init(config: WorkerConfig) -> None:
    """Initializer for multiprocessing.Pool workers."""
    global _worker
    _worker = ThreadWorker(
        img_size=config.img_size,
        dpi=config.dpi,
        config=config,
    )
    logging.getLogger("worker").debug(f"Worker initialized in PID {os.getpid()}")


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]:
    """Execute one drawing job and return (path, meta_json, error)."""
    global _worker
    try:
        _worker.plot_reset()
        for _ in range(5):
            _worker.draw_line()
        out, meta = _worker.save_image(output_path)
        return out, meta, None
    except Exception as e:
        logging.getLogger("worker").error(f"Failed to process {output_path}: {e}")
        return None, None, e
