"""
orchestration.py - Worker orchestration logic for multiprocessing harness.
"""

import os
import sys
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from logging_utils import configure_logging
    from worker import ThreadWorker
    from config import WorkerConfig
else:
    from .logging_utils import configure_logging
    from .worker import ThreadWorker
    from .config import WorkerConfig


PathLike = Union[str, Path]
_worker: Optional[ThreadWorker] = None


def worker_init(config: WorkerConfig) -> None:
    """Initializer for multiprocessing.Pool workers."""
    log_path = configure_logging(level=config.logger_level, name="worker")
    logger = logging.getLogger("worker")
    logger.info(f"Worker initialized in PID {os.getpid()}")
    logger.debug(f"WorkerConfig: {asdict(config)}")

    global _worker
    _worker = ThreadWorker(
        img_size=config.img_size,
        dpi=config.dpi,
        logger=logging.getLogger("worker"),
        config=config,
    )


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]:
    """Execute one drawing job and return (path, meta_json, error)."""
    global _worker
    try:
        _worker.plot_reset()
        for _ in range(5):
            _worker.draw_line(orientation="vertical", alpha=1, hand_drawn=True)
        out, meta = _worker.save_image(output_path)
        return out, meta, None
    except Exception as e:
        logging.getLogger("worker").error(f"Failed to process {output_path}: {e}")
        return None, None, e
