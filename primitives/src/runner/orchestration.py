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
WORKER_LOGGER_NAME = "worker"
_worker: Optional[ThreadWorker] = None


def worker_init(config: WorkerConfig):
    """Initializer for multiprocessing.Pool workers."""
    try:
        log_path = configure_logging(level=config.logger_level, name=WORKER_LOGGER_NAME)
        logger = logging.getLogger(WORKER_LOGGER_NAME)
        logger.info(f"Worker initialized PID {os.getpid()}")
        global _worker
        _worker = ThreadWorker(img_size=config.img_size, dpi=config.dpi, config=config)
    except Exception as e:
        import traceback
        print(f"[worker_init] FATAL: {e}\n{traceback.format_exc()}", flush=True)
        raise  # bubble up to main process


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
        logging.getLogger(WORKER_LOGGER_NAME).error(f"Failed to process {output_path}: {e}")
        return None, None, e
