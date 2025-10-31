"""
orchestration.py - Worker orchestration logic for multiprocessing harness.
"""

import os
import sys
import logging
import multiprocessing as mp
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
LOGGER_NAME = "worker"
_worker: Optional[ThreadWorker] = None

_init_error: Union[Exception, None] = None
_init_error_traceback = None


def worker_init(config: WorkerConfig) -> None:
    """Initializer for multiprocessing.Pool workers (per process)."""
    try:
        pid = os.getpid()

        # configure per-worker log (file + colorized console)
        log_path = configure_logging(
            level=config.logger_level,
            name=LOGGER_NAME,
            run_prefix=f"worker_{pid}"
        )
        logger = logging.getLogger(LOGGER_NAME)
        logger.info(f"[worker_init] Starting worker PID={pid}")
        logger.debug(f"WorkerConfig: {config!r}")

        # create worker instance (actual figure, RNG, etc.)
        global _worker
        _worker = ThreadWorker(img_size=config.img_size, dpi=config.dpi, config=config)
        logger.info(f"[worker_init] Worker PID={pid} initialized OK -> {log_path}")

    except Exception as e:
        import traceback
        global _init_error
        global _init_error_traceback
        _init_error = e
        _init_error_traceback = traceback.format_exc()
        # print to stderr in case logging itself failed
        print(f"[worker_init][PID={os.getpid()}] FATAL: {e}\n{_init_error_traceback}", file=sys.stderr, flush=True)


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]:
    """Execute one drawing job and return (path, meta_json, error)."""
    global _worker
    global _init_error
    global _init_error_traceback
    if _init_error:
        logging.getLogger("worker").error(f"Worker-{os.getpid()} initilization error in worker_init().")
        logging.getLogger("worker").error(f"Error: {_init_error}. Traceback:")
        logging.getLogger("worker").error(f"{_init_error_traceback}")
        raise _init_error

    try:
        _worker.plot_reset()
        for _ in range(5):
            _worker.draw_line()
            # _worker.draw_line(orientation="vertical", color="blue", alpha=1, hand_drawn=True)
        out, meta = _worker.save_image(output_path)
        return out, meta, None
    except Exception as e:
        logging.getLogger(LOGGER_NAME).error(f"Failed to process {output_path}: {e}")
        return None, None, e
