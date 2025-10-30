"""
orchestration.py - Worker orchestration logic.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import SyntheticImageWorker
else:
    from .worker import SyntheticImageWorker

PathLike = Union[str, Path]
_worker: Optional[SyntheticImageWorker] = None


def worker_init():
    """Initializer for multiprocessing.Pool."""
    global _worker
    _worker = SyntheticImageWorker()


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
