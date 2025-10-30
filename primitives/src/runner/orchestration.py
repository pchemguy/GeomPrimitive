"""
orchestration.py
"""

import os
import sys
import json
import logging
from typing import Union, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import SyntheticImageWorker
else:
    from .worker import SyntheticImageWorker


PathLike = Union[str, os.PathLike]
_worker = None


def worker_init():
    global _worker
    _worker = SyntheticImageWorker()


def main_worker(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]:
    """Flow logic per job."""
    global _worker
    try:
        _worker.plot_reset()
        with plt.xkcd():
            pass
            # _worker.draw_spline()
            # _worker.draw_ellipse()
            # _worker.draw_line()
        _worker.draw_line_ex()
        _worker.draw_line_ex()
        _worker.draw_line_ex()
        _worker.draw_line_ex()
        _worker.draw_line_ex()
        out, meta = _worker.save_image(output_path)
        return out, meta, None
    except Exception as e:
        logging.getLogger("worker").error(f"Failed to process {output_path}: {e}")
        return None, None, e
