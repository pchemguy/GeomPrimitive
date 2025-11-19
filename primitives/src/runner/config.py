"""
config.py - Configuration dataclass synthetic image generation.

Multiprocessing machinery serializes it, passes to each thread
worker context, and deserializes for use by ProcessWorker.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable configuration passed to each ProcessWorker process."""
    logger_level: int = logging.DEBUG
    img_size: Tuple[int, int] = (1024, 1024) #(1920, 1080)
    dpi: int = 100
    output_dir: Path = Path("./out")

    def __post_init__(self):
        # Ensure paths exist for safety
        self.output_dir.mkdir(parents=True, exist_ok=True)
