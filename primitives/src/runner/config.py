"""
config.py - Configuration dataclass synthetic image generation.

Multiprocessing machinery serializes it, passes to each thread
worker context, and deserializes for use by ThreadWorker.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable configuration passed to each ThreadWorker process."""
    img_size: Tuple[int, int] = (1920, 1080)
    dpi: int = 100
    output_dir: Path = Path("./out")

    def __post_init__(self):
        # Ensure paths exist for safety
        self.output_dir.mkdir(parents=True, exist_ok=True)
