"""
logging_utils.py
"""

__all__ = ["configure_logging", "ColorFormatter"]

import os
import time
import logging
from typing import Optional, Union
from logging.handlers import RotatingFileHandler
from pathlib import Path

from colorama import Fore, Style, init as colorama_init

PathLike = Union[str, os.PathLike]


class ColorFormatter(logging.Formatter):
    """Colorized console formatter."""
    COLORS = {
        "DEBUG":    Fore.CYAN,
        "INFO":     Fore.GREEN,
        "WARNING":  Fore.YELLOW,
        "ERROR":    Fore.RED,
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


def configure_logging(level: Optional[int] = logging.INFO,
                      log_dir: Optional[PathLike] = "logs",
                      name: Optional[str] = "root",
                      run_prefix: Optional[str] = "run") -> Path:
    """Configure colorized console + rotating file logging."""
    colorama_init(strip=False, convert=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = log_dir / f"{run_prefix}_PID{os.getpid()}_{ts}.log"
  
    mono_fmt = "[%(asctime)s] [%(process)5d] [%(levelname)-5s] %(message)s"
    datefmt = "%H:%M:%S"
  
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
  
    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter(datefmt=datefmt))

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(logging.Formatter(mono_fmt, datefmt))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging initialized - PID {os.getpid()}; file {log_path}")
    return log_path
