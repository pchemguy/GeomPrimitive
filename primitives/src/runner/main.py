"""
main.py - Entry point for parallel synthetic image generation.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import asdict
from multiprocessing import Pool
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Import handling for both package and script execution
# ---------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from logging_utils import configure_logging
    from summary import RunSummary
    from orchestration import worker_init, main_worker
    from config import WorkerConfig
else:
    from .logging_utils import configure_logging
    from .summary import RunSummary
    from .orchestration import worker_init, main_worker
    from .config import WorkerConfig


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main(batch_size: int = 10, output_dir: Union[Path, str] = "./out") -> None:
    """Run parallel synthetic image generation."""
    config = WorkerConfig()

    log_path = configure_logging(level=config.logger_level)
    logger = logging.getLogger("main")
    logger.info(f"WorkerConfig: {asdict(config)}")
    logger.debug(f"Logged at logging.DEBUG level.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_cores = os.cpu_count() or 1
    num_cores = max(1, int(total_cores * 0.75))
    if total_cores > 10:
        num_cores = max(10, num_cores)

    logger.info(f"Using {num_cores} workers for {batch_size} images...")
    logger.info(f"Logs written to: {log_path}")

    summary = RunSummary(total_jobs=batch_size, log_path=log_path)
    job_paths = [output_dir / f"synthetic_{i:06d}.jpg" for i in range(batch_size)]
    results_meta = {}

    try:
        with Pool(processes=num_cores, initializer=worker_init, initargs=(config,)) as pool:
            for i, (path, meta, err) in enumerate(
                pool.imap_unordered(main_worker, job_paths, chunksize=10)
            ):
                summary.record_result(success=(err is None))
                if err:
                    logger.error(f"Job {i} failed: {err}")
                elif path and meta:
                    results_meta[str(path)] = json.loads(meta)
    finally:
        summary.finalize()

    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_file = output_dir / f"batch_{ts}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(results_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Batch metadata written: {batch_file}")


if __name__ == "__main__":
    main()
