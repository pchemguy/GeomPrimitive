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
import multiprocessing as mp
from multiprocessing import Pool
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Import handling for both package and script execution
# ---------------------------------------------------------------------------
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.logging_utils import configure_logging
if __package__ is None or __package__ == "":
    from summary import RunSummary
    from orchestration import worker_mp_pool_init, main_worker_imap_task
    from config import WorkerConfig
else:
    from .summary import RunSummary
    from .orchestration import worker_mp_pool_init, main_worker_imap_task
    from .config import WorkerConfig


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main(batch_size: int = 100, output_dir: Union[Path, str] = "./out") -> None:
    """Run parallel synthetic image generation."""
    config = WorkerConfig()
    main_process = mp.current_process() 

    log_path = configure_logging(
        level=config.logger_level,
        name="root",
        run_prefix=f"main_{main_process.pid}"
    )
    logger = logging.getLogger("main")

    logger.info(f"Running MainProcess: {main_process}")
    logger.info(f"Process name: {main_process.name}")
    logger.info(f"Process PID: {main_process.pid}")
    logger.info(f"Process PID: {os.getpid()}")

    logger.info("Starting parallel generation")
    logger.info(f"WorkerConfig: {asdict(config)}")

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
    
    max_failures: int = 5
    fail_count: int = 0
    try:
        with Pool(processes=num_cores, initializer=worker_mp_pool_init, initargs=(config,)) as pool:
            for i, result in enumerate(pool.imap_unordered(main_worker_imap_task, job_paths, chunksize=10)):
                if not result:
                    err = mp.ProcessError
                else:
                    path, meta, err = result
                summary.record_result(success=(err is None))
                if err:
                    fail_count += 1
                    if fail_count >= max_failures:
                        logger.critical(f"Fatal error in job {i}: {err}")
                        logger.critical(f"Too many worker failures ({fail_count}). Aborting.")
                        pool.terminate()
                        raise err
                else:
                    results_meta[str(path)] = json.loads(meta)
    except Exception as e:
        logger.critical(f"Run aborted due to fatal error: {e}")
        raise SystemExit(1)
    finally:
        summary.finalize()

    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_file = output_dir / f"batch_{ts}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(results_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Batch metadata written: {batch_file}")


if __name__ == "__main__":
    main()
