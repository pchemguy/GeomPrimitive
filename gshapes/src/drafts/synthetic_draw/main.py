import os
import sys
import json
import time
import logging
from pathlib import Path
from multiprocessing import Pool

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from logging_utils import configure_logging
    from summary import RunSummary
    from orchestration import worker_init, main_worker
else:
    from .logging_utils import configure_logging
    from .summary import RunSummary
    from .orchestration import worker_init, main_worker


def main():
    log_path = configure_logging()
    logger = logging.getLogger("main")

    BATCH_SIZE = 90
    OUTPUT_DIR = Path("./synth_images_final")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_cores = max(1, int(os.cpu_count() * 0.75))

    summary = RunSummary(total_jobs=BATCH_SIZE, log_path=log_path)
    logger.info(f"Using {num_cores} workers for {BATCH_SIZE} images...")
    logger.info(f"Logs written to: {log_path}")

    job_paths = [OUTPUT_DIR / f"synthetic_{i:06d}.jpg" for i in range(BATCH_SIZE)]
    results_meta = {}

    try:
        with Pool(processes=num_cores, initializer=worker_init) as pool:
            for i, (path, meta, err) in enumerate(pool.map(main_worker, job_paths, chunksize=10)):
                summary.record_result(success=(err is None))
                if err:
                    logger.error(f"Job {i} failed: {err}")
                elif path and meta:
                    results_meta[str(path)] = json.loads(meta)
    finally:
        summary.finalize()

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    batch_file = OUTPUT_DIR / f"batch_{ts}_{ms:03d}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(results_meta, f, indent=4, ensure_ascii=True)

    logger.info(f"Batch metadata written - {batch_file}")


if __name__ == "__main__":
    main()
