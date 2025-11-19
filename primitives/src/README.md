# Overview

This repo contains several components with focus on two objective:  
- Generation of scenes with randomized grids, geometric primitives (lines, elliptic arcs, triangles, and rectangles), randomized styles (color, line patterns, thickness, transparency), handwriting imitation (straight line segments and elliptic arcs are represented by a chain of splines with random deviations from ideal shape; angles and coordinates are jittered). The vector scene is rendered with Matplotlib, rasterized, and a suite of photo like distortions is then introduced as the second stage (noise, rotation, lens distortion, uneven lighting).
- Analysis and enhancement of real photographs of biological samples over millimeter graph paper. The main objective is to detect the background grid, analyze it, and use geometric distortion information for compensation of distortion and subsequent programmatic sample area measurement.

## Python Environment

Python environment bootstrapper is located within "python_env/ImageGen" (see also `pchemguy/FFCVonWindows` and `pchemguy/aipg:win10/Python Bootsrap`):
- `Anaconda.bat` is used to bootstrap and setup environment.
- `conda_far.bat` starts an activated shell with `FarManager`, if available.

## Photo Enhancement Tool (pet)

`pet`

This is an early draft. Main entry is `pet_allinone.py`. The source image is hardcoded inside `SAMPLE`. The core implemented code is in `pet_geom.py` expects a pre-enhanced image and performs initial detection of millimeter graph grid in the background and subsequent early fileting and sorting into vertical/horizontal families. (Tested with `photo_2025-11-17_23-50-05.jpg` image processed through [Fiji ImageJ](https://fiji.sc/) - Plugins -> Integral Image Filters -> Normalize Local Contrast - 40x40x5.00/center/stretch.) Note, attempt to perform WB and exposure correction resulted in images with more severe lighting unevenness.

When executed and the image file location is correctly specified inside the script, it should generate `debug_raw_segments.jpg` and `debug_filtered_segments.jpg` containing the source image with overlay of raw detected segments and the two prefiltered sets. 

## Primitives

`primitives`

This is the first attempt at developing random scene generator. The core modules are `base.py` and `line.py`. These modules do not include self-tests / demo code. Sample usage can be found in tests. Alternatively, this package is used by the multiprocessing framework to generate random images. See [Multiprocessing](Multiprocessing) for details.

## Multiprocessing Overview

### 1. Entry Point: `main.py`

This script is the primary driver for batch synthetic-image generation.

#### Responsibilities

* Parse arguments (e.g., `batch_size`, `output_dir`) or use default values.
* Instantiate the immutable configuration dataclass (`WorkerConfig`).
* Configure logging for the main process (via `configure_logging`).
* Determine a target number of worker processes (approximately 0.75 × CPU cores, with a safe upper cap).
* Generate a list of output file paths (`job_paths`) for `batch_size` images:

  ```python
  job_paths = [output_dir / f"synthetic_{i:06d}.jpg" for i in range(batch_size)]
  ```
* Launch a `multiprocessing.Pool` of worker processes, using `imap_unordered` to distribute the jobs.
* Monitor results: count successes/failures via `RunSummary`, abort early if failures exceed a threshold.
* Collect per-image metadata JSON returned by each worker, compile it into a batch JSON file at the end.
* Print a summary of the run (duration, throughput, failures) via `RunSummary.finalize()`.

#### Why this matters

Using `Pool.imap_unordered` allows work to be dynamically balanced across processes (faster workers don’t idle).
Early failure detection (via threshold) ensures long-running batches don’t silently degrade.

---

### 2. Orchestration Layer: `orchestration.py`

This module handles worker initialization and task dispatch.

#### Key Functions

##### `worker_mp_pool_init(config: WorkerConfig)`

* Called **once** in each new worker process before any tasks run.
* Configures process-local logging.
* Instantiates the `ProcessWorker` (see next section) with the supplied `config`.
* Seeds the RNG and creates the drawing canvas.
* If any exception occurs here, it is **caught**, recorded to module-level `_init_error` and `_init_error_traceback`, and then suppressed from propagating — to prevent the `multiprocessing` pool from endlessly respawning faulty workers.

##### `main_worker_imap_task(output_path: PathLike) -> Tuple[Optional[Path], Optional[str], Optional[Exception]]`

* Executed once per job by the `Pool` processes.
* Steps:
    1. If `_init_error` is set (indicating failed initialization), log an error and re-raise the stored exception — this causes the parent process to abort cleanly.
    2. Call `ProcessWorker.plot_reset()` to prepare the canvas and metadata container.
    3. Execute the drawing loop (currently `for _ in range(5): draw_line()`).
    4. Call `ProcessWorker.save_image(output_path)` -> returns `(path, meta_json_str)`.
    5. Return `(path, meta_json_str, None)` on success, or `(None, None, exception)` on failure.

#### Architectural Note

The current design embeds the fixed “draw 5 lines” logic into the orchestration layer.
A cleaner design would move that logic into a method of `ProcessWorker` (e.g., `process_one_image(output_path)`), letting `main_worker_imap_task()` simply forward the `output_path` to that method.
If you’d like, we can refactor accordingly.

---

### 3. Worker Implementation: `worker.py`

This module implements the per-process drawing logic via the `ProcessWorker` class.

#### Responsibilities

* On worker startup:
    * Determine a unique seed (`pid ^ time.microseconds`) to ensure separate streams across processes.
    * Seed global primitives and RNG via `Primitive.reseed()`.
    * Create a Matplotlib figure and axes (`plt.subplots`) sized according to `img_size` and `dpi`.
* For each job:
    * `plot_reset()` – clears the axes, sets coordinate limits, disables axis display, and resets metadata container (including timestamp, seed, PID, config).
    * `draw_line(**kwargs)` – uses the reusable `Line` primitive to create geometry and draw it to the axes; appends metadata for each primitive drawn.
    * `save_image(output_path)` – saves the figure into a JPG file; returns the path and metadata JSON string.

#### Why this matters

* Re-using a single `ProcessWorker` per process avoids repeated expensive figure setups.
* Seeding per process isolates random streams, ensuring reproducible behaviour.
* Encapsulating metadata collection per image ensures traceability of every drawing primitive.

---

### 4. Supporting Modules

#### `utils/logging_utils.py`

Sets up standard logging for both main and worker processes:  
* Colorized console output (`ColorFormatter`)
* Rotating file handler for persistent logs
* Ensures both console and file logs use consistent format and timestamping

#### `utils/rng.py`

Provides the `RNG` class and `get_rng()` accessor:  
* Thread-safe and process-safe random number generation
* Supports either Python’s `random.Random` or NumPy’s `random.Generator` backend
* Useful when primitives or other modules need reproducible random values

#### `config.py`

Defines the `WorkerConfig` dataclass:  
* Contains fields like `img_size`, `dpi`, `logger_level`, `output_dir`
* Marked `frozen=True` for immutability and safe pickling across processes
* Ensures the `output_dir` exists in `__post_init__()`

#### `summary.py`

Captures run-level statistics:  
* Tracks total jobs, completed jobs, failed jobs, start and end times
* Computes throughput (images per second)
* Prints a colored summary footer to the console

---

### 5. Summary and Best Practices

* **Separation of concerns**:  
    * `main.py` → orchestration of the batch and pool.
    * `orchestration.py` → process initialization + per-job dispatch.
    * `worker.py` → actual drawing logic and metadata collection.
    * Utilities handle logging, random generation, configuration, summary output.
* **Robust error handling**:
    * Worker initialization exceptions are trapped, preventing spawn-restart loops.
    * Task failures are counted, and excessive failures trigger early termination of the batch.
* **Scalability**:
    * The number of workers scales with system cores.
    * `imap_unordered` ensures work is evenly distributed.
    * Per-process resources (figure, RNG) reduce overhead.
* **Extensibility**:
    * You can change number or type of primitives drawn by modifying `ProcessWorker`.
    * Additional job-types (e.g., draw circles, textures) can reuse the same pool and orchestration.

---



