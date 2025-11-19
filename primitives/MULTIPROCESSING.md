# Multiprocessing Framework Design

## Overview

This project implements a **multiprocessing synthetic image generator** built for high-throughput production of line-based synthetic datasets with structured metadata. It emphasizes determinism, efficient multiprocessing, and reusable per-process rendering state.

---

## 1. Architecture Overview

The system is organized into four layers:

```
main.py  →  orchestration.py  →  worker.py  →  primitives / utils / config
```

---

## 2. `main.py` — Entry Point / Batch Controller

### Responsibilities

- Parse arguments or defaults (`batch_size`, `output_dir`, etc.)
- Apply main-process logging configuration
- Instantiate immutable `WorkerConfig`
- Determine number of worker subprocesses (≈ 0.75 × CPU cores)
- Build a list of `job_paths` for output images
- Create a multiprocessing `Pool`
- Dispatch work using `imap_unordered`
- Track job results via `RunSummary`
- Stop early on failure threshold
- Save combined metadata JSON
- Print final summary

### Why it matters

`main.py` contains **no drawing logic**.  
Its job is orchestration, safety, and throughput.

---

## 3. `orchestration.py` — Worker Initialization & Task Execution

### 3.1 Worker Initialization

#### `worker_mp_pool_init(config: WorkerConfig)`

Executed once for each worker process:  
- Configure logging for that process
- Instantiate persistent `ProcessWorker`
- Seed all RNG systems
- Create Matplotlib figure and axes
- Catch _all_ exceptions and store them into module-level variables  
    (`_init_error`, `_init_error_traceback`)

This avoids infinite respawn loops inside Python’s multiprocessing pool.

---

### 3.2 Per-Job Execution

#### `main_worker_imap_task(output_path: Path)`

Does one unit of work:

1. Check for initialization errors stored in `_init_error`
2. Reset the worker (`worker.plot_reset()`)
3. Execute the drawing logic  
    _(currently draws 5 lines; should be moved into `ProcessWorker` main entry)_
4. Save image and metadata (`worker.save_image`)
5. Return tuple:

```
(path, metadata_json, None)
→ success

(None, None, exception)
→ failure
```    

---

## 4. `worker.py` — The Per-Process Drawing Engine

Implements the long-lived drawing engine used by each worker.

### 4.1 ProcessWorker Lifecycle

#### On startup:

- Generate deterministic seed per process: `pid XOR time_microseconds`
- `Primitive.reseed()` for process-level RNG consistency
- Create persistent Matplotlib figure + axes
- Configure logging

#### Per Job

##### `plot_reset()`

- Clear axes
- Reset limits and aspect ratio
- Hide axes
- Create fresh metadata container with:
    - PID
    - Seed
    - Timestamp
    - Config (snapshot)
    - List of primitives drawn

##### `draw_line(**kwargs)`

- Uses reusable `Line` primitive instance
- Calls:

```
line.make_geometry(ax, **kwargs).draw()
```
   
- Appends primitive metadata

##### `save_image(output_path)`

- Saves figure as JPG (`dpi` configurable)
- Serializes metadata to JSON string
- Returns `(path, json_string)`

---

## 5. Supporting Modules

#### `utils/logging_utils.py`

- Console and rotating log files    
- Colored logs (via `ColorFormatter`)
- Unified logging across processes

#### `utils/rng.py`

- Process-safe random number generator
- Supports Python `random` and NumPy `Generator`
- Reseeding entry point for primitives

#### `config.py`

Defines immutable `WorkerConfig` dataclass:  
- `img_size`
- `dpi`
- `logger_level`
- `output_dir`

Validates directory creation.

#### `summary.py`

Tracks batch progress:  
- Completed jobs
- Failed jobs
- Throughput (images/s)    
- Total duration
- Pretty final summary footer

---

## 6. Execution Flow Diagram

```
Main Process
   |
   |-- WorkerConfig
   |
   |-- Create Pool of N workers
              |
              |-- worker_mp_pool_init()
              |       (build ProcessWorker)
              |
              |-- For each job_path:
              |       main_worker_imap_task()
              |           |
              |           |-- worker.plot_reset()
              |           |-- draw_line() × N
              |           |-- worker.save_image()
              |
   |-- Collect results
   |-- Stop pool
   |-- Save run metadata JSON
   |-- Print summary
```

---

## 7. Extending the System

#### Add new primitives

Implement new primitive classes under `primitives/` following the `Line` interface.

#### Add new high-level drawing logic

Define:

```python
def process_one_image(self, output_path):
    self.plot_reset()
    for _ in range(self.random_line_count()):
        self.draw_line()
    return self.save_image(output_path)
```

Then modify `main_worker_imap_task()` to:

```python
return worker.process_one_image(output_path)
```

#### Add new output formats

Extend `save_image()` using Matplotlib backends (PNG, TIFF, etc.).

---

## 8. Key Design Principles

- **Persistent workers** → amortized initialization cost
- **Initialization error capture** → prevents infinite respawn loops
- **Separation of concerns** → easy maintenance
- **Deterministic RNG** → reproducible multiprocessing output
- **Structured metadata** → downstream analysis
- **High throughput** → balanced load via `imap_unordered`

---

## 9. Example Output Metadata

```json
{
  "timestamp": "2025-02-17T10:54:12",
  "pid": 39204,
  "seed": 4021938122,
  "config": {
    "img_size": [512, 512],
    "dpi": 100
  },
  "draw_ops": [
    {
      "Line": {
        "x0": 0.12,
        "y0": 0.88,
        "
```

## Multiprocessing Architecture

### Overview

This project uses a custom multiprocessing framework built on Python’s `multiprocessing.Pool` and `imap_unordered`. The system distributes image‑generation jobs across a fixed number of worker processes (default: **75% of available CPU cores**). Each job corresponds to a single image identified by its file paths, which also function as task identifiers.

Workers share no mutable state; each process maintains a private `ProcessWorker` instance.

### `main.py`

`main.py` is the primary entry point. It:
- Creates the process pool.
- Submits the list of `job_paths` using `Pool.imap_unordered`.
- Manages the batch‑execution lifecycle.

The job list (`job_paths`) acts as a **task queue**, with each entry processed independently.

### `orchestration.py`

This module contains the multiprocessing control logic.

#### `worker_mp_pool_init()` — Worker Initialization

Executed **once per worker subprocess**. Requirements:
- Must **never allow uncaught exceptions**. If an exception escapes, the pool supervisor aggressively restarts the process, risking an infinite restart loop.
- To safely abort the run on initialization failure:
  1. Catch all exceptions inside `worker_mp_pool_init()`.
  2. Store exception info in the module‑level variable `_init_error`.
  3. Later, `main_worker_imap_task()` checks `_init_error` and raises a controlled exception, cleanly terminating the entire run.

#### `main_worker_imap_task(job_entry)` — Worker Task Entry

This is the function invoked for every element of `job_paths`.

**Current issue:** It embeds high‑level generation logic (`for _ in range(5): ...`) directly inside the task function.

**Correct design:**
- Move all high‑level production logic into a dedicated method of `ProcessWorker`.
- `main_worker_imap_task()` should:
  - Validate `_init_error`.
  - Forward the job entry to the high‑level `ProcessWorker` method.
  - Return the result.

This preserves single‑responsibility and makes the system testable and maintainable.

### `worker.py`

Implements the `ProcessWorker` class.
- Each process holds a **persistent instance** (`_worker`).
- This instance is created once in `worker_mp_pool_init()`.
- All work for that subprocess is routed through this object.

### Utilities

- `utils.logging_utils.py` — Logging configuration for parent and worker processes.
- `utils.rng` — Process‑safe random number generator.

### Additional Modules

- `config.py` — Frozen dataclass‑based configuration. Pickle‑safe for transfer to workers.
- `summary.py` — Output aggregation and report‑generation utilities.
