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

## Synthetic Photo Tool (SPT) — Architectural Overview


## 1. System Overview

SPT is divided into two large stages:

#### **A. Vector Scene Generation (Matplotlib subsystem: `mpl_*`)**

Produces **clean, analytical vector graphics** — such as graph paper, grids, geometric paths, or simple drawings — and renders them into RGBA images.

#### **B. Image Degradation / Camera Artefacts (SPT subsystem: `spt_*`)**

Transforms the clean render into something that resembles a **real photographic capture**, with lighting variations, noise, textures, optics, sensor quirks, and lens geometry.

A **unified composite pipeline class** that glues these two stages together **does not exist yet** but is planned.

---

## 2. Two Generations of the SPT Subsystem

#### **Generation 1 — Procedural Effects Pipeline**

This is the current, functioning, end-to-end system.
Entry point: **`spt.py`**.

Implements a straightforward deterministic chain:

```
Matplotlib render
 → Lighting
 → Texture
 → Noise
 → Geometry
 → Color
```

#### **Generation 2 — Correction Engine**

A more advanced system that models imaging as a **physical camera pipeline**, including RAW domain, sensor physics, CFA demosaicing artifacts, ISP tone mapping, PRNU/FPN noise, sharpening, JPEG modeling, etc.

Modules:

* `spt_correction_engine.py`
* `spt_correction_engine_random.py`

These will eventually replace portions of Generation 1.

A **major pending redesign** (to be implemented) is documented here:

> [https://chatgpt.com/c/691b778b-c4dc-8325-b7ec-9537f67a0e25](https://chatgpt.com/c/691b778b-c4dc-8325-b7ec-9537f67a0e25)

Ultimately, both generations will merge into a **single coherent subpackage**.

---

## 3. Matplotlib Subsystem

*Vector Scene Construction & Rendering*

This stage is responsible ONLY for producing clean vector graphics and converting them to RGBA images.
All degradation is done later by `spt_*`.

---

### **Primary Engines**

#### **1) `mpl_grid_gen.py` — Primary Grid Generator**

Responsible for:

* analytic grid geometry (major/minor lines)
* perspective variants
* density, spacing, obliqueness
* line families, orientations
* construction of vector geometries representing graph paper

This will also **absorb the first three effects** currently in `mpl_grid_gen_effects.py` (which is scheduled for dissolution).

#### **2) `mpl_path_utils.py` — Primary Path Engine**

Responsible for:

* all manipulation of Matplotlib `Path` objects
* transformations (scale, rotate, translate)
* sampling, splitting, joining
* path deformation utilities
* preparing vector shapes before rendering

This is the core geometric/mathematical engine for vector shapes other than grids.

---

### **Supporting Modules**

| Module                      | Role                                                                                                                  |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **`mpl_utils.py`**          | Shared utilities: type aliases, RGBA↔BGR helpers, rendering helpers, Matplotlib config, grid visualizers.             |
| **`mpl_renderer.py`**       | High-level orchestrator for rendering vector scenes: consistent figsize, dpi, color themes, predictable RGBA capture. |
| **`mpl_artist_preview.py`** | Preview utilities for debugging vector artists and shapes.                                                            |

---

### **Modules Planned for Removal / Consolidation**

| Module                         | Status                                                                                                                                                                                   |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`mpl_grid_gen_effects.py`**  | **To be dissolved.** Contains four “effects”: the first three move to `mpl_grid_gen.py`; the fourth likely belongs to `spt_*` pending confirmation that it is a raster-domain operation. |
| **`mpl_grid_gen_pipeline.py`** | **To be removed.** It served exploratory/demo purposes and is superseded by cleaner designs.                                                                                             |

---

## 4. SPT Subsystem

*Image Degradation / Photographic Artefacts*

These modules take the clean RGBA output of the Matplotlib subsystem and apply ordered degradations to simulate real-camera imperfections.

---

### **Generation 1 — Procedural Pipeline**

| Module                | Responsibility                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| **`spt.py`**          | Current main orchestrator stitching together the first-generation SPT pipeline.                 |
| **`spt_lighting.py`** | Uneven lighting: linear and radial gradients, global exposure-like adjustments.                 |
| **`spt_texture.py`**  | Paper textures: additive/multiplicative noise, multi-scale structure, fibers, creases, presets. |
| **`spt_noise.py`**    | Noise models: Poisson (shot), Gaussian (read), salt-and-pepper, speckle, Gaussian blur.         |
| **`spt_geometry.py`** | Perspective tilt and radial lens distortion (k1, k2).                                           |
| **`spt_color.py`**    | Simple color and vignette models (warmth, falloff).                                             |
| **`spt_pipeline.py`** | Demonstration pipeline; a linear script-based version of the stage chain.                       |
| **`spt_config.py`**   | Batch/interactive mode flags, backend selection.                                                |

---

### **Generation 2 — Correction Engine**

| Module                                | Responsibility                                                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **`spt_correction_engine.py`**        | High-fidelity, physics-inspired camera model in RGBfloat space. Designed to replace large parts of Gen 1. |
| **`spt_correction_engine_random.py`** | Parameter sampler for realistic camera configurations.                                                    |

These two constitute the **future direction** and will eventually integrate into the main SPT package once the redesign is complete.

---

## 5. Integration Status and Planned Evolution

#### **Current**

* Matplotlib (`mpl_*`) and SPT (`spt_*`) exist as **two separate but sequential stages**.
* Integration is currently manual: Generation 1 pipeline uses simple function calls.
* Generation 2 correction engine is developed but **not yet integrated**.
* Certain Matplotlib modules are scheduled for dissolution.

#### **Planned**

* A unified **Composite Pipeline Class** that encapsulates:
  **vector scene → correction engine (Gen 2) → raster artefacts → final output**
* Consolidation of vector effects into the primary engines (`mpl_grid_gen.py`, `mpl_path_utils.py`).
* Implementation of the major correction engine redesign from the proposal.
* Dissolution or merging of obsolete exploratory modules.

---

## 6. Demo Philosophy

Almost every module (`mpl_*` and `spt_*`) follows the same pattern:

- When executed directly (`python module.py`),  
    the module runs a **self-contained demo** of its functionality:
    - Generates a baseline synthetic scene.
    - Runs a parameter sweep or randomized configuration.
    - Displays results via Matplotlib using a grid helper.
    - Prints relevant metadata or debug information.        

This makes each module **self-documenting** and easy to test in isolation.
