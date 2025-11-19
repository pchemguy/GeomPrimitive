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

## Multiprocessing

### `main.py`

Main entry to be executed. It contains the process setup/spawning `with Pool` loop.

The `runner` package implements basic Python multiprocessing framework based on `Pool` and `imap_unordered`. This framework spawns a predefined number of subprocesses (0.75 * number of cores) and use this process to execute the queue, which involves generation of multiple images (`batch_size`). Each image is identified by its paths and this id is also used as task identifier. Generated `job_paths` list acts as a job queue.

### `orchestration.py`

Contains two routines:
- `worker_mp_pool_init()` is executed once for each spawn process to initialize it.
- `main_worker_imap_task()` is the main task entry. For each list element in `job_paths`, the multiprocessing manager passes entry to subprocess manager that executes `main_worker_imap_task()` passing the current `job_paths` entry.

- `orchestration.py`: 
- `config.py`: basic config file, implemented as frozen dataclass suitable for pickling/unpickling when passed between the parent and spawn processes.
- `logging_utils.py`: performs basic logger configuration `summary.py` is used for report summaries.


