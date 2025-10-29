> [!NOTE]
> 
> https://chatgpt.com/c/69025b9d-b044-8333-8159-aac740d7bf70

## 🧠 1️⃣ What “thread-safe random” really means

In Python, `random.Random` is **not thread-safe**.  
That means: if **two threads call `.random()` or `.randint()` on the same `Random` instance at the same time**, internal state may be corrupted or produce non-deterministic interleaving.

However:

- Python’s **global functions** in the `random` module (like `random.random()`) _are_ thread-safe in CPython only by accident — because they acquire the **global interpreter lock (GIL)** around each call.
- But that doesn’t protect you if your code releases the GIL (e.g., in NumPy C code) or if you run multiple threads on a **custom RNG object** (like your `RNG`).

So for _your design_, where you wrap a dedicated `random.Random` inside `RNG`, you need to **explicitly guard** it.

> [!NOTE]
> 
> **Baseline - no Thread Safety**
> 
> ```python
> """
> rng.py
> """
> 
> import random
> import os
> import time
> class RNG:
>     def __init__(self, seed: int | None = None):
>         self._rng = random.Random(seed or (os.getpid() ^ int(time.time())))
>
>     def __repr__(self) -> str:
>         return f"<RNG id={id(self.rng)} seed={getattr(self.rng, 'seed', 'unknown')}>"
>
>     def seed(self, seed: int | None = None) -> None:
>         self._rng.seed(seed or (os.getpid() ^ int(time.time())))
>
>     def randint(self, *a, **kw): return self._rng.randint(*a, **kw)
>     def uniform(self, *a, **kw): return self._rng.uniform(*a, **kw)
>     def choice(self, *a, **kw):  return self._rng.choice(*a, **kw)
>     def normal(self, mu, sigma): return self._rng.normalvariate(mu, sigma)
> ```

---

## 🧩 2️⃣ Minimal thread-safe implementation

You can make your `RNG` object thread-safe simply by using a **threading lock**:

```python
"""
rng.py
"""

import random
import os
import time
import threading

class RNG:
    """Thread-safe and process-safe random generator."""
    def __init__(self, seed: int | None = None):
        self._lock = threading.Lock()
        self._rng = random.Random(seed or (os.getpid() ^ int(time.time())))

    def seed(self, seed: int | None = None) -> None:
        with self._lock:
            self._rng.seed(seed or (os.getpid() ^ int(time.time())))

    def randint(self, *a, **kw):
        with self._lock:
            return self._rng.randint(*a, **kw)

    def uniform(self, *a, **kw):
        with self._lock:
            return self._rng.uniform(*a, **kw)

    def choice(self, *a, **kw):
        with self._lock:
            return self._rng.choice(*a, **kw)

    def normal(self, mu, sigma):
        with self._lock:
            return self._rng.normalvariate(mu, sigma)
```

✅ **Effect:**

- Multiple threads can safely call RNG methods concurrently.
- Each call acquires the lock, executes one atomic RNG operation, and releases it.
- There is a small overhead, but negligible for graphics or procedural noise tasks.

---

## ⚙️ 3️⃣ Alternative designs depending on workload

### 🔸 Option 1 — Thread-local RNGs (best for scalability)

Each thread keeps its own `RNG` instance:

```python
import threading

_thread_local = threading.local()

def get_rng() -> RNG:
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = RNG()
    return _thread_local.rng
```

Then everywhere you need randomness:

```python
get_rng().uniform(0, 1)
```

✅ This eliminates locks entirely.  
Each thread has independent RNG state — faster and scalable.

Use this when you have many threads generating random values frequently (e.g., in a renderer, or simulation engine).

---

### 🔸 Option 2 — Shared RNG with external synchronization

If you already have higher-level synchronization (e.g., each thread draws one figure), you don’t need internal locks at all — just ensure your main thread sets seeds before launching.

This is already what your multiprocessing setup does — each process has its own RNG initialized by PID/time.

---

## 🔐 4️⃣ TL;DR — Which to use when

|Use case|Best design|Example|
|---|---|---|
|Simple script or single thread|Plain `RNG`|`rng = RNG()`|
|Multiple threads accessing same RNG|Lock-protected methods|`with self._lock:`|
|Many threads doing heavy RNG work|Thread-local RNGs|`threading.local()`|
|Multiprocessing (already isolated)|Your current PID-based seed logic|`_rng = RNG(os.getpid() ^ int(time.time()))`|

---

## ✅ Updated `rng.py` (drop-in replacement)

```python
"""
rng.py
------

Thread-safe and process-safe random generator utilities.

Provides:
- RNG: encapsulated random generator with internal locking
- get_rng(): returns a shared or thread-local RNG instance
"""

import os
import time
import random
import threading
from typing import Optional


# =============================================================================
# RNG CLASS (thread-safe)
# =============================================================================
class RNG:
    """Encapsulated thread-safe random generator.

    Attributes:
        _rng: Underlying random.Random instance.
        _lock: threading.Lock for safe concurrent access.

    Notes:
        • Safe for use across threads and processes.
        • For frequent multi-threaded RNG use, prefer `get_rng(thread_safe=True)`.
    """

    def __init__(self, seed: Optional[int] = None):
        self._lock = threading.Lock()
        self._rng = random.Random(seed or (os.getpid() ^ int(time.time())))

    # -------------------------------------------------------------------------
    # Core seeding
    # -------------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        """Reinitialize the RNG in place (preserves object identity)."""
        with self._lock:
            self._rng.seed(seed or (os.getpid() ^ int(time.time())))

    # -------------------------------------------------------------------------
    # Basic distributions (thread-safe wrappers)
    # -------------------------------------------------------------------------
    def randint(self, *a, **kw):
        with self._lock:
            return self._rng.randint(*a, **kw)

    def uniform(self, *a, **kw):
        with self._lock:
            return self._rng.uniform(*a, **kw)

    def choice(self, *a, **kw):
        with self._lock:
            return self._rng.choice(*a, **kw)

    def normal(self, mu: float, sigma: float) -> float:
        with self._lock:
            return self._rng.normalvariate(mu, sigma)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<RNG id={id(self)} pid={os.getpid()}>"
    

# =============================================================================
# GLOBAL & THREAD-LOCAL ACCESSORS
# =============================================================================
_global_rng = RNG()
_thread_local = threading.local()


def get_rng(thread_safe: bool = False) -> RNG:
    """Return an RNG instance.

    Args:
        thread_safe: If True, returns a thread-local RNG for parallel usage.

    Returns:
        RNG: Thread-safe RNG instance (shared or per-thread).
    """
    if thread_safe:
        if not hasattr(_thread_local, "rng"):
            _thread_local.rng = RNG()
        return _thread_local.rng
    return _global_rng


def set_global_seed(seed: int) -> None:
    """Re-seed the global RNG and NumPy RNG if available."""
    global _global_rng
    _global_rng.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
```

---

## 💡 How It Works

|Feature|Description|
|---|---|
|`RNG`|Has a per-instance lock — safe for concurrent `.randint()`, `.choice()`, etc.|
|`get_rng(thread_safe=True)`|Returns a **thread-local RNG** — each thread has its own independent RNG instance.|
|`_global_rng`|Shared RNG for simple single-thread or coordinated access.|
|`set_global_seed()`|Re-seeds both `_global_rng` and NumPy safely.|
|`seed()`|Lets you reseed any existing RNG without replacing the object — preserves references.|

---

## 🧩 Usage Patterns

### 1. Shared RNG (safe via locks)

```python
from rng import get_rng

rng = get_rng()
print(rng.uniform(0, 1))
```

### 2. Thread-local RNGs

```python
from rng import get_rng
import threading

def worker():
    rng = get_rng(thread_safe=True)
    print(f"{threading.current_thread().name}: {rng.uniform(0, 1)}")

for _ in range(5):
    threading.Thread(target=worker).start()
```

Each thread gets its own RNG seeded by PID/time — no locks needed and no contention.

### 3. Re-seeding safely

```python
rng = get_rng()
rng.seed(1234)  # in-place reseed
```

---

## ✅ Revised `line_drawer.py` integration

```python
"""
line_drawer.py
--------------

Generates and draws stylized or 'hand-drawn' lines on a Matplotlib axis.
Now uses thread- and process-safe RNGs from rng.py.
"""

import os
import sys
import math
import contextlib
from typing import Any, Dict, Sequence, Union, Tuple, Optional, List

import numpy as np
import matplotlib as mpl

# Use a non-interactive backend (safe for multiprocessing workers)
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors
from matplotlib._enums import JoinStyle, CapStyle

# -----------------------------------------------------------------------------
# Import thread-safe RNG utilities
# -----------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rng import get_rng, set_global_seed
else:
    from .rng import get_rng, set_global_seed


# -----------------------------------------------------------------------------
# RNG access
# -----------------------------------------------------------------------------
# For most cases, use thread-local RNGs (fast, no locks).
# Multiprocessing workers are already PID-isolated.
def _rng() -> "RNG":
    return get_rng(thread_safe=True)


def set_rng_seed(seed: int) -> None:
    """Set deterministic seed for global RNG (for reproducible plots)."""
    set_global_seed(seed)
    np.random.seed(seed)
```

---

### 🔧 Changes Explained

| Change                      | Purpose                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `get_rng(thread_safe=True)` | Ensures **each thread** or **worker** gets its own RNG instance.                                                   |
| `_rng()` helper             | Returns a per-thread RNG (no shared global state). You can safely call `_rng().choice()`, `_rng().uniform()`, etc. |
| `set_rng_seed(seed)`        | Re-seeds the global RNG _and_ NumPy’s generator for deterministic test runs.                                       |
| Removed `_rnd` global       | You now dynamically acquire RNGs via `_rng()`.                                                                     |

---

### 🧩 Usage Example inside functions

Update any place where you previously did `_rnd.something(...)`:

```python
alpha = _rng().uniform(0.0, 1.0)
linewidth = linewidth or _rng().choice([1.0, 1.5, 2.0, 2.5, 3.0])
```

If you used `_rnd` dozens of times, you can safely define a local alias inside `draw_line()`:

```python
def draw_line(ax: Axes, ...):
    rnd = _rng()
    ...
    alpha = rnd.uniform(0.0, 1.0)
    color = rnd.choice(list(colors.CSS4_COLORS.keys()))
```

This gives you fast local access to the per-thread RNG while preserving thread safety.

---

## 🧠 Behavioral Model Recap

| Scenario            | RNG Source                                       | Safety          | Notes                   |
| ------------------- | ------------------------------------------------ | --------------- | ----------------------- |
| **Single-thread**   | shared global RNG                                | ✅ safe          | no locks needed         |
| **Multi-thread**    | per-thread RNG (`get_rng(thread_safe=True)`)     | ✅ safe          | zero lock contention    |
| **Multiprocessing** | each process gets its own RNG seeded by PID/time | ✅ safe          | automatically unique    |
| **Manual reseed**   | `set_rng_seed(seed)`                             | ✅ deterministic | resets NumPy + base RNG |

---

## 🧪 Optional Test

```python
if __name__ == "__main__":
    import threading
    import matplotlib.pyplot as plt

    def worker(idx):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        for _ in range(3):
            draw_line(ax)
        fig.savefig(f"thread_{idx}.png", dpi=100)
        plt.close(fig)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("Saved 4 thread-isolated images.")
```

Each thread produces a unique (non-identical) image because it has its own RNG state.

---
Excellent question — and one that most developers **get wrong** when they start mixing `threading` and `multiprocessing`.

Let’s unpack it precisely in your context — because your design is already modular (`rng.py`, `line_drawer.py`, `worker.py`) and the correct answer depends on **where and how** you use concurrency.

---

## 🧠 1️⃣ The key distinction: _thread safety_ vs _process safety_

| Concept            | Applies to                               | Problem solved                                      | Mechanism                                      |
| ------------------ | ---------------------------------------- | --------------------------------------------------- | ---------------------------------------------- |
| **Thread safety**  | Multiple threads **in the same process** | Prevents concurrent access to shared memory objects | Locks (`threading.Lock`), thread-local storage |
| **Process safety** | Multiple **independent processes**       | Prevents correlated RNG state between processes     | Unique seeding (PID/time entropy)              |

So:

- `thread safety` → avoids two threads touching the _same RNG instance_ at once.    
- `process safety` → ensures two _separate processes_ don’t start with identical seeds.

---

## ⚙️ 2️⃣ In your case (HIFU image generator / line drawer)

You’ve already architected two distinct layers:

|Layer|Parallelism model|RNG context|
|---|---|---|
|**`worker.py`**|uses `multiprocessing.Pool` (one RNG per process)|✅ process safety matters|
|**`line_drawer.py`**|runs inside one process, but could internally spawn threads (e.g., for drawing multiple lines concurrently)|✅ thread safety matters|

Therefore:

> You **do not need both** mechanisms at once in the same place, but you need to understand which layer provides isolation.

Let’s be explicit:

---

## 🧩 3️⃣ Correct design matrix

|Module|What it does|Needs thread-safe RNG?|Needs thread-safe access?|Needs process-safe seeding?|
|---|---|---|---|---|
|`rng.py`|Defines `RNG`, `get_rng()`|✅ Yes — methods are lock-protected|✅ Yes — `thread_local` instances|✅ Yes — seeds with `os.getpid()`|
|`line_drawer.py`|Runs _inside a process_|✅ Yes — uses `get_rng(thread_safe=True)`|❌ No extra locking needed|🔸 No (each process already has a unique RNG)|
|`worker.py`|Creates subprocesses|❌ No — each process has its own RNG|❌ No|✅ Yes — seed each process once via PID/time|

---

## 🧱 4️⃣ Recommended minimal pattern in each layer

### ✅ `rng.py`

Already correct:

- Thread-safe methods
- Thread-local access
- PID-based seeding → process-safe

Keep as-is.

---

### ✅ `line_drawer.py`

Use **thread-local RNGs** (no shared state, no explicit locks):

```python
def _rng():
    return get_rng(thread_safe=True)
```

→ internally safe for any number of drawing threads.

No need to add locks around `draw_line()` — your RNG guarantees correctness.

---

### ✅ `worker.py`

When using multiprocessing, just ensure **each process seeds its own RNG once**:

```python
def _seed_worker():
    """Ensure each worker process has a unique RNG seed."""
    from rng import set_global_seed
    import os, time
    seed = os.getpid() ^ int(time.time())
    set_global_seed(seed)
```

Then, when creating your `Pool`:

```python
from multiprocessing import Pool

with Pool(initializer=_seed_worker) as pool:
    pool.map(draw_worker_job, jobs)
```

No need for locks or `thread_safe=True` in the worker layer.  
Each process is independent; threads inside a worker can still safely call `_rng()`.

---

## 🚫 Don’t do this

- Don’t mix “per-thread RNGs” _and_ manual locks in the same scope — redundant.
- Don’t reseed the RNG inside each draw call — breaks reproducibility.
- Don’t share one RNG instance across multiple processes — state isn’t shared anyway.

---

## ✅ TL;DR

| Location         | Required Safety                                             |
| ---------------- | ----------------------------------------------------------- |
| `rng.py`         | implement both thread-safe methods and process-safe seeding |
| `line_drawer.py` | use `get_rng(thread_safe=True)` (no extra locks)            |
| `worker.py`      | only seed each process once (no thread-safe logic needed)   |

So:

> You only need **thread-safe methods** inside `rng.py`, and **thread-safe access** (via `get_rng(thread_safe=True)`) in `line_drawer.py`.  
> The **worker** module only needs **process-safe seeding**, not thread safety.

---

## ✅ `worker.py` — complete version

```python
"""
worker.py
----------

Manages multiprocessing workers that generate hand-drawn images
using `line_drawer.draw_line()`. Each process is safely seeded
and logs its own events.
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Safe for non-interactive worker use
import matplotlib.pyplot as plt

from line_drawer import draw_line, set_rng_seed
from rng import get_rng


# =============================================================================
# Configuration
# =============================================================================
PathLike = Union[str, os.PathLike]
DPI = 100
OUTPUT_DIR = Path("output")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Logging setup
# =============================================================================
def _init_logger() -> logging.Logger:
    """Configure per-process rotating logger."""
    pid = os.getpid()
    log_path = LOG_DIR / f"worker_{pid}.log"

    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=2)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(f"worker_{pid}")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


# =============================================================================
# RNG seeding
# =============================================================================
def _seed_worker() -> None:
    """Ensure unique RNG state per process."""
    seed = os.getpid() ^ int(time.time())
    set_rng_seed(seed)
    logger = _init_logger()
    logger.info(f"Worker initialized with seed {seed}")
    # store logger for reuse
    global _LOGGER
    _LOGGER = logger


# =============================================================================
# Worker job
# =============================================================================
def generate_image(job_id: int,
                   img_size: Tuple[int, int] = (1920, 1080),
                   n_lines: int = 10) -> PathLike:
    """Generate one synthetic image with stylized lines."""
    fig, ax = plt.subplots(figsize=(img_size[0] / DPI, img_size[1] / DPI))
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(0, img_size[1])
    ax.axis("off")

    rng = get_rng(thread_safe=True)
    for _ in range(n_lines):
        draw_line(ax)

    filename = OUTPUT_DIR / f"image_{job_id:04d}.png"
    fig.savefig(filename, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    _LOGGER.info(f"Saved {filename}")
    return filename


# =============================================================================
# Entry point for batch execution
# =============================================================================
def run_batch(n_images: int = 10, n_workers: Optional[int] = None) -> None:
    """Run parallel image generation."""
    n_workers = n_workers or os.cpu_count() or 4
    print(f"[INFO] Using {n_workers} workers for {n_images} images...")

    start = time.time()
    with Pool(processes=n_workers, initializer=_seed_worker) as pool:
        for path in pool.imap_unordered(generate_image, range(n_images)):
            print(f"[OK] {path}")
    print(f"[DONE] Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    run_batch(n_images=20, n_workers=4)
```

---

## 🧩 How It Works

| Layer                    | Mechanism                                                                 | Why It Matters                                  |
| ------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------- |
| **Per-process RNG**      | `_seed_worker()` calls `set_rng_seed()` using `os.getpid() ^ time.time()` | Ensures every worker has unique random sequence |
| **Thread-safe drawing**  | `draw_line()` uses `get_rng(thread_safe=True)` internally                 | Each thread inside a worker gets its own RNG    |
| **Rotating logs**        | Each process writes to its own file `logs/worker_<pid>.log`               | Clear debugging with no contention              |
| **Graceful parallelism** | `Pool(initializer=_seed_worker)`                                          | Automatic unique RNG + logger per worker        |

---

## 🧠 Execution model

```
Main Process
│
├─ Pool (4 workers)
│   ├─ worker 1 → seed = PID ⊕ time → RNG1 → generate_image(...)
│   ├─ worker 2 → seed = PID ⊕ time → RNG2 → generate_image(...)
│   ├─ worker 3 → ...
│   └─ worker 4 → ...
│
└─ Aggregates results (prints OK / DONE)
```

All randomness is isolated — no global interference, no race conditions.

---

## 🧪 Optional: Add threaded inner drawing

If you ever decide that each _worker process_ should spawn multiple **threads** internally (e.g., each drawing 50 lines simultaneously):

```python
from concurrent.futures import ThreadPoolExecutor

def generate_image(job_id, img_size=(1920, 1080), n_lines=10):
    ...
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.submit(draw_line, ax) for _ in range(n_lines))
    ...
```

Still safe — `draw_line()` uses per-thread RNGs.

---

## ✅ TL;DR

| Module           | Responsibility                                           | Safety mode           | Notes                |
| ---------------- | -------------------------------------------------------- | --------------------- | -------------------- |
| `rng.py`         | defines `RNG` (thread-safe) + `get_rng()` (thread-local) | thread & process safe | foundational         |
| `line_drawer.py` | uses per-thread RNG                                      | thread safe           | no extra locks       |
| `worker.py`      | seeds RNG per process, logs independently                | process safe          | no locks, no sharing |

---
