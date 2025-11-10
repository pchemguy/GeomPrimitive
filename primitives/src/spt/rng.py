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
from typing import Optional, Any


# =============================================================================
# RNG CLASS (thread-safe)
# =============================================================================
class RNG:
    """Encapsulated thread-safe random generator.

    Attributes:
        _rng:  Underlying random.Random instance.
        _lock: threading.Lock for safe concurrent access.

    Notes:
        - Safe for use across threads and processes.
        - For frequent multi-threaded RNG use, prefer `get_rng(thread_safe=True)`.
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
    def randint(self, *a, **kw) -> int:
        with self._lock:
            return self._rng.randint(*a, **kw)

    def randrange(self, *a, **kw) -> int:
        with self._lock:
            return self._rng.randrange(*a, **kw)

    def uniform(self, *a, **kw) -> float:
        with self._lock:
            return self._rng.uniform(*a, **kw)

    def choice(self, *a, **kw) -> Any:
        with self._lock:
            return self._rng.choice(*a, **kw)

    def shuffle(self, seq):
        with self._lock:
            self._rng.shuffle(seq)
            return seq

    def normal(self, mu: float, sigma: float) -> float:
        with self._lock:
            return self._rng.normalvariate(mu, sigma)

    def normalvariate(self, mu: float, sigma: float) -> float:
        with self._lock:
            return self._rng.normalvariate(mu, sigma)

    def paretovariate(self, alpha: Optional[float] = 1) -> float:
        with self._lock:
            return self._rng.paretovariate(alpha)

    def getstate(self):
        with self._lock:
            return self._rng.getstate()

    def setstate(self, state):
        with self._lock:
            return self._rng.setstate(state)

    def random(self) -> float:
        with self._lock:
            return self._rng.random()

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
    except ImportError:
        pass    
