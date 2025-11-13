"""
rng.py
------

Thread-safe and process-safe random generator utilities.

Upgraded version:
- Supports both `random.Random` and `numpy.random.Generator` backends.
- Retains identical API for scalar use.
- Thread-safe lock for concurrent access.
- Easy integration with your SPT pipeline.
"""

from __future__ import annotations

__all__ = ["RNGBackend", "RNG", "get_rng",]

import os
import time
import random
import threading
from numbers import Real, Integral
from typing import Any, TypeAlias, Union

try:
  import numpy as np
  from numpy.typing import NDArray

except ImportError:
  np = None

# ---------------------------------------------------------------------
# Type alias (forward-compatible)
# ---------------------------------------------------------------------
RNGBackend: TypeAlias = Union[random.Random, "np.random.Generator", "RNG"]


# ---------------------------------------------------------------------
# RNG class
# ---------------------------------------------------------------------
class RNG:
    """Encapsulated, thread-safe hybrid random generator.

    Attributes:
        _rng:  Backend RNG (random.Random or numpy.random.Generator).
        _lock: threading.Lock for safe concurrent access.

    Notes:
        - Uses Python stdlib RNG by default (no NumPy dependency).
        - If NumPy is available, you can enable vectorized sampling
          via `.as_numpy()`.
    """

    def __init__(self, seed: int = None, use_numpy: bool = False):
        self._lock = threading.Lock()
        seed_val = (
            seed or 
            os.getpid() ^ (time.time_ns() & 0xFFFFFFFF) ^ random.getrandbits(32)
        )

        self._use_numpy = use_numpy

        if use_numpy and np is not None:
            self._rng: RNGBackend = np.random.default_rng(seed_val)
        else:
            self._rng: RNGBackend = random.Random(seed_val)

    # -----------------------------------------------------------------
    # Core seeding
    # -----------------------------------------------------------------
    def seed(self, seed: int = None) -> None:
        """Reinitialize the RNG in place (preserves object identity)."""
        with self._lock:
            seed_val = (
                seed or 
                os.getpid() ^ (time.time_ns() & 0xFFFFFFFF) ^ random.getrandbits(32)
            )

            if self._use_numpy and np is not None:
                self._rng = np.random.default_rng(seed_val)
            else:
                self._rng.seed(seed_val)

    # -----------------------------------------------------------------
    # Basic random methods
    # -----------------------------------------------------------------
    """
    - If NumPy backend is used and returns NDarray - pass as is.
    - If result is a scalar (0-D ndarray) - automatically converted to float.
    - Otherwise returns Python float (stdlib backend).

    """

    def random(self, *a, **kw) -> float:
        with self._lock:
            out = self._rng.random(*a, **kw)
            if self._use_numpy:
                if isinstance(out, Real):
                    return float(out)
                return out
            return out

    def randint(self, a: int, b: int, **kw) -> int:
        with self._lock:
            if self._use_numpy:
                out = self._rng.integers(a, b + 1, **kw)
                if isinstance(out, Real):
                    return int(out)
                return out
            return self._rng.randrange(a, b + 1)

    def randrange(self, a: int, b: int = None, **kw) -> int:
        with self._lock:
            if b is None: a, b = 0, a
            if self._use_numpy:
                out = self._rng.integers(a, b, **kw)
                if isinstance(out, Real):
                    return int(out)
                return out
            return self._rng.randrange(a, b)

    def uniform(self, *a, **kw) -> Union[float, np.ndarray]:
        """Return a uniform random value or array, matching backend behavior."""
        with self._lock:
            out = self._rng.uniform(*a, **kw)
            if self._use_numpy:
                if isinstance(out, Real):
                    return float(out)
                return out
            return out
    
    def choice(self, *a, **kw) -> Any:
        with self._lock:
            out = self._rng.choice(*a, **kw)
            if self._use_numpy:
                if isinstance(out, Integral):
                    return int(out)
                elif isinstance(out, Real):
                    return float(out)
                else:
                    return out
            return out

    def normal(self, *a, **kw) -> float:
        with self._lock:
            if self._use_numpy:
                out = self._rng.normal(*a, **kw)
                if isinstance(out, Real):
                    return float(out)
                return out
            return self._rng.normalvariate(*a, **kw)

    def normalvariate(self, *a, **kw) -> float:
        with self._lock:
            if self._use_numpy:
                out = self._rng.normal(*a, **kw)
                if isinstance(out, Real):
                    return float(out)
                return out
            return self._rng.normalvariate(*a, **kw)

    def normal3s(self) -> float:
        """Return a normally distributed random value in [-1, 1], sigma = 1/3.
        
        Generates a clipped normal variable centered at 0 with a standard deviation
        of 1/3, truncated to the interval [-1, 1]. Approximately 99.7% of values
        fall within these bounds.
        
        Returns:
            float: Random number in [-1, 1].
        """
        with self._lock:
            if self._use_numpy:
                var = float(self._rng.normal(0, 1.0/3.0))
            else:
                var = self._rng.normalvariate(0, 1.0/3.0)
            return max(-1, min(1, var))

    def shuffle(self, seq: list[Any]) -> list[Any]:
        with self._lock:
            if self._use_numpy and hasattr(self._rng, "permutation"):
                arr = self._rng.permutation(seq)
                return list(arr)
            self._rng.shuffle(seq)
            return seq
    
    # -----------------------------------------------------------------
    # Utility & Introspection
    # -----------------------------------------------------------------
    def getstate(self):
        with self._lock:
            if self._use_numpy:
                return self._rng.bit_generator.state
            return self._rng.getstate()

    def setstate(self, state):
        with self._lock:
            if self._use_numpy:
                self._rng.bit_generator.state = state
            else:
                self._rng.setstate(state)

    def as_numpy(self):
        """Return NumPy-compatible Generator instance (if available)."""
        if self._use_numpy:
            return self._rng
        if np is None:
            raise RuntimeError("NumPy not available.")
        return np.random.default_rng(int(time.time()))

    def __repr__(self) -> str:
        backend = "numpy" if self._use_numpy else "stdlib"
        return f"<RNG backend={backend} pid={os.getpid()} id={id(self)}>"

# =============================================================================
# GLOBAL & THREAD-LOCAL ACCESSORS
# =============================================================================
_global_rng = RNG()
_thread_local = threading.local()


def get_rng(thread_safe: bool = False, use_numpy: bool = False) -> RNG:
    """Return an RNG instance (shared or per-thread)."""
    if thread_safe:
        if not hasattr(_thread_local, "rng"):
            _thread_local.rng = RNG(use_numpy=use_numpy)
        return _thread_local.rng
    return _global_rng


def set_global_seed(seed: int) -> None:
    """Re-seed the global RNG (and NumPy RNG if available)."""
    global _global_rng
    _global_rng.seed(seed)
    if np is not None:
        np.random.seed(seed)
