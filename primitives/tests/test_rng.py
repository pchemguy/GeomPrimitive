"""
test_rng.py
-----------

Unit tests for rng.py (RNG and get_rng).
Covers deterministic behavior, thread safety, reseeding,
NumPy fallback, and API parity.
"""

import os
import time
import math
import threading
import pytest

import utils.rng as rng

# Optional import of NumPy (tests still pass without it)
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def seeded_rng():
    """Provide a deterministic RNG with fixed seed."""
    return rng.RNG(seed=123)


@pytest.fixture
def fixed_rng():
    """Provide a deterministic RNG with fixed seed."""
    return rng.RNG(seed=123)


@pytest.fixture
def fresh_rng():
    """Provide a new RNG seeded with PID/time entropy."""
    return rng.RNG()


# ---------------------------------------------------------------------
# 1. Basic construction and repr
# ---------------------------------------------------------------------
def test_rng_initialization_and_repr(seeded_rng):
    r = seeded_rng
    assert isinstance(r._rng, (rng.random.Random,)) or HAVE_NUMPY
    text = repr(r)
    assert "RNG" in text
    assert "pid" in text


def test_rng_reseed_changes_sequence(seeded_rng):
    """Reseeding should produce a different random sequence."""
    vals1 = [seeded_rng.random() for _ in range(5)]
    seeded_rng.seed(999)
    vals2 = [seeded_rng.random() for _ in range(5)]
    assert vals1 != vals2


# ---------------------------------------------------------------------
# 2. Determinism and reproducibility
# ---------------------------------------------------------------------
def test_reproducibility_fixed_seed():
    """Same seed => identical sequences."""
    r1 = rng.RNG(seed=42)
    r2 = rng.RNG(seed=42)
    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]
    assert seq1 == seq2


def test_pid_time_entropy_differs():
    """PID/time seeding should yield non-identical RNGs."""
    r1 = rng.RNG()
    time.sleep(0.002)
    r2 = rng.RNG()
    assert [r1.random() for _ in range(3)] != [r2.random() for _ in range(3)]


# ---------------------------------------------------------------------
# 3. Core methods work and within bounds
# ---------------------------------------------------------------------
def test_randint_and_uniform_bounds(seeded_rng):
    for _ in range(100):
        x = seeded_rng.randint(0, 10)
        y = seeded_rng.uniform(0, 1)
        assert 0 <= x <= 10
        assert 0.0 <= y <= 1.0


def test_choice_and_shuffle(seeded_rng):
    seq = list(range(10))
    val = seeded_rng.choice(seq)
    assert val in seq

    before = seq.copy()
    shuffled = seeded_rng.shuffle(seq.copy())
    assert sorted(shuffled) == sorted(before)
    assert isinstance(shuffled, list)


def test_normal_distribution_mean_variance(seeded_rng):
    """Sample mean and variance should roughly match parameters."""
    samples = [seeded_rng.normal(0, 1) for _ in range(5000)]
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    assert abs(mean) < 0.1
    assert 0.5 < var < 1.5


def test_normal3s_bounds_and_distribution(fixed_rng):
    vals = [fixed_rng.normal3s() for _ in range(10000)]
    arr = np.array(vals)
    assert np.all(arr >= -1) and np.all(arr <= 1)
    # Expect mean ~ 0 and std ~ 1/3 within tolerance
    assert abs(arr.mean()) < 0.05
    assert 0.25 < arr.std() < 0.38


# ---------------------------------------------------------------------
# 4. Thread-safety
# ---------------------------------------------------------------------
def test_thread_safety_parallel_invocation():
    """Concurrent access should not raise or produce identical results."""
    r = rng.RNG(seed=999)
    results = []

    def worker(idx):
        results.append((idx, r.randint(0, 1000)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    vals = [v for _, v in results]
    assert len(set(vals)) > 1  # not all identical


# ---------------------------------------------------------------------
# 5. Global and thread-local RNG behavior
# ---------------------------------------------------------------------
def test_get_rng_shared_and_threadlocal():
    """Thread-safe flag should isolate RNGs."""
    global_rng = rng.get_rng(thread_safe=False)
    t_rng_1 = rng.get_rng(thread_safe=True)
    t_rng_2 = rng.get_rng(thread_safe=True)
    # same thread should reuse same local rng
    assert t_rng_1 is t_rng_2
    # global is a different object
    assert global_rng is not t_rng_1


# ---------------------------------------------------------------------
# 6. NumPy integration (if available)
# ---------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not available")
def test_numpy_backend_random_and_uniform():
    r = rng.RNG(seed=123, use_numpy=True)
    x = r.random()
    y = r.uniform(0, 1)
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0


@pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not available")
def test_as_numpy_returns_generator():
    r = rng.RNG(use_numpy=True)
    g = r.as_numpy()
    import numpy as np
    assert isinstance(g, np.random.Generator)
    arr = g.integers(0, 10, size=5)
    assert len(arr) == 5


@pytest.mark.skipif(not HAVE_NUMPY, reason="NumPy not available")
def test_numpy_seed_reproducibility():
    r1 = rng.RNG(seed=2024, use_numpy=True)
    r2 = rng.RNG(seed=2024, use_numpy=True)
    seq1 = [r1.random() for _ in range(5)]
    seq2 = [r2.random() for _ in range(5)]
    assert seq1 == seq2


# ---------------------------------------------------------------------
# 7. State handling
# ---------------------------------------------------------------------
def test_getstate_and_setstate(seeded_rng):
    state = seeded_rng.getstate()
    vals1 = [seeded_rng.random() for _ in range(3)]
    seeded_rng.setstate(state)
    vals2 = [seeded_rng.random() for _ in range(3)]
    assert vals1 == vals2


# ---------------------------------------------------------------------
# 8. set_global_seed
# ---------------------------------------------------------------------
def test_set_global_seed_affects_rng(monkeypatch):
    import builtins
    called = {}

    # patch np.random.seed to detect call if NumPy available
    if HAVE_NUMPY:
        import numpy as np
        called["np_called"] = False
        old_seed = np.random.seed
        np.random.seed = lambda s: called.__setitem__("np_called", True)

    rng.set_global_seed(321)
    assert isinstance(rng._global_rng, rng.RNG)

    if HAVE_NUMPY:
        np.random.seed = old_seed
        assert called["np_called"]


# ---------------------------------------------------------------------
# 9. Performance sanity (quick smoke test)
# ---------------------------------------------------------------------
def test_perf_benchmark(benchmark):
    r = rng.RNG(seed=111)
    result = benchmark(lambda: [r.random() for _ in range(1000)])
    assert isinstance(result, list)
    assert len(result) == 1000
