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
from utils.rng import RNG

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


@pytest.fixture(scope="module")
def std_rng():
    """Standard-library RNG backend."""
    return rng.RNG(seed=42, use_numpy=False)


@pytest.fixture(scope="module")
def np_rng():
    """NumPy RNG backend."""
    return rng.RNG(seed=42, use_numpy=True)


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


# =====================================================================
# UNIFORM
# =====================================================================
def test_uniform_scalar_stdlib(std_rng):
    """Stdlib RNG: always returns Python float."""
    val = std_rng.uniform(0, 1)
    assert isinstance(val, float)
    assert 0 <= val <= 1


def test_uniform_scalar_numpy(np_rng):
    """NumPy RNG: scalar call returns float (converted from 0-D ndarray)."""
    val = np_rng.uniform(0, 1)
    assert isinstance(val, float)
    assert 0 <= val <= 1


def test_uniform_array_numpy(np_rng):
    """NumPy RNG: vectorized call returns ndarray."""
    arr = np_rng.uniform(0, 1, size=(2, 3))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert np.all((arr >= 0) & (arr <= 1))


# ---------------------------------------------------------------------
# Tests: reproducibility
# ---------------------------------------------------------------------
def test_uniform_reproducible_between_instances():
    """Two RNGs with same seed and backend must yield same uniform scalar."""
    r1 = rng.RNG(seed=123, use_numpy=False)
    r2 = rng.RNG(seed=123, use_numpy=False)
    assert r1.uniform(0, 1) == r2.uniform(0, 1)

    n1 = rng.RNG(seed=123, use_numpy=True)
    n2 = rng.RNG(seed=123, use_numpy=True)
    np.testing.assert_allclose(n1.uniform(0, 1, size=4), n2.uniform(0, 1, size=4))


# ---------------------------------------------------------------------
# Tests: shape edge cases
# ---------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1,), (2, 1), (1, 3), (4, 4)])
def test_uniform_array_shapes(np_rng, shape):
    """Ensure arbitrary shapes are supported."""
    arr = np_rng.uniform(-5, 5, size=shape)
    assert arr.shape == shape
    assert np.all((arr >= -5) & (arr <= 5))


# =====================================================================
# NORMAL
# =====================================================================
def test_normal_scalar_stdlib(std_rng):
    """Stdlib RNG: returns Python float."""
    val = std_rng.normal()
    assert isinstance(val, float)
    assert -5 < val < 5  # sanity bound


def test_normal_scalar_numpy(np_rng):
    """NumPy RNG: scalar call returns float."""
    val = np_rng.normal()
    assert isinstance(val, float)
    assert -5 < val < 5


def test_normal_array_numpy(np_rng):
    """NumPy RNG: vectorized call returns ndarray."""
    arr = np_rng.normal(0, 1, size=(3, 3))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 3)


# =====================================================================
# RANDINT
# =====================================================================
def test_randint_scalar_stdlib(std_rng):
    """Stdlib RNG: returns Python int."""
    val = std_rng.randint(0, 10)
    assert isinstance(val, int)
    assert 0 <= val <= 10


def test_randint_scalar_numpy(np_rng):
    """NumPy RNG: returns Python int (converted from numpy scalar)."""
    val = np_rng.randint(0, 10)
    assert isinstance(val, int)
    assert 0 <= val <= 10


def test_randint_array_numpy(np_rng):
    """NumPy RNG: vectorized call."""
    arr = np_rng.randint(0, 10, size=(4, 2))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 2)
    assert np.all((arr >= 0) & (arr <= 10))


# =====================================================================
# CROSS-BACKEND REPRODUCIBILITY & PARITY
# =====================================================================
def test_same_seed_produces_same_results():
    """RNG instances with identical seeds produce identical outputs."""
    # Stdlib
    r1, r2 = RNG(seed=123, use_numpy=False), RNG(seed=123, use_numpy=False)
    assert r1.uniform(0, 1) == r2.uniform(0, 1)
    assert r1.randint(0, 10) == r2.randint(0, 10)

    # NumPy
    n1, n2 = RNG(seed=123, use_numpy=True), RNG(seed=123, use_numpy=True)
    np.testing.assert_allclose(n1.uniform(0, 1, size=4), n2.uniform(0, 1, size=4))
    np.testing.assert_allclose(n1.normal(0, 1, size=4), n2.normal(0, 1, size=4))


# =====================================================================
# SHAPE CONSISTENCY & BOUNDS
# =====================================================================
@pytest.mark.parametrize("shape", [(1,), (2, 2), (3, 1, 2)])
def test_uniform_and_normal_shapes(np_rng, shape):
    """Ensure shape argument works for both uniform and normal calls."""
    u = np_rng._rng.uniform(-2, 2, size=shape)
    n = np_rng._rng.normal(0, 1, size=shape)
    assert u.shape == shape
    assert n.shape == shape
    assert np.all((u >= -2) & (u <= 2))

