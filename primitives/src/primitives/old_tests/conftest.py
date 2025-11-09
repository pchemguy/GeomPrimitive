"""
-------
conftest.py
-------
Shared pytest fixtures for primitive tests.
"""

import pytest
import matplotlib
matplotlib.use("Agg")  # ensure headless backend for CI and multiprocessing
import matplotlib.pyplot as plt

from primitives.line import Line


# -----------------------------------------------------------------------------
# Core Matplotlib fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="function")
def fig_ax():
  """
  Create and yield an isolated Matplotlib Figure/Axes pair.

  The figure is automatically closed after the test to avoid memory leaks.
  """
  fig, ax = plt.subplots(figsize=(4, 3))
  yield fig, ax
  plt.close(fig)


# -----------------------------------------------------------------------------
# Deterministic RNG setup
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_rng():
  """
  Automatically reseed the RNG before and after each test
  for deterministic, repeatable results.
  """
  Line.reseed(42)
  yield
  Line.reseed(42)


# -----------------------------------------------------------------------------
# Primitive fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="function")
def line_instance(fig_ax):
  """
  Return a freshly initialized Line instance
  attached to a dedicated test Axes.
  """
  _, ax = fig_ax
  return Line(ax)
