import pytest
import matplotlib
matplotlib.use("Agg")  # ensure headless mode
import matplotlib.pyplot as plt

from primitives.base import Primitive
from primitives.line import Line


@pytest.fixture(scope="function")
def fig_ax():
  """Return a Matplotlib Figure and Axes for isolated tests."""
  fig, ax = plt.subplots(figsize=(4, 3))
  yield fig, ax
  plt.close(fig)


@pytest.fixture(scope="function")
def line_instance():
  """Return a fresh Line instance with default RNG."""
  return Line()


@pytest.fixture(autouse=True)
def reset_rng():
  """Ensure deterministic RNG before each test."""
  Line.reseed(42)
  yield
  Line.reseed(42)
