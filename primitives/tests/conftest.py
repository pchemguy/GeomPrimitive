"""
-------
conftest.py
-------
Shared pytest fixtures for primitive tests.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure headless backend for CI and multiprocessing
import matplotlib.pyplot as plt

from spt.spt_base import ImageBGR


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
# spt fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def gray_image() -> ImageBGR:
  """Return a small uniform 100x100 mid-gray BGR image."""
  return np.full((100, 100, 3), 128, dtype=np.uint8)

@pytest.fixture
def radial_image() -> ImageBGR:
    """
    128x128 radial ramp: dark in center, bright at edges.
    Same value in all BGR channels.
    """
    h = w = 128
    y, x = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / (r.max() + 1e-9)
    img = (r_norm * 255.0).astype(np.uint8)
    img3 = np.stack([img] * 3, axis=-1)
    return img3

@pytest.fixture
def gradient_image() -> ImageBGR:
    """Synthetic radial brightness gradient image."""
    h = w = 128
    y, x = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = (r / (r.max() + 1e-9)).astype(np.float32)
    img = (255 * (1 - r_norm)).astype(np.uint8)  # bright center, dark corners
    img3 = np.stack([img] * 3, axis=-1)
    return img3

