import math
import random

import numpy as np
import pytest
from matplotlib.collections import LineCollection

from mpl_grid_utils import GridJitterConfig, generate_grid_collections


def test_jitter_config_validation_types_and_signs() -> None:
  # Valid
  cfg = GridJitterConfig(
    global_angle_deg=6,
    line_angle_deg=3.0,
    line_offset_factor=0.4,
    line_offset_fraction=0.25,
    drop_fraction=0.05,
  )
  assert isinstance(cfg.global_angle_deg, float)
  assert cfg.global_angle_deg == 6.0

  # Negative value should raise ValueError
  with pytest.raises(ValueError):
    GridJitterConfig(global_angle_deg=-1.0)

  # Non-real should raise TypeError
  with pytest.raises(TypeError):
    GridJitterConfig(global_angle_deg="oops")  # type: ignore[arg-type]


def test_jitter_config_sampling_ranges() -> None:
  rng = random.Random(42)
  cfg = GridJitterConfig(
    global_angle_deg=6.0,
    line_angle_deg=3.0,
    line_offset_factor=0.4,
    line_offset_fraction=0.25,
    drop_fraction=0.05,
  )

  # Angle jitters must be within [-K, +K]
  for _ in range(100):
    g = cfg.jitter_angle_global(rng)
    assert -6.0 <= g <= 6.0

    la = cfg.jitter_line_angle(rng)
    assert -3.0 <= la <= 3.0

    off = cfg.jitter_line_offset(rng)
    assert -0.4 <= off <= 0.4

  # Fractions must be within [0, K]
  for _ in range(100):
    jf = cfg.jitter_offset_fraction(rng)
    assert 0.0 <= jf <= 0.25

    df = cfg.jitter_drop_fraction(rng)
    assert 0.0 <= df <= 0.05


def test_generate_grid_collections_orthogonal_no_jitter() -> None:
  bbox = (-10.0, -10.0, 10.0, 10.0)

  x_major, x_minor, y_major, y_minor = generate_grid_collections(
    bbox=bbox,
    obliquity_deg=90.0,
    rotation_deg=0.0,
    x_major_step=5.0,
    x_minor_step=2.5,
    y_major_step=5.0,
    y_minor_step=2.5,
    jitter=None,
    rng=random.Random(123),
  )

  # Basic type checks
  assert isinstance(x_major, LineCollection)
  assert isinstance(x_minor, LineCollection)
  assert isinstance(y_major, LineCollection)
  assert isinstance(y_minor, LineCollection)

  # For an orthogonal grid over [-10, 10] with step = 5,
  # indices are k in [-3, 3] -> 7 lines in each family.
  assert len(x_major.get_segments()) == 7
  assert len(y_major.get_segments()) == 7

  # For step = 2.5, indices are k in [-5, 5] -> 11 lines.
  assert len(x_minor.get_segments()) == 11
  assert len(y_minor.get_segments()) == 11

  # All points should lie inside (or extremely close to) the bbox
  for lc in (x_major, x_minor, y_major, y_minor):
    segs = np.asarray(lc.get_segments())
    xs = segs[..., 0]
    ys = segs[..., 1]
    assert xs.min() >= -10.0001
    assert xs.max() <= 10.0001
    assert ys.min() >= -10.0001
    assert ys.max() <= 10.0001


def test_generate_grid_collections_zero_jitter_equals_none() -> None:
  bbox = (-10.0, -10.0, 10.0, 10.0)

  zero_jitter = GridJitterConfig(
    global_angle_deg=0.0,
    line_angle_deg=0.0,
    line_offset_factor=0.0,
    line_offset_fraction=0.0,
    drop_fraction=0.0,
  )

  rng1 = random.Random(1)
  rng2 = random.Random(1)

  xM0, xm0, yM0, ym0 = generate_grid_collections(
    bbox=bbox,
    obliquity_deg=90.0,
    rotation_deg=0.0,
    x_major_step=5.0,
    x_minor_step=2.5,
    y_major_step=5.0,
    y_minor_step=2.5,
    jitter=None,
    rng=rng1,
  )

  xM1, xm1, yM1, ym1 = generate_grid_collections(
    bbox=bbox,
    obliquity_deg=90.0,
    rotation_deg=0.0,
    x_major_step=5.0,
    x_minor_step=2.5,
    y_major_step=5.0,
    y_minor_step=2.5,
    jitter=zero_jitter,
    rng=rng2,
  )

  # With zero jitter, geometry should be identical
  for lc0, lc1 in ((xM0, xM1), (xm0, xm1), (yM0, yM1), (ym0, ym1)):
    seg0 = np.asarray(lc0.get_segments())
    seg1 = np.asarray(lc1.get_segments())
    assert seg0.shape == seg1.shape
    np.testing.assert_allclose(seg0, seg1, rtol=0.0, atol=1e-9)
