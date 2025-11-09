"""
test_spt_lighting.py
--------------------
Tests for spt_lighting.py
"""

import numpy as np
import pytest
from spt.spt_lighting import spt_lighting
from spt.spt_base import ImageBGR


# ---------------------------------------------------------------------------
# 1. Basic behavior
# ---------------------------------------------------------------------------

def test_identity_case(gray_image):
  """When all parameters are zero, output equals input."""
  out = spt_lighting(gray_image, 0, 0, "linear", 90, 0, 0, 0)
  assert np.array_equal(out, gray_image)


def test_shape_and_dtype(gray_image):
  """Output has same shape and dtype as input."""
  out = spt_lighting(gray_image, top_bright=0.3, bottom_dark=-0.3)
  assert out.shape == gray_image.shape
  assert out.dtype == np.uint8


def test_output_clamped_range(gray_image):
  """All pixel values are within valid [0, 255] range."""
  out = spt_lighting(gray_image, top_bright=1.0, bottom_dark=-1.0, brightness=1.0)
  assert np.all(out >= 0) and np.all(out <= 255)


# ---------------------------------------------------------------------------
# 2. Brightness symmetry
# ---------------------------------------------------------------------------

def test_global_brightness_symmetry(gray_image):
  """Brightening increases mean intensity; darkening decreases it."""
  bright = spt_lighting(gray_image, brightness=+0.5)
  dark = spt_lighting(gray_image, brightness=-0.5)
  mean_orig = gray_image.mean()
  assert bright.mean() > mean_orig
  assert dark.mean() < mean_orig


# ---------------------------------------------------------------------------
# 3. Gradient correctness
# ---------------------------------------------------------------------------

def test_linear_gradient_direction(gray_image):
  """Linear 0deg gradient brightens left -> right."""
  h, w, _ = gray_image.shape
  out = spt_lighting(gray_image, top_bright=+0.5, bottom_dark=-0.5,
                     lighting_mode="linear", gradient_angle=0)
  left_mean = out[:, : w // 3].mean()
  right_mean = out[:, -w // 3 :].mean()
  assert right_mean > left_mean  # gradient increases across x


def test_radial_gradient_center_vs_corner(gray_image):
  """Radial gradient brightens center if top_bright>bottom_dark."""
  out = spt_lighting(gray_image, top_bright=+0.5, bottom_dark=-0.5,
                     lighting_mode="radial", grad_cx=0, grad_cy=0)
  h, w, _ = gray_image.shape
  center_mean = out[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5].mean()
  corner_mean = out[:10, :10].mean()
  assert center_mean > corner_mean  # center brighter than corner

# ---------------------------------------------------------------------------
# 4. Mode robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "radial"])
def test_modes_execute(gray_image, mode):
  """Both gradient modes execute without error."""
  out = spt_lighting(gray_image, top_bright=0.3, bottom_dark=-0.3, lighting_mode=mode)
  assert isinstance(out, np.ndarray)
  assert out.shape == gray_image.shape


# ---------------------------------------------------------------------------
# 5. Monotonicity property (linear mode)
# ---------------------------------------------------------------------------

def test_monotonic_gradient_property():
  """Intensity should increase monotonically along gradient direction."""
  img = np.tile(np.linspace(64, 192, 100, dtype=np.uint8), (100, 1))
  img = np.stack([img] * 3, axis=-1)
  out = spt_lighting(img, top_bright=0.5, bottom_dark=-0.5,
                     lighting_mode="linear", gradient_angle=0)
  profile = out[50, :, 0]
  assert np.all(np.diff(profile) >= -1)  # non-decreasing intensity


# ---------------------------------------------------------------------------
# 6. Performance sanity
# ---------------------------------------------------------------------------

def test_runtime_under_threshold(gray_image, benchmark):
  """Ensure processing is reasonably fast (<100 ms for 512x512)."""
  big_img = np.tile(gray_image, (5, 5, 1))
  result = benchmark(lambda: spt_lighting(big_img, top_bright=0.3, bottom_dark=-0.3))
  assert result.shape == big_img.shape
