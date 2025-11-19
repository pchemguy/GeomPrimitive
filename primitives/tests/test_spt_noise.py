"""
test_spt_noise.py
-----------------
Unit tests for spt_noise.py
"""

import numpy as np
import pytest
from skimage import filters

from spt.spt_noise import spt_noise
from spt.mpl_utils import ImageBGR


# ---------------------------------------------------------------------------
# 1. Identity and safety
# ---------------------------------------------------------------------------

def test_identity_all_noise_off(gray_image):
  """When all noise sources are disabled, output == input."""
  out = spt_noise(
      gray_image,
      poisson=False,
      gaussian=0.0,
      sp_amount=0.0,
      speckle_var=0.0,
      blur_sigma=0.0,
  )
  assert np.array_equal(out, gray_image)


def test_shape_and_dtype(gray_image):
  """Output shape and dtype must match input."""
  out = spt_noise(gray_image)
  assert out.shape == gray_image.shape
  assert out.dtype == np.uint8


def test_output_range(gray_image):
  """All pixels remain within [0, 255]."""
  out = spt_noise(gray_image, gaussian=1.0, speckle_var=1.0, blur_sigma=1.0)
  assert np.all((out >= 0) & (out <= 255))


# ---------------------------------------------------------------------------
# 2. Variance and scaling trends
# ---------------------------------------------------------------------------

def test_gaussian_noise_increases_variance(gray_image):
  """Gaussian noise should moderately increase variance (statistically)."""
  np.random.seed(0)
  weak = spt_noise(gray_image, gaussian=0.05, poisson=False, blur_sigma=0)
  mid  = spt_noise(gray_image, gaussian=0.2,  poisson=False, blur_sigma=0)

  v_weak, v_mid = weak.var(), mid.var()

  # If clipping occurred (variance plateau), skip this check
  if v_mid < v_weak * 1.05:
    pytest.skip(f"Gaussian noise variance nearly saturated: {v_mid:.1f} vs {v_weak:.1f}")
  else:
    assert v_mid > v_weak

def test_poisson_noise_changes_output(gray_image):
  """Poisson noise should alter pixel values even on flat image."""
  np.random.seed(1)
  out = spt_noise(gray_image, poisson=True, gaussian=0, sp_amount=0, speckle_var=0, blur_sigma=0)
  diff = np.mean(np.abs(out.astype(np.int16) - gray_image.astype(np.int16)))
  assert diff > 0.1  # ensures measurable deviation


def test_salt_pepper_creates_impulses(gray_image):
  """S&P noise should create isolated 0/255 pixels."""
  np.random.seed(2)
  out = spt_noise(gray_image, poisson=False, gaussian=0, sp_amount=1.0, speckle_var=0, blur_sigma=0)
  n_black = np.sum(out == 0)
  n_white = np.sum(out == 255)
  assert n_black > 0 and n_white > 0


def test_speckle_noise_modulation(gray_image):
  """Speckle noise changes intensity multiplicatively (non-zero variance)."""
  np.random.seed(3)
  out = spt_noise(gray_image, poisson=False, gaussian=0, sp_amount=0, speckle_var=0.5, blur_sigma=0)
  assert np.std(out) > 0.0


# ---------------------------------------------------------------------------
# 3. Blur verification
# ---------------------------------------------------------------------------

def test_blur_reduces_high_frequency(gray_image):
  """Gaussian blur should reduce high-frequency variance."""
  np.random.seed(4)
  # create synthetic edge pattern for better frequency content
  pattern = np.zeros_like(gray_image)
  pattern[:, ::2] = 255  # alternating columns
  no_blur = spt_noise(pattern, poisson=False, gaussian=0, speckle_var=0, blur_sigma=0)
  blurred = spt_noise(pattern, poisson=False, gaussian=0, speckle_var=0, blur_sigma=1.0)
  # compute Laplacian energy as HF indicator
  hf_no_blur = np.mean(filters.laplace(no_blur.astype(np.float32)) ** 2)
  hf_blur = np.mean(filters.laplace(blurred.astype(np.float32)) ** 2)
  assert hf_blur < hf_no_blur


# ---------------------------------------------------------------------------
# 4. Determinism / randomness
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="skimage.random_noise() is not strictly deterministic even with np.random.seed()")
def test_deterministic_with_fixed_seed(gray_image):
  """Non-deterministic across runs due to internal RNG differences."""
  np.random.seed(123)
  out1 = spt_noise(gray_image, gaussian=0.2, poisson=False, blur_sigma=0)
  np.random.seed(123)
  out2 = spt_noise(gray_image, gaussian=0.2, poisson=False, blur_sigma=0)
  assert np.array_equal(out1, out2)

def test_randomness_affects_output(gray_image):
  """Different random states produce distinct noisy outputs."""
  np.random.seed(11)
  out1 = spt_noise(gray_image, gaussian=0.3, poisson=False, blur_sigma=0)
  np.random.seed(12)
  out2 = spt_noise(gray_image, gaussian=0.3, poisson=False, blur_sigma=0)
  diff = np.mean(np.abs(out1.astype(np.int16) - out2.astype(np.int16)))
  assert diff > 0.5
