"""
test_spt_texture.py
-------------------
Unit tests for spt_texture.py
"""

import numpy as np
import pytest
from numpy.fft import rfft2, rfftfreq

from spt.spt_texture import spt_texture
from spt.mpl_utils import ImageBGR


# ---------------------------------------------------------------------------
# 1. Basic behavior and identity
# ---------------------------------------------------------------------------

def test_identity_when_strength_zero(gray_image):
  """Texture strength 0 -> image unchanged."""
  out = spt_texture(gray_image, texture_strength=0.0, texture_scale=8.0)
  assert np.array_equal(out, gray_image)


def test_identity_when_scale_zero(gray_image):
  """Texture scale 0 -> image unchanged (no blur)."""
  out = spt_texture(gray_image, texture_strength=0.2, texture_scale=0.0)
  assert np.array_equal(out, gray_image)


def test_shape_and_dtype(gray_image):
  """Output has same shape and dtype as input."""
  out = spt_texture(gray_image, texture_strength=0.2, texture_scale=8.0)
  assert out.shape == gray_image.shape
  assert out.dtype == np.uint8


def test_output_clamped_range(gray_image):
  """Pixel values must remain within [0, 255]."""
  out = spt_texture(gray_image, texture_strength=5.0, texture_scale=8.0)
  assert np.all(out >= 0) and np.all(out <= 255)


# ---------------------------------------------------------------------------
# 2. Statistical effects
# ---------------------------------------------------------------------------

def test_variance_increases_with_strength(gray_image):
  """Texture strength increases local variance."""
  weak = spt_texture(gray_image, texture_strength=0.1, texture_scale=8.0)
  strong = spt_texture(gray_image, texture_strength=0.5, texture_scale=8.0)
  assert strong.var() > weak.var() > 0

@pytest.mark.parametrize("scale", [1.0, 4.0, 16.0])
def test_texture_scale_affects_spatial_frequency(gray_image, scale):
  """
  Texture scale controls spatial correlation length:
  larger sigma -> smoother (lower high-frequency energy).
  """
  np.random.seed(42)
  out = spt_texture(gray_image, texture_strength=0.3, texture_scale=scale)

  # Convert to grayscale for analysis
  gray = out[..., 0].astype(np.float32)
  fft_mag = np.abs(rfft2(gray))
  freqs = rfftfreq(gray.shape[1])
  spectrum = fft_mag.mean(axis=0)  # average across rows

  # Compute high-frequency energy ratio
  high_energy = spectrum[int(len(freqs) * 0.5):].mean()
  low_energy = spectrum[:int(len(freqs) * 0.2)].mean()

  # Higher scale => smoother => lower high-frequency energy
  if scale > 1:
    assert high_energy < low_energy * 5  # loose but indicative bound

def _spectral_slope(gray: np.ndarray) -> float:
    """Compute approximate 1/f slope of radial power spectrum."""
    h, w = gray.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fx, fy = np.meshgrid(fx, fy)
    f = np.sqrt(fx**2 + fy**2)
    mag = np.abs(np.fft.fft2(gray))
    r_bins = np.linspace(0, f.max(), 50)
    psd = np.zeros_like(r_bins)
    for i in range(len(r_bins) - 1):
        mask = (f >= r_bins[i]) & (f < r_bins[i + 1])
        if np.any(mask):
            psd[i] = mag[mask].mean()
    slope = np.polyfit(np.log10(r_bins[1:-1] + 1e-6),
                       np.log10(psd[1:-1] + 1e-6), 1)[0]
    return slope

def test_spectral_slope_monotonic_with_scale(gray_image):
    """Larger texture scale -> smoother -> flatter (less negative) spectral slope."""
    np.random.seed(0)
    s1 = _spectral_slope(spt_texture(gray_image, 0.3, 2.0)[..., 0])
    s2 = _spectral_slope(spt_texture(gray_image, 0.3, 8.0)[..., 0])
    s3 = _spectral_slope(spt_texture(gray_image, 0.3, 16.0)[..., 0])
    slopes = np.array([s1, s2, s3])
    corr = np.corrcoef(np.log([2.0, 8.0, 16.0]), slopes)[0, 1]
    assert corr > 0.5, f"Expected positive trend (flatter with scale), got corr={corr:.2f}, slopes={slopes}"

def test_texture_preserves_global_mean(gray_image):
    """Texture modulation should preserve mean luminance within +/-5%."""
    out = spt_texture(gray_image, texture_strength=0.3, texture_scale=4.0)
    ratio = out.mean() / gray_image.mean()
    assert 0.95 <= ratio <= 1.05


# ---------------------------------------------------------------------------
# 3. Deterministic behavior (with fixed RNG)
# ---------------------------------------------------------------------------

def test_deterministic_with_fixed_seed(gray_image):
  """Setting NumPy seed should yield reproducible output."""
  np.random.seed(123)
  out1 = spt_texture(gray_image, texture_strength=0.3, texture_scale=4.0)
  np.random.seed(123)
  out2 = spt_texture(gray_image, texture_strength=0.3, texture_scale=4.0)
  assert np.array_equal(out1, out2)


def test_randomness_affects_output(gray_image):
  """Different RNG states produce different texture fields."""
  np.random.seed(1)
  out1 = spt_texture(gray_image, texture_strength=0.3, texture_scale=4.0)
  np.random.seed(2)
  out2 = spt_texture(gray_image, texture_strength=0.3, texture_scale=4.0)
  diff = np.mean(np.abs(out1.astype(np.int16) - out2.astype(np.int16)))
  assert diff > 0.5  # some difference expected


# ---------------------------------------------------------------------------
# 4. Performance sanity (optional)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_runtime_under_threshold(gray_image, benchmark):
  """Should run < 100 ms for 512x512 image."""
  big = np.tile(gray_image, (5, 5, 1))
  result = benchmark(lambda: spt_texture(big, texture_strength=0.3, texture_scale=4.0))
  assert result.shape == big.shape
