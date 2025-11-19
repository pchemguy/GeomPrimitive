"""
test_spt_color.py
-----------------
Unit tests for spt_vignette_and_color().
"""

import numpy as np
import pytest
from spt.spt_color import spt_vignette_and_color
from spt.mpl_utils import ImageBGR


# ---------------------------------------------------------------------------
# 1. Identity and safety
# ---------------------------------------------------------------------------

def test_identity_when_all_zero(gray_image):
    """Zero strengths must return the same image."""
    out = spt_vignette_and_color(gray_image, 0.0, 0.0)
    assert np.array_equal(out, gray_image)


def test_shape_and_dtype(gray_image):
    out = spt_vignette_and_color(gray_image, 0.3, 0.2)
    assert out.shape == gray_image.shape
    assert out.dtype == np.uint8


def test_output_range(gray_image):
    out = spt_vignette_and_color(gray_image, 1.0, 0.5)
    assert np.all((out >= 0) & (out <= 255))


# ---------------------------------------------------------------------------
# 2. Vignette behavior
# ---------------------------------------------------------------------------

def test_vignette_darkens_corners(gradient_image):
    """
    For an image bright in center and dark at edges,
    vignette_strength>0 should darken edges further.
    """
    base = gradient_image
    out = spt_vignette_and_color(base, vignette_strength=0.4, warm_strength=0.0)

    h, w = base.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask_center = r < 0.2 * r.max()
    mask_border = r > 0.7 * r.max()

    center_mean_base = base[mask_center].mean()
    border_mean_base = base[mask_border].mean()
    border_mean_out = out[mask_border].mean()

    # border_mean_out should be darker than before
    assert border_mean_out < border_mean_base
    # center brightness roughly unchanged
    assert abs(out[mask_center].mean() - center_mean_base) < 5.0


# ---------------------------------------------------------------------------
# 3. Warmth behavior
# ---------------------------------------------------------------------------

def test_warm_strength_increases_red_decreases_blue(gray_image):
    """
    On neutral gray input, warm_strength>0 should raise red and lower blue.
    """
    out = spt_vignette_and_color(gray_image, vignette_strength=0.0, warm_strength=0.5)
    mean_b, mean_g, mean_r = out.mean(axis=(0, 1))
    assert mean_r > mean_g
    assert mean_b < mean_g


def test_warm_strength_zero_preserves_balance(gray_image):
    """Without warm strength, channels remain equal."""
    out = spt_vignette_and_color(gray_image, vignette_strength=0.4, warm_strength=0.0)
    m = out.mean(axis=(0, 1))
    assert np.allclose(m[0], m[1], atol=1)
    assert np.allclose(m[1], m[2], atol=1)


# ---------------------------------------------------------------------------
# 4. Numerical sanity
# ---------------------------------------------------------------------------

def test_no_nans_or_infs(gray_image):
    """No NaN or Inf values should appear."""
    out = spt_vignette_and_color(gray_image, vignette_strength=1.0, warm_strength=0.5)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
