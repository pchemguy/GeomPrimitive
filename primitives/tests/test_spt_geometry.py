"""
test_spt_geometry.py
--------------------
Unit tests for spt_geometry.py
"""

import numpy as np
import pytest

from spt.spt_geometry import spt_geometry
from spt.mpl_utils import ImageBGR


# ---------------------------------------------------------------------------
# 1. Identity and safety
# ---------------------------------------------------------------------------

def test_identity_all_zero_params(gray_image):
    """No tilt and no distortion should return the original image."""
    out = spt_geometry(
        gray_image,
        tilt_x=0.0,
        tilt_y=0.0,
        k1=0.0,
        k2=0.0,
    )
    assert np.array_equal(out, gray_image)


def test_shape_and_dtype(gray_image):
    """Output shape and dtype must match input."""
    out = spt_geometry(gray_image)
    assert out.shape == gray_image.shape
    assert out.dtype == np.uint8


def test_output_range(gray_image):
    """All pixels remain in valid [0, 255] range."""
    out = spt_geometry(gray_image, tilt_x=1.0, tilt_y=1.0, k1=1.0, k2=1.0)
    assert np.all((out >= 0) & (out <= 255))


# ---------------------------------------------------------------------------
# 2. Perspective tilt behavior
# ---------------------------------------------------------------------------

def test_tilt_changes_image(gray_image):
    """
    Non-zero tilt should produce a different image than identity.
    We don't assert exact geometry, just that it is not a no-op.
    """
    out = spt_geometry(gray_image, tilt_x=0.5, tilt_y=0.3, k1=0.0, k2=0.0)
    diff = np.mean(np.abs(out.astype(np.int16) - gray_image.astype(np.int16)))
    assert diff > 0.1  # some measurable change


def test_tilt_only_preserves_shape_dtype(gray_image):
    """Tilt only (no distortion) keeps size and dtype."""
    out = spt_geometry(gray_image, tilt_x=0.5, tilt_y=-0.5, k1=0.0, k2=0.0)
    assert out.shape == gray_image.shape
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. Radial distortion: barrel vs pincushion
# ---------------------------------------------------------------------------

def test_distortion_identity_when_k_zero(radial_image):
    """k1=k2=0 implies pure identity (no radial distortion)."""
    out = spt_geometry(radial_image, tilt_x=0.0, tilt_y=0.0, k1=0.0, k2=0.0)
    assert np.array_equal(out, radial_image)


def test_center_pixel_stable_under_distortion(radial_image):
    """Center pixel should remain nearly unchanged under radial distortion."""
    h, w = radial_image.shape[:2]
    cy, cx = h // 2, w // 2

    out = spt_geometry(radial_image, tilt_x=0.0, tilt_y=0.0, k1=0.5, k2=0.1)
    center_orig = radial_image[cy, cx].mean()
    center_out = out[cy, cx].mean()
    # Distortion centered around optical center should not wildly change center
    assert abs(center_orig - center_out) < 5.0


def test_barrel_vs_pincushion_on_radial_ramp(radial_image):
    """
    On a radial ramp image (darker center, brighter edges),
    barrel (k1<0) should make the edges brighter than pincushion (k1>0).

    Rationale:
      - Input I(r) increases with radius r.
      - Mapping uses r_u = r_d / factor where factor = 1 + k1 r_d^2 + k2 r_d^4.
      - For k1<0 -> factor<1 at edges -> r_u > r_d -> sample from larger radius -> brighter.
      - For k1>0 -> factor>1 at edges -> r_u < r_d -> sample from smaller radius -> darker.
    """
    h, w = radial_image.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max()

    mask_border = r > 0.7 * r_max  # outer ring

    barrel = spt_geometry(radial_image, tilt_x=0.0, tilt_y=0.0, k1=-0.5, k2=0.0)
    pinc  = spt_geometry(radial_image, tilt_x=0.0, tilt_y=0.0, k1=+0.5, k2=0.0)

    border_barrel_mean = barrel[mask_border].mean()
    border_pinc_mean   = pinc[mask_border].mean()

    # Edges should be brighter under barrel than under pincushion
    assert border_barrel_mean > border_pinc_mean


# ---------------------------------------------------------------------------
# 4. Numerical sanity
# ---------------------------------------------------------------------------

def test_no_nans_or_infs(gray_image):
    """Ensure no NaN or Inf values sneak into the output."""
    out = spt_geometry(gray_image, tilt_x=1.0, tilt_y=-1.0, k1=1.0, k2=-0.5)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
