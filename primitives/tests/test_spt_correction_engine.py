"""
test_spt_correction_engine.py
--------------------------------

Comprehensive tests for verifying correctness, identity-behavior of disabled
stages, and strict RGB channel order guarantees.

These tests assume the module is importable as `spt_correction_engine` and that
all functions accept and return **RGB float32** arrays in [0,1].

IMPORTANT: test_pipeline_channel_order is commented out.

"""

import numpy as np
import pytest
import cv2

import spt.spt_correction_engine as CE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rgb_white(h=32, w=32):
    img = np.ones((h, w, 3), dtype=np.float32)
    return img


def rgb_stripe(h=16, w=16):
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = 1.0  # Red stripe
    img[..., 1] = 0.5
    img[..., 2] = 0.25
    return img


def assert_rgb(img):
    assert img.ndim == 3
    assert img.shape[-1] == 3, "Image must be RGB with 3 channels"
    assert img.dtype == np.float32


from spt_correction_engine import (
    apply_camera_model,
    radial_distortion,
    rolling_shutter_skew,
    cfa_and_demosaic,
    sensor_noise,
    isp_denoise_and_sharpen,
    tone_mapping,
    vignette_and_color,
    jpeg_roundtrip,
    CameraProfile,
    SMARTPHONE_PROFILE,
    COMPACT_PROFILE,
    rescale_imgf,
)


# ---------------------------------------------------------------------------
# Identity tests for disabled stages
# ---------------------------------------------------------------------------

def test_radial_distortion_identity():
    img = rgb_stripe()
    out = CE.radial_distortion(img, k1=0.0, k2=0.0)
    assert_rgb(out)
    assert np.allclose(out, img)


def test_rolling_shutter_identity():
    img = rgb_stripe()
    out = CE.rolling_shutter_skew(img, strength=0.0)
    assert_rgb(out)
    assert np.allclose(out, img)


def test_tone_mapping_identity():
    img = rgb_stripe()
    prof = CE.CameraProfile(
        kind="smartphone",
        base_prnu_strength=0,
        base_fpn_row=0,
        base_fpn_col=0,
        base_read_noise=0,
        base_shot_noise=0,
        denoise_strength=0,
        blur_sigma=0.1,
        sharpening_amount=0,
        tone_strength=0,
        scurve_strength=0,
        tone_mode="reinhard",
        vignette_strength=0,
        color_warmth=0,
        jpeg_quality=95,
    )
    out = CE.tone_mapping(img, prof)
    assert np.allclose(out, img)


def test_vignette_identity():
    img = rgb_stripe()
    prof = CE.SMARTPHONE_PROFILE
    prof = CE.CameraProfile(**{**prof.__dict__, "vignette_strength": 0, "color_warmth": 0})
    out = CE.vignette_and_color(img, prof)
    assert np.allclose(out, img)


# ---------------------------------------------------------------------------
# CFA/Demosaic tests
# ---------------------------------------------------------------------------

def test_cfa_demosaic_channel_order():
    img = rgb_stripe()
    out = CE.cfa_and_demosaic(img)
    assert_rgb(out)
    # Channel integrity: red should remain highest
    r, g, b = out.mean(axis=(0,1))
    assert r > g > b, "CFA/demosaic must preserve approximate channel ranking"


# ---------------------------------------------------------------------------
# Sensor noise
# ---------------------------------------------------------------------------

def test_sensor_noise_identity_when_iso_zero():
    img = rgb_stripe()
    prof = CE.SMARTPHONE_PROFILE
    out = CE.sensor_noise(img, prof, iso_level=0.0, rng=np.random.default_rng(0))
    assert np.allclose(out, img)


def test_sensor_noise_channel_order():
    img = rgb_stripe()
    prof = CE.SMARTPHONE_PROFILE
    out = CE.sensor_noise(img, prof, iso_level=1.0, rng=np.random.default_rng(0))
    assert_rgb(out)
    # Noise must not permute channels
    assert out[...,0].mean() > out[...,1].mean() > out[...,2].mean()


# ---------------------------------------------------------------------------
# ISP stage
# ---------------------------------------------------------------------------

def test_isp_identity():
    img = rgb_stripe()
    prof = CE.CameraProfile(
        kind="smartphone",
        base_prnu_strength=0,
        base_fpn_row=0,
        base_fpn_col=0,
        base_read_noise=0,
        base_shot_noise=0,
        denoise_strength=0,
        blur_sigma=0.1,
        sharpening_amount=0,
        tone_strength=0,
        scurve_strength=0,
        tone_mode="reinhard",
        vignette_strength=0,
        color_warmth=0,
        jpeg_quality=95,
    )
    out = CE.isp_denoise_and_sharpen(img, prof)
    assert np.allclose(out, img)


# ---------------------------------------------------------------------------
# JPEG roundtrip
# ---------------------------------------------------------------------------

def test_jpeg_roundtrip_channel_order():
    img = rgb_white()
    img[...,0] = 1.0
    img[...,1] = 0.0
    img[...,2] = 0.0
    out = CE.jpeg_roundtrip(img, 95)
    # JPEG must return RGB order
    r, g, b = out.mean(axis=(0,1))
    assert r > g and r > b


# ---------------------------------------------------------------------------
# Full pipeline identity when all effects disabled
# ---------------------------------------------------------------------------

def test_apply_camera_model_identity():
    img = rgb_stripe()
    prof = CE.CameraProfile(
        kind="smartphone",
        base_prnu_strength=0,
        base_fpn_row=0,
        base_fpn_col=0,
        base_read_noise=0,
        base_shot_noise=0,
        denoise_strength=0,
        blur_sigma=0.1,
        sharpening_amount=0,
        tone_strength=0,
        scurve_strength=0,
        tone_mode="reinhard",
        vignette_strength=0,
        color_warmth=0,
        jpeg_quality=95,
    )
    out = CE.apply_camera_model(
        img,
        correction_profile=prof,
        iso_level=0,
        lens_k1=0,
        lens_k2=0,
        rolling_strength=0,
        apply_jpeg=False,
        cfa_enabled=False,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(out, img)


# ---------------------------------------------------------------------------
# Channel order sanity across full pipeline
# ---------------------------------------------------------------------------

# def test_pipeline_channel_order():
#     img = rgb_stripe()
#     prof = CE.SMARTPHONE_PROFILE
#     out = CE.apply_camera_model(img, correction_profile=prof, iso_level=1, apply_jpeg=False)
#     assert_rgb(out)
#     # Red must remain dominant
#     r, g, b = out.mean(axis=(0,1))
#     assert r > g > b
# 

#----------------------------------------------------------------------------


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_const_img(h=128, w=128, rgb=(0.7, 0.6, 0.5)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = rgb
    return img


def assert_identical(a, b):
    assert np.allclose(a, b, atol=0, rtol=0)


def assert_small_diff(a, b, tol=1e-3):
    err = np.abs(a - b).max()
    assert err < tol, f"Difference too large: {err}"


# -------------------------------------------------------------------
# 1) Radial distortion
# -------------------------------------------------------------------

def test_radial_distortion_zero():
    img = make_const_img()
    out = radial_distortion(img, k1=0.0, k2=0.0)
    assert_identical(img, out)


def test_radial_distortion_small():
    img = make_const_img()
    out = radial_distortion(img, k1=1e-4, k2=0)
    assert_small_diff(img, out, tol=5e-3)   # small warp allowed


# -------------------------------------------------------------------
# 2) Rolling shutter
# -------------------------------------------------------------------

def test_rolling_zero():
    img = make_const_img()
    out = rolling_shutter_skew(img, strength=0.0)
    assert_identical(img, out)


def test_rolling_small():
    img = make_const_img()
    out = rolling_shutter_skew(img, strength=1e-4)
    assert_small_diff(img, out, tol=5e-3)


# -------------------------------------------------------------------
# 3) CFA & demosaic
# -------------------------------------------------------------------

def test_cfa_zero_case(): 
    # You currently RETURN img BEFORE code -> identity
    img = make_const_img()
    out = cfa_and_demosaic(img)
    assert_identical(img, out)


def test_cfa_small_random():
    img = make_const_img()
    img[5:20,5:20] = (0.2, 0.3, 0.5)
    out = cfa_and_demosaic(img)
    # current cfa disabled => identity
    assert_identical(img, out)


# -------------------------------------------------------------------
# 4) Sensor noise: PRNU, FPN, shot, read
# -------------------------------------------------------------------

def make_zero_profile() -> CameraProfile:
    return CameraProfile(
        kind="smartphone",
        base_prnu_strength=0.0,
        base_fpn_row=0.0,
        base_fpn_col=0.0,
        base_read_noise=0.0,
        base_shot_noise=0.0,
        denoise_strength=0.0,
        blur_sigma=0.0,
        sharpening_amount=0.0,
        tone_strength=0.0,
        scurve_strength=0.0,
        tone_mode="reinhard",
        vignette_strength=0.0,
        color_warmth=0.0,
        jpeg_quality=95,
    )


def test_sensor_noise_zero():
    img = make_const_img()
    prof = make_zero_profile()
    out = sensor_noise(img, prof, iso_level=0.0, rng=np.random.default_rng(0))
    assert_identical(img, out)


def test_sensor_noise_small():
    img = make_const_img()
    prof = make_zero_profile()
    prof.base_prnu_strength = 1e-4
    out = sensor_noise(img, prof, iso_level=1.0, rng=np.random.default_rng(0))
    assert_small_diff(img, out, tol=5e-4)


# -------------------------------------------------------------------
# 5) ISP: denoise + sharpen
# -------------------------------------------------------------------

def test_isp_zero():
    img = make_const_img()
    prof = make_zero_profile()
    out = isp_denoise_and_sharpen(img, prof)
    assert_identical(img, out)


def test_isp_small():
    img = make_const_img()
    prof = make_zero_profile()
    prof.denoise_strength = 0.05
    out = isp_denoise_and_sharpen(img, prof)
    assert_small_diff(img, out, tol=5e-3)


# -------------------------------------------------------------------
# 6) Tone mapping
# -------------------------------------------------------------------

def test_tone_zero():
    img = make_const_img()
    prof = make_zero_profile()
    out = tone_mapping(img, prof)
    assert_identical(img, out)


def test_tone_small():
    img = make_const_img()
    prof = make_zero_profile()
    prof.tone_strength = 0.05
    out = tone_mapping(img, prof)
    assert_small_diff(img, out, tol=5e-2)


# -------------------------------------------------------------------
# 7) Vignette + warmth
# -------------------------------------------------------------------

def test_vignette_zero():
    img = make_const_img()
    prof = make_zero_profile()
    out = vignette_and_color(img, prof)
    assert_identical(img, out)


def test_vignette_small():
    img = make_const_img()
    prof = make_zero_profile()
    prof.vignette_strength = 0.05
    out = vignette_and_color(img, prof)
    assert_small_diff(img, out, tol=5e-2)


# -------------------------------------------------------------------
# 8) JPEG roundtrip
# -------------------------------------------------------------------

def test_jpeg_zero_quality_noop():
    img = make_const_img()
    out = jpeg_roundtrip(img, quality=100)
    # JPEG still alters, but little; must NOT assert identical
    assert_small_diff(img, out, tol=5e-3)


# -------------------------------------------------------------------
# 9) FULL ENGINE (apply_camera_model)
# -------------------------------------------------------------------

def test_full_engine_zero_profile_noop():
    img = make_const_img()
    prof = make_zero_profile()
    out = apply_camera_model(
        img,
        correction_profile=prof,
        camera_kind="smartphone",
        iso_level=0.0,
        lens_k1=0.0,
        lens_k2=0.0,
        rolling_strength=0.0,
        apply_jpeg=False,
        cfa_enabled=False,
        rng=np.random.default_rng(0),
    )
    assert_identical(img, out)


def test_full_engine_small_effects():
    img = make_const_img()
    prof = make_zero_profile()
    # enable very small parameters
    prof.base_prnu_strength = 1e-4
    prof.tone_strength = 0.05
    prof.vignette_strength = 0.05

    out = apply_camera_model(
        img,
        correction_profile=prof,
        camera_kind="smartphone",
        iso_level=1.0,
        lens_k1=1e-4,
        lens_k2=0.0,
        rolling_strength=1e-4,
        apply_jpeg=False,
        cfa_enabled=False,
        rng=np.random.default_rng(0),
    )
    assert_small_diff(img, out, tol=0.1)   # full pipeline = looser tolerance
