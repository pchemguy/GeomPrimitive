"""
spt_correction_engine.py
------------------------

Ref: https://chatgpt.com/c/69172a6a-b78c-8326-b080-7b02e61b4730

Camera-Like Correction Engine (Internal RGB-Float ISP Simulator)
===============================================================

This module implements a self-contained, physically-motivated camera
simulation pipeline ("Correction Engine") designed to inject the kinds
of imperfections, artifacts, and statistics that are present in real
laboratory photographs acquired using smartphones or compact cameras.

The engine operates entirely in RGB float32 linear space [0, 1], which
matches the internal domain of real imaging pipelines. The output is
also RGB float32. Only the JPEG simulation temporarily converts to
uint8/BGR for encoding.

The goal is to ensure that synthetic images produced by rendering
systems (e.g., Matplotlib, synthetic textures, geometric overlays) will
exhibit realistic photographic traces - including optical distortions,
CFA/demosaicing artifacts, sensor noise, ISP processing, color
characteristics, tone mapping, and JPEG quantization - so that forensic
tools and ML pipelines cannot easily distinguish real lab photos from
synthetic data.

Pipeline Overview
-----------------

The Correction Engine consists of the following sequential stages:

1. Optical Lens Model
   - Radial distortion (barrel/pincushion)
   - Rolling-shutter geometric skew
   - (Extensible to chromatic aberration or PSF blurring)

2. CFA & Demosaicing Simulation
   - Convert RGB into virtual Bayer RAW (RGGB)
   - Apply bilinear / Bayer demosaicing
   - Introduces zippering, color moire, and channel-correlated artifacts

3. Sensor Pattern Noise (ISO-dependent)
   - PRNU: pixel-level multiplicative gain variations
   - FPN: row/column banding patterns
   - Proper ISO scaling factors

4. Shot Noise + Read Noise
   - Brightness-dependent Poisson-like noise
   - Additive Gaussian read noise
   - Correct linear-RGB distribution

5. ISP Denoising + Sharpening
   - Bilateral denoiser (approximates smartphone denoise)
   - Unsharp mask halo formation (edge overshoot/undershoot)

6. Tone Mapping
   - Global tone-mapping (Reinhard or filmic/Hable)
   - Optional S-curve local contrast "pop"

7. Vignetting & Color Tone
   - Optical falloff simulated via radial mask
   - Mild color warmth/bias (e.g. phone LED tint)

8. JPEG Simulation (optional)
   - Roundtrip through JPEG at configurable quality
   - Introduces DCT blocking, grid patterns, and quantization ghosts

Design Notes
------------

- All steps operate in linear RGB, not sRGB. Real photon noise, PRNU,
  and demosaicing must be applied before gamma/tone curves.

- The engine is parameterized via `CameraProfile`, allowing different
  "personalities": smartphone-like, compact-camera-like, noisy, clean,
  etc.

- Every stage can be disabled by setting its coefficients to zero
  (e.g. vignette_strength=0, tone_strength=0, base_prnu_strength=0,
  or iso_level=0).

- This module is intentionally modular: adding PSF blur, chromatic
  aberration, local tone mapping, or more sophisticated denoisers is
  straightforward.

"""

from __future__ import annotations

__all__ = ["apply_camera_model"]

import os
import sys
import time
from dataclasses import dataclass
from numbers import Real
import random
from types import MappingProxyType
from typing import Literal, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

import matplotlib as mpl
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Local imports (type aliases from your mpl_utils)
# ---------------------------------------------------------------------------

from utils.rng import RNG, get_rng
from utils.logging_utils import configure_logging

from mpl_utils import (
    # Conversion helpers
    bgr_from_rgba, rgb_from_bgr, rgbf_from_rgba, rgb_from_rgbf,
    # Rendering helpers
    show_RGBx_grid, render_scene,
    # Type aliases
    ImageBGR, ImageRGB, ImageRGBA, ImageRGBx,
    # Constants
    PAPER_COLORS,
)

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

EPSILON: float = 1e-8

ISOLevel = Literal["low", "mid", "high"]
ISO_SF: MappingProxyType = MappingProxyType({
    "low": 0.6,
    "mid": 1.0,
    "high": 1.6,
})

CameraKind = ["smartphone", "compact"]
ToneMode = ["reinhard", "filmic"]


# ---------------------------------------------------------------------------
# Ranges for all parameters (from the master table)
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "k1"                 : (-0.20, 0.20), # Norm 3*sigma
    "k2"                 : (-0.02, 0.02), # Norm 3*sigma
    "rolling_strength"   : (0.0, 0.03),   # Norm 3*sigma

    # Sensor noise
    "base_prnu_strength" : (0.0, 0.010),
    "base_fpn_row"       : (0.0, 0.010),
    "base_fpn_col"       : (0.0, 0.010),
    "base_read_noise"    : (0.0, 0.030),
    "base_shot_noise"    : (0.0, 0.008),

    # ISP
    "denoise_strength"   : (0.0, 1.0),
    "blur_sigma"         : (0.5, 3.0),
    "sharpening_amount"  : (0.0, 1.0),

    # Tone
    "tone_strength"      : (0.0, 0.80),
    "scurve_strength"    : (0.0, 0.50),

    # Vignette + color
    "vignette_strength"  : (0.0, 0.50),
    "color_warmth"       : (0.0, 0.20),

    # JPEG
    "jpeg_quality"     : (70, 98),

    # ISO
    "iso_sf"           : (0.5, 2.0),
}
    


@dataclass
class CameraProfile:
    """Camera-like behavior configuration.

    Attributes:
        kind: Camera category ('smartphone' or 'compact').
        base_prnu_strength: Base multiplicative per-pixel PRNU amplitude.
        base_fpn_row: Row-wise fixed-pattern noise amplitude.
        base_fpn_col: Column-wise fixed-pattern noise amplitude.
        base_read_noise: Additive read noise (std) in linear RGB domain.
        base_shot_noise: Base scale for brightness-dependent shot noise.
        denoise_strength: Scaling factor for strength and smoothness of denoising.
        blur_sigma: Unsharpen radius control.
        sharpening_amount: Unsharp-mask strength for ISP sharpening.
        tone_strength: Base tone-mapping strength (0=off, 1=strong compression).
        scurve_strength: S-curve contrast strength (0=off, 1=strong).
        tone_mode: Tone-mapping mode ('reinhard' or 'filmic').
        vignette_strength: Vignetting strength (center-to-edge).
        color_warmth: Warm bias, positive shifts towards warmer tones.
        jpeg_quality: JPEG quality for optional roundtrip (higher = less compressed).
    """

    kind              : str      = "smartphone"
    base_prnu_strength: float    = 0
    base_fpn_row      : float    = 0
    base_fpn_col      : float    = 0
    base_read_noise   : float    = 0
    base_shot_noise   : float    = 0
    denoise_strength  : float    = 0
    blur_sigma        : float    = 0
    sharpening_amount : float    = 0
    tone_strength     : float    = 0
    scurve_strength   : float    = 0
    tone_mode         : ToneMode = "filmic"
    vignette_strength : float    = 0
    color_warmth      : float    = 0
    jpeg_quality      : int      = 90


SMARTPHONE_PROFILE: CameraProfile = CameraProfile(
    kind="smartphone",
    base_prnu_strength=0.003,
    base_fpn_row=0.002,
    base_fpn_col=0.002,
    base_read_noise=0.002,  # low
    base_shot_noise=0.01,  # medium
    denoise_strength=0.6,
    blur_sigma=1,
    sharpening_amount=0.6,
    tone_strength=0.55,
    scurve_strength=0.25,
    tone_mode="filmic",
    vignette_strength=0.3,
    color_warmth=0.1,
    jpeg_quality=88,
)

COMPACT_PROFILE: CameraProfile = CameraProfile(
    kind="compact",
    base_prnu_strength=0.004,
    base_fpn_row=0.003,
    base_fpn_col=0.003,
    base_read_noise=0.003,  # slightly higher
    base_shot_noise=0.012,
    denoise_strength=0.4,
    blur_sigma=1,
    sharpening_amount=0.4,
    tone_strength=0.35,
    scurve_strength=0.15,
    tone_mode="reinhard",
    vignette_strength=0.2,
    color_warmth=0.05,
    jpeg_quality=90,
)

CAMERA_PROFILES: MappingProxyType = MappingProxyType({
    "smartphone": SMARTPHONE_PROFILE,
    "compact": COMPACT_PROFILE,
})


def get_camera_profile(kind: str) -> str:
    """Return camera profile for a given camera kind."""
    key = str(kind).strip().lower()
    camera_profile = CAMERA_PROFILES.get(key)
    if camera_profile is None:
        raise ValueError(f"Unknown camera kind: {kind!r}")
    return camera_profile


# ---------------------------------------------------------------------------
# Utility conversion
# ---------------------------------------------------------------------------

def rescale_imgf(img_f: ImageRGBF, *a, **kw) -> ImageRGBF:
    """Exposure-preserving replacement for np.clip(img, 0.0, 1.0)."""
    mn, mx = float(img_f.min()), float(img_f.max())
    if mn >= 0 and mx <= 1:
        return img_f
    
    if mn < 0 and (mx - mn) <= 1:
        return img_f + mn

    return (img_f + mn) / (mx - mn)


def uint8_from_float32(img_f: ImageRGBF | ImageBGRF) -> ImageRGB | ImageBGR:
    """Convert RGB/BGR float32 in [0, 1] to RGB/BGR uint8 with rounding."""
    return (rescale_imgf(img_f) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lens Model Injection: radial distortion + rolling shutter
# ---------------------------------------------------------------------------

def radial_distortion(img: ImageRGBF, k1: float, k2: float = 0.0) -> ImageRGBF:
    """Apply simple radial lens distortion in RGB float space.

    Disable effect by setting k1=0 and k2=0.

    Args:
        img: Input image in RGB float [0, 1], shape (H, W, 3).
        k1: Quadratic radial distortion coefficient.
        k2: Quartic radial distortion coefficient.

    Returns:
        Distorted image, RGB float [0, 1].
    """
    if abs(k1) < EPSILON and abs(k2) < EPSILON:
        return img

    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = (xx - w / 2.0) / (w / 2.0)
    y = (yy - h / 2.0) / (h / 2.0)
    r2 = x * x + y * y

    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_dist = x * radial
    y_dist = y * radial

    map_x = (x_dist * (w / 2.0) + w / 2.0).astype(np.float32)
    map_y = (y_dist * (h / 2.0) + h / 2.0).astype(np.float32)

    warped = cv2.remap(uint8_from_float32(img), map_x, map_y,
                  interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT,
                  borderValue=(255, 255, 255))
    return warped.astype(np.float32) / 255.0


def rolling_shutter_skew(img: ImageRGBF, strength: float = 0.0) -> ImageRGBF:
    """Apply vectorized rolling-shutter-like horizontal skew in RGB float.

    Disable effect by setting strength=0.

    Args:
        img: RGB float image in [0, 1].
        strength: Horizontal skew fraction, e.g. 0.03.
                  Positive -> bottom shifts right, negative -> left.

    Returns:
        Skewed RGB float image.
    """
    if abs(strength) < EPSILON:
        return img

    h, w = img.shape[:2]
    # Per-row fractional shift
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    shift_x = (strength * y * w).astype(np.float32)  # shape (H,)

    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = xx + shift_x[:, None]
    map_y = yy

    warped = cv2.remap(uint8_from_float32(img), map_x, map_y,
                 interpolation=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_REFLECT,
             )
    return warped.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# CFA & Demosaicing
# ---------------------------------------------------------------------------

def cfa_and_demosaic(img: ImageRGBF) -> ImageRGBF:
    """Simulate Bayer CFA and demosaicing to introduce CFA artifacts.

    Args:
        img: RGB float [0, 1].

    Returns:
        Demosaiced RGB float [0, 1].
    """
    img_u8 = uint8_from_float32(img)

    h, w, _ = img_u8.shape
    R = img_u8[..., 0]
    G = img_u8[..., 1]
    B = img_u8[..., 2]

    # RGGB CFA pattern
    mosaic = np.zeros((h, w), dtype=np.uint8)
    # R at (0,0) modulo 2
    mosaic[0::2, 0::2] = R[0::2, 0::2]
    # G at (0,1) and (1,0)
    mosaic[0::2, 1::2] = G[0::2, 1::2]
    mosaic[1::2, 0::2] = G[1::2, 0::2]
    # B at (1,1)
    mosaic[1::2, 1::2] = B[1::2, 1::2]

    # NOTE: cv2.COLOR_BayerRG2BGR returns RGB, not BGR
    rgb_dm = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2BGR).astype(np.float32) / 255.0
    return rgb_dm

# ---------------------------------------------------------------------------
# Sensor noise: PRNU + FPN + shot/read noise (ISO-dependent)
# ---------------------------------------------------------------------------

def sensor_noise(
        img: ImageRGBF,
        profile: CameraProfile,
        iso_level: ISOLevel | Real,
        rng: np.random.Generator,
    ) -> ImageRGBF:
    """Add sensor-like noise (PRNU, FPN, shot, read) in RGB float space."""
    h, w, _ = img.shape

    # ISO scaling factor
    if isinstance(iso_level, Real):
        iso_factor = float(np.clip(abs(iso_level), 0.0, 3.0))
    else:
        key = str(iso_level).strip().lower()
        iso_factor = ISO_SF.get(key, 1.0)

    if iso_factor < EPSILON:
        # All noise effectively off
        return img

    img = rescale_imgf(img.astype(np.float32))

    # PRNU (multiplicative)
    base_prnu = profile.base_prnu_strength
    if base_prnu > EPSILON:
        prnu_sigma = base_prnu * iso_factor
        prnu_map = rng.normal(
            loc=1.0,
            scale=prnu_sigma,
            size=(h, w, 1),
        ).astype(np.float32)
    else:
        prnu_map = np.ones((h, w, 1), dtype=np.float32)

    # Row/column FPN
    base_fpn_row = profile.base_fpn_row
    base_fpn_col = profile.base_fpn_col

    if base_fpn_row > EPSILON:
        fpn_row_sigma = base_fpn_row * iso_factor
        row_pattern = rng.normal(loc=0.0, scale=fpn_row_sigma, size=(h, 1, 1),
                                ).astype(np.float32)
    else:
        row_pattern = np.zeros((h, 1, 1), dtype=np.float32)

    if base_fpn_col > EPSILON:
        fpn_col_sigma = base_fpn_col * iso_factor
        col_pattern = rng.normal(loc=0.0, scale=fpn_col_sigma, size=(1, w, 1),
                                ).astype(np.float32)
    else:
        col_pattern = np.zeros((1, w, 1), dtype=np.float32)

    fpn = 1.0 + row_pattern + col_pattern

    # Combine multiplicative effects
    base = img * prnu_map * fpn

    # Shot noise (brightness dependent)
    if profile.base_shot_noise > EPSILON:
        shot_sigma = profile.base_shot_noise * iso_factor
        shot_std = shot_sigma * np.sqrt(rescale_imgf(base))
        shot_noise = shot_std * rng.normal(loc=0.0, scale=1.0, size=base.shape,
                                          ).astype(np.float32)
    else:
        shot_noise = np.zeros_like(base, dtype=np.float32)

    # Read noise (additive)
    if profile.base_read_noise > EPSILON:
        read_sigma = profile.base_read_noise * iso_factor
        read_noise = rng.normal(0.0, read_sigma, size=base.shape).astype(np.float32)
    else:
        read_noise = np.zeros_like(base, dtype=np.float32)

    noisy = base + shot_noise + read_noise
    return rescale_imgf(noisy)


# ---------------------------------------------------------------------------
# ISP Denoising & Sharpening
# ---------------------------------------------------------------------------

def isp_denoise_and_sharpen(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """
    ISP processing stage: denoising (bilateral) + unsharp-mask sharpening.

    Controls:
        denoise_strength: 0..1 - more = stronger noise reduction & smoothing
        blur_sigma:       0.5..3 - radius of unsharp-mask blur kernel
        sharpening_amount:0..1 - strength of sharpening (halos)
    """

    img_u8 = uint8_from_float32(img)

    # ---- 1. Bilateral denoising -------------------------------------------
    ds = profile.denoise_strength
    if ds > EPSILON:
        # Kernel diameter: 5..15 px
        d = int(np.clip(5 + ds * 20, 5, 25))

        # Color sigma: 10..60
        sigmaColor = float(10 + ds * 50)

        # Spatial sigma: 2..14
        sigmaSpace = float(2 + ds * 12)

        den = cv2.bilateralFilter(img_u8, d=d, sigmaColor=sigmaColor,
                                  sigmaSpace=sigmaSpace)
        img_f = den.astype(np.float32) / 255.0
    else:
        img_f = img  # no denoising

    # ---- 2. Sharpening -----------------------------------------------------
    sharp_amt = profile.sharpening_amount
    if sharp_amt < EPSILON:
        return rescale_imgf(img_f)

    # Gaussian blur radius (sigma)
    blur_sigma = max(0.1, float(profile.blur_sigma))

    # Unsharp mask
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=blur_sigma)
    sharp = img_f + sharp_amt * (img_f - blur)

    return rescale_imgf(sharp)


# ---------------------------------------------------------------------------
# Tone Mapping Stage
# ---------------------------------------------------------------------------

def tone_map_reinhard(img: ImageRGBF, strength: float) -> ImageRGBF:
    """Reinhard global tone curve. Strength 0..1."""
    if strength <= EPSILON:
        return img
    tone = img / (1.0 + img)
    return img * (1.0 - strength) + tone * strength


def tone_map_filmic(img: ImageRGBF, strength: float) -> ImageRGBF:
    """Hable filmic operator. Strength 0..1."""
    if strength <= EPSILON:
        return img

    a = 0.22
    b = 0.30
    c = 0.10
    d = 0.20
    e = 0.01
    f = 0.30

    def hable(x: np.ndarray) -> np.ndarray:
        return ((x * (a * x + c * b) + d * e) /
                (x * (a * x + b) + d * f) - e / f)

    tone = hable(img)
    tone = rescale_imgf(tone)
    return img * (1.0 - strength) + tone * strength


def tone_scurve(img: ImageRGBF, strength: float) -> ImageRGBF:
    """Local contrast S-curve using tanh. Gives typical 'HDR' pop."""
    if strength <= EPSILON:
        return img
    x = img * 2.0 - 1.0
    y = np.tanh(x * (1.0 + strength * 2.0))
    return rescale_imgf((y + 1.0) * 0.5)


def tone_mapping(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Combined tone-mapping stage."""
    tone_strength = profile.tone_strength
    scurve_strength = profile.scurve_strength
    tone_mode = profile.tone_mode
    if tone_mode == "filmic":
        img = tone_map_filmic(img, tone_strength)
    else:
        img = tone_map_reinhard(img, tone_strength)

    img = tone_scurve(img, scurve_strength)
    return rescale_imgf(img)


# ---------------------------------------------------------------------------
# Vignette + color warmth
# ---------------------------------------------------------------------------

def vignette_and_color(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Apply vignetting and mild color bias in RGB float32 [0, 1].

    Assumes:
        - img.dtype is np.float32
        - img values are in [0, 1]
        - channel order is RGB (0=R, 1=G, 2=B)
    """
    vignette_strength = float(profile.vignette_strength)
    warm_strength = float(profile.color_warmth)

    # Nothing to do
    if vignette_strength <= 0.0 and abs(warm_strength) <= 0.0:
        return img

    h, w, _ = img.shape

    # Normalized coordinates in [-1, 1]
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    x = (np.arange(w, dtype=np.float32) - cx) / max(cx, 1.0)
    y = (np.arange(h, dtype=np.float32) - cy) / max(cy, 1.0)
    xx, yy = np.meshgrid(x, y)

    # Radial distance, normalized to [0, 1]
    r = np.sqrt(xx * xx + yy * yy)
    r_max = float(r.max())
    r_norm = r / r_max

    # --- Radial masks (float32) ---
    # Quadratic vignette
    vignette_mask = 1.0 - vignette_strength * (r_norm ** 2)

    # Warm/cool masks (for color_warmth)
    if abs(warm_strength) > EPSILON:
        r_pow = r_norm ** 1.2
        warm_mask = 1.0 + warm_strength * r_pow
        cool_mask = 1.0 - 0.5 * warm_strength * r_pow
    else:
        warm_mask = 1.0
        cool_mask = 1.0

    # --- Apply modulation (RGB order) ---
    out = img * vignette_mask[..., None]

    if abs(warm_strength) > 0.0:
        out[..., 0] *= warm_mask   # R channel
        out[..., 2] *= cool_mask   # B channel

    # Stay in float32, just clamp range
    return rescale_imgf(out)


# ---------------------------------------------------------------------------
# JPEG round-trip - injects realistic DCT/blockiness
# ---------------------------------------------------------------------------

def jpeg_roundtrip(img: ImageRGBF, quality: int) -> ImageRGBF:
    """Run image through JPEG encode/decode to add DCT artifacts.

    Args:
        img: RGB float [0, 1].
        quality: JPEG quality (e.g., 85-95).

    Returns:
        RGB float [0, 1] after JPEG roundtrip.
    """
    bgr = cv2.cvtColor(uint8_from_float32(img), cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr, encode_param)
    if not ok:
        # Fallback: return original if encoding fails.
        return img

    bgr_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb_jpeg = cv2.cvtColor(bgr_jpeg, cv2.COLOR_BGR2RGB)
    return rgb_jpeg.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Correction Engine
# ---------------------------------------------------------------------------

def apply_camera_model(
        img: ImageRGBF,
        correction_profile: CameraProfile = None,
        camera_kind: str = "smartphone",
        iso_level: ISOLevel | Real = "mid",
        k1: float = -0.15,
        k2: float = 0.02,
        rolling_strength: float = 0.03,
        apply_jpeg: bool = True,
        cfa_enabled: bool = True,
        rng: np.random.Generator = None,
    ) -> ImageRGBF:
    """Camera-like correction engine.

    Args:
        img: Input RGB float image in [0, 1].
        camera_kind: 'smartphone' or 'compact'.
        iso_level: 'low', 'mid', 'high' or a numeric ISO factor.
        k1: Primary radial distortion coefficient.
        k2: Secondary radial distortion coefficient.
        rolling_strength: Skew magnitude, e.g. ~0.02-0.05.
        apply_jpeg: Whether to run a JPEG roundtrip for DCT artifacts.
        rng: Optional NumPy Generator. If None, a default RNG is created.

    Returns:
        RGB float image in [0, 1] with camera-like artifacts.
    """
    if rng is None:
        rng = np.random.default_rng()

    if correction_profile is None:
        profile = get_camera_profile(camera_kind)
    else:
        profile = correction_profile

    img = rescale_imgf(img.astype(np.float32))

    # 1) Lens geometry in linear RGB
    img = radial_distortion(img, k1=k1, k2=k2)
    img = rolling_shutter_skew(img, strength=rolling_strength)

    # 2) CFA + demosaic
    if cfa_enabled:
        img = cfa_and_demosaic(img)

    # 3) Sensor noise (PRNU + FPN + shot / read)
    img = sensor_noise(img, profile=profile, iso_level=iso_level, rng=rng)

    # 4) ISP denoise + sharpening
    img = isp_denoise_and_sharpen(img, profile=profile)

    # 4.5) Tone mapping
    img = tone_mapping(img, profile)

    # 5) Vignette + color bias
    img = vignette_and_color(img, profile=profile)

    # 6) Optional JPEG roundtrip
    if apply_jpeg:
        img = jpeg_roundtrip(img, quality=profile.jpeg_quality)

    return rescale_imgf(img)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def round_sig(x, sig=3):
    x = float(x)
    if x == 0:
        return 0.0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def demo():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_rgbf: ImageRGBF = rgbf_from_rgba(base_rgba)

    rng = random.Random(os.getpid() ^ int(time.time()))

    img_rnd = rgbf_from_rgba(render_scene(
                  canvas_bg_idx = rng.randrange(len(PAPER_COLORS)),
                  plot_bg_idx = rng.randrange(len(PAPER_COLORS)),
              ))
    random_props = {
        "img":            img_rnd,
    }

    demos = {
        "BASELINE": img_rnd,
    }

    default_core = {
        # 3) Sensor noise

        "base_prnu_strength": 0, # [0, 0.01]
        "base_fpn_row"      : 0, # [0, 0.01]
        "base_fpn_col"      : 0, # [0, 0.01]
        "base_read_noise"   : 0, # [0, 0.01]
        "base_shot_noise"   : 0, # [0, 0.03]

        # 4) ISP denoise + sharpen

        "denoise_strength"  : 0, # [0, 1]
        "blur_sigma"        : 0, # [0, 3]
        "sharpening_amount" : 0, # [0, 1]

        # 5) Tone mapping
        
        "tone_strength"     : 0, # [0, 1]
        "scurve_strength"   : 0, # [0, 0.5]

        # 6) Vignette + color warmth

        "vignette_strength" : 0, # [0, 0.5]
        "color_warmth"      : 0, # [0, 0.2]
    }

    default_extras = {
        "iso_level"         : 1, # [0, 2]
    
        "rolling_strength"  : 0, # [0, 0.05]

        # 1) Lens geometry

        "k1"                : 0, # [-0.2, 0.2]
        "k2"                : 0, # [-0.02, 0.02]

        # 7) JPEG

        "apply_jpeg"        : False,

        # 2) CFA + demosaic (always on, to keep realism)

        "cfa_enabled"       : False,
    }

    # ---------------------------------------------------------------------------
    # Demo Sets
    # ---------------------------------------------------------------------------
    

    # 1) Lens geometry
    # ----------------
    varname = "k1"
    stepcount = 9
    (mn, mx) = PARAM_RANGES[varname]
    span = mx - mn + EPSILON

    custom_extras_k1 = [
        {varname: round_sig(val)} for val in np.linspace(mn, mx, stepcount).astype(float).tolist()
    ]
    
    varname = "k2"
    stepcount = 9
    (mn, mx) = PARAM_RANGES[varname]
    span = mx - mn + EPSILON

    custom_extras_k2 = [
        {varname: round_sig(val)} for val in np.linspace(mn, mx, stepcount).astype(float).tolist()
    ]

    varname = "rolling_strength"
    stepcount = 9
    (mn, mx) = PARAM_RANGES[varname]
    span = mx - mn + EPSILON

    custom_extras_rolling_strength = [
        {varname: round_sig(val)} for val in np.linspace(mn, mx, stepcount).astype(float).tolist()
    ]
 
    # 2) CFA + demosaic
    # -----------------
    custom_extras_cfa_enabled = [
        {"cfa_enabled": True}
    ]
 
#     # 3) Sensor noise
#     iso_val = sliders["ISO"].val
#     if iso_val < 0:
#       iso_val = 0.0
#     img = sensor_noise(img, profile=profile, iso_level=iso_val, rng=rng)
# 
#     # 4) ISP denoise + sharpen
#     img = isp_denoise_and_sharpen(img, profile=profile)
# 
#     # 5) Tone mapping
#     img = tone_mapping(img, profile=profile)
# 
#     # 6) Vignette + color warmth
#    img = vignette_and_color(img, profile=profile)
#
#    # JPEG intentionally skipped in interactive mode
    
    
    
    custom_core_prnu = [
        {"base_prnu_strength": 0.000},
        {"base_prnu_strength": 0.002},
        {"base_prnu_strength": 0.004},
        {"base_prnu_strength": 0.006},
        {"base_prnu_strength": 0.008},
        {"base_prnu_strength": 0.010},
    ]

    # ---------------------------------------------------------------------------
    # Demo Runner
    # ---------------------------------------------------------------------------

    custom_core   = [{}] #custom_core_prnu
    custom_extras = custom_extras_cfa_enabled
    
    print(custom_core)
    if len(custom_core) > 1:
        for custom_props in custom_core:
            title = []
            for key, val in custom_props.items():
                title.append(f"key: '{key}': val '{val}'")
            title = "".join(title)
            print(title)
            demos[title] = rgb_from_rgbf(apply_camera_model(
                img=img_rnd,
                correction_profile=CameraProfile(**{**default_core, **custom_props}),
                **{**default_extras, **custom_extras[0]}
            ))
    else:
        for custom_props in custom_extras:
            title = []
            for key, val in custom_props.items():
                title.append(f"key: '{key}': val '{val}'")
            title = "".join(title)
            print(title)
            demos[title] = rgb_from_rgbf(apply_camera_model(
                img=img_rnd,
                correction_profile=CameraProfile(**{**default_core, **custom_core[0]}),
                **{**default_extras, **custom_props}
            ))

 
    show_RGBx_grid(demos, n_columns=4)


# ---------------------------------------------------------------------------
# Interactive Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo()
  
  
  
  
def dummy():  
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Slider, RadioButtons, CheckButtons
  import time

  # ---- Background colors -------------------------------------------------
  COLORS = {
      "red": (1.0, 0.0, 0.0),
      "green": (0.0, 1.0, 0.0),
      "blue": (0.0, 0.0, 1.0),
      "white": (1.0, 1.0, 1.0),
      "cornsilk": (1.0, 0.9725, 0.8627),
      "ivory": (1.0, 1.0, 0.9412),
      "oldlace": (0.992, 0.961, 0.902),
      "floralwhite": (1.0, 0.9804, 0.9412),
      "whitesmoke": (0.9608, 0.9608, 0.9608),
  }

  H, W = 512, 512
  init_color = "white"
  base_img = np.ones((H, W, 3), dtype=np.float32)
  base_img[:] = COLORS[init_color]

  fig, ax = plt.subplots(figsize=(7, 7))
  plt.subplots_adjust(left=0.32, bottom=0.05)
  disp = ax.imshow(base_img, vmin=0, vmax=1)
  ax.set_axis_off()

  # ---- Collapsible Panels via CheckButtons -------------------------------
  panel_ax = plt.axes([0.02, 0.55, 0.25, 0.40])
  panel = CheckButtons(
      panel_ax,
      ["Optics", "Sensor", "ISP", "Tone", "Vignette/Color"],
      [True, False, False, False, False],
  )

  # ---- Slider Groups: name, min, max, init -------------------------------
  slider_groups = {
      "Optics": [
          ("k1", -0.3, 0.3, 0.0),
          ("k2", -0.05, 0.05, 0.0),
          ("rolling", -0.1, 0.1, 0.0),
          ("ISO", 0.0, 3.0, 0.0),
      ],
      "Sensor": [
          ("prnu", 0.0, 0.02, 0.0),
          ("fpn_row", 0.0, 0.02, 0.0),
          ("fpn_col", 0.0, 0.02, 0.0),
          ("shot", 0.0, 0.03, 0.0),
          ("read", 0.0, 0.01, 0.0),
      ],
      "ISP": [
          ("denoise", 0.0, 1.0, 0.0),
          ("blur_sigma", 0.1, 3.0, 0.0),
          ("sharpen", 0.0, 1.0, 0.0),
      ],
      "Tone": [
          ("tone", 0.0, 1.0, 0.0),
          ("scurve", 0.0, 1.0, 0.0),
      ],
      "Vignette/Color": [
          ("vignette", 0.0, 0.6, 0.0),
          ("warm", 0.0, 0.2, 0.0),
      ],
  }

  sliders: dict[str, Slider] = {}
  slider_axes: dict[str, list[plt.Axes]] = {}

  ypos = 0.45
  for group, defs in slider_groups.items():
    slider_axes[group] = []
    for name, vmin, vmax, vinit in defs:
      ax_s = plt.axes([0.32, ypos, 0.60, 0.022])
      ax_s.set_visible(group == "Optics")  # only optics visible initially
      sliders[name] = Slider(ax_s, name, vmin, vmax, valinit=vinit)
      slider_axes[group].append(ax_s)
      ypos -= 0.03

  # Tone mode radio button
  ax_mode = plt.axes([0.02, 0.34, 0.25, 0.15])
  rb_mode = RadioButtons(ax_mode, ["reinhard", "filmic"], active=0)
  ax_mode.set_visible(False)

  # Background color radio buttons
  ax_color = plt.axes([0.02, 0.10, 0.25, 0.20])
  rb_color = RadioButtons(
      ax_color,
      list(COLORS.keys()),
      active=list(COLORS.keys()).index(init_color),
  )

  # ---- Panel toggle: show only sliders of selected group -----------------
  def panel_toggle(label: str) -> None:
    for g, axes_g in slider_axes.items():
      visible = (g == label)
      for ax_s in axes_g:
        ax_s.set_visible(visible)
    ax_mode.set_visible(label == "Tone")
    fig.canvas.draw_idle()

  panel.on_clicked(panel_toggle)

  # ---- Real-time throttling state ----------------------------------------
  state = {"last_t": 0.0}
  min_interval = 0.05  # 20 FPS cap

  rng = np.random.default_rng(1234)

  def build_profile_from_sliders() -> "CameraProfile":
    """Construct a neutral-ish profile from the slider values."""
    return CameraProfile(
        kind="smartphone",
        base_prnu_strength=sliders["prnu"].val,
        base_fpn_row=sliders["fpn_row"].val,
        base_fpn_col=sliders["fpn_col"].val,
        base_read_noise=sliders["read"].val,
        base_shot_noise=sliders["shot"].val,
        denoise_strength=sliders["denoise"].val,
        blur_sigma=max(0.1, sliders["blur_sigma"].val),
        sharpening_amount=sliders["sharpen"].val,
        tone_strength=sliders["tone"].val,
        scurve_strength=sliders["scurve"].val,
        tone_mode=rb_mode.value_selected,
        vignette_strength=sliders["vignette"].val,
        color_warmth=sliders["warm"].val,
        jpeg_quality=90,
    )

  def apply_pipeline(img_rgb: np.ndarray, profile: "CameraProfile") -> np.ndarray:
    """Run the full engine explicitly with the given profile."""
    img = rescale_imgf(img_rgb.astype(np.float32))

    # 1) Lens geometry
    k1 = sliders["k1"].val
    k2 = sliders["k2"].val
    rolling = sliders["rolling"].val
    if abs(k1) > 0 or abs(k2) > 0:
      img = radial_distortion(img, k1=k1, k2=k2)
    if abs(rolling) > 0:
      img = rolling_shutter_skew(img, strength=rolling)

    # 2) CFA + demosaic (always on, to keep realism)
    img = cfa_and_demosaic(img)

    # 3) Sensor noise
    iso_val = sliders["ISO"].val
    if iso_val < 0:
      iso_val = 0.0
    img = sensor_noise(img, profile=profile, iso_level=iso_val, rng=rng)

    # 4) ISP denoise + sharpen
    img = isp_denoise_and_sharpen(img, profile=profile)

    # 5) Tone mapping
    img = tone_mapping(img, profile=profile)

    # 6) Vignette + color warmth
    img = vignette_and_color(img, profile=profile)

    # JPEG intentionally skipped in interactive mode
    return rescale_imgf(img)

  def update(_):
    t = time.time()
    if t - state["last_t"] < min_interval:
      return
    state["last_t"] = t

    # update background plain color
    base_img[:] = COLORS[rb_color.value_selected]

    # build profile from current sliders
    profile = build_profile_from_sliders()

    # apply full pipeline
    out = apply_pipeline(base_img, profile)

    disp.set_data(out)
    fig.canvas.draw_idle()

  # connect sliders and controls
  for s in sliders.values():
    s.on_changed(update)
  rb_mode.on_clicked(update)
  rb_color.on_clicked(update)

  # ---- Reset button: zero active panel -----------------------------------
  ax_reset = plt.axes([0.05, 0.02, 0.20, 0.05])
  btn_reset = CheckButtons(ax_reset, ["Reset Active Panel"], [False])

  def reset_active(_):
    # find active group
    active = None
    for label, state_flag in zip(panel.labels, panel.get_status()):
      if state_flag:
        active = label.get_text()
        break
    if active in slider_axes:
      # reset all sliders in that group to 0
      for (name, _vmin, _vmax, _vinit) in slider_groups[active]:
        sliders[name].reset()  # resets to valinit (which we set to 0.0)
    fig.canvas.draw_idle()

  btn_reset.on_clicked(reset_active)

  # initial render
  update(None)
  plt.show()
