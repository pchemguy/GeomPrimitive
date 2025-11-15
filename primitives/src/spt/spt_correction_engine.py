"""
spt_correction_engine.py
------------------------

Ref: https://chatgpt.com/c/69172a6a-b78c-8326-b080-7b02e61b4730

Camera-Like Correction Engine (Internal ISP Simulator)
===============================================================

This module implements a self-contained, physically-motivated camera
simulation pipeline (“Correction Engine”) designed to inject the kinds
of imperfections, artifacts, and statistics that are present in real
laboratory photographs acquired using smartphones or compact cameras.

Each methods implements a particular task, taking and returning the
image being processed as RGB float32 in [0, 1]. Individual methods
convert from RGBf and back as necessary.

The goal is to ensure that synthetic images produced by rendering
systems (e.g., Matplotlib, synthetic textures, geometric overlays) will
exhibit realistic photographic traces — including optical distortions,
CFA/demosaicing artifacts, sensor noise, ISP processing, color
characteristics, and JPEG quantization — so that forensic tools and
ML pipelines cannot easily distinguish real lab photos from synthetic
data, thus reducing the risks of artificial biases if generated images
are used for AI training.


Pipeline Overview
-----------------

The Correction Engine consists of the following sequential stages:

1. **Optical Lens Model**
   - Radial distortion (barrel/pincushion)
   - Rolling-shutter geometric skew
   - (Extensible to chromatic aberration or PSF blurring)

2. **CFA & Demosaicing Simulation**
   - Convert RGB into virtual Bayer RAW (RGGB)
   - Apply bilinear/VNG demosaicing
   - Introduces zippering, color moire, and channel-correlated artifacts

3. **Sensor Pattern Noise (ISO-dependent)**
   - PRNU: pixel-level multiplicative gain variations
   - FPN: row/column banding patterns
   - Proper ISO scaling factors

4. **Shot Noise + Read Noise**
   - Brightness-dependent Poisson noise
   - Additive Gaussian read noise
   - Correct linear-RGB distribution

5. **ISP Denoising + Sharpening**
   - Bilateral denoiser (approximates smartphone denoise)
   - Unsharp mask halo formation (edge overshoot/undershoot)

6. **Vignetting & Color Tone**
   - Optical falloff simulated via radial mask
   - Mild color warmth/bias (e.g. phone LED tint)

7. **JPEG Simulation (optional)**
   - Roundtrip through JPEG at configurable quality
   - Introduces DCT blocking, grid patterns, and quantization ghosts

8. **Metadata Injection / Removal (future extension)**


Design Notes
------------

- All steps operate in **linear RGB**, not sRGB. Real photon noise,
  PRNU, and demosaicing must be applied before gamma/tone curves.

- The engine is parameterized via `CameraProfile`, allowing different
  "personalities": smartphone-like, compact-camera-like, noisy, clean,
  etc.

- Every stage can be disabled by setting its coefficients to zero.

- This module is intentionally modular: adding PSF blur, chromatic
  aberration, local tone mapping, or more sophisticated denoisers is
  straightforward.

"""

from __future__ import annotations

__all__ = [apply_camera_model]

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from types import MappingProxyType
from numbers import Real

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

from mpl_utils import ImageBGR, ImageBGRF, ImageRGB, ImageRGBF, ImageRGBA


EPSILON = 1e-8
ISOLevel = Literal["low", "mid", "high"]
ISO_SF: dict[ISOLevel, float] = MappingProxyType({
    "low"  : 0.6, 
    "mid"  : 1.0, 
    "high" : 1.6,
})

CameraKind = Literal["smartphone", "compact"]


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
        sharpening_amount: Unsharp-mask strength for ISP sharpening.
        vignette_strength: Vignetting strength (center-to-edge).
        color_warmth: Warm bias, positive shifts towards warmer tones.
        jpeg_quality: JPEG quality for optional roundtrip (higher = less compressed).
    """
    kind: CameraKind
    base_prnu_strength: float
    base_fpn_row: float
    base_fpn_col: float
    base_read_noise: float
    base_shot_noise: float
    sharpening_amount: float
    vignette_strength: float
    color_warmth: float
    jpeg_quality: int


SMARTPHONE_PROFILE: CameraProfile = CameraProfile(
    kind="smartphone",
    base_prnu_strength=0.003,
    base_fpn_row=0.002,
    base_fpn_col=0.002,
    base_read_noise=0.002,   # low
    base_shot_noise=0.01,    # medium
    sharpening_amount=0.6,
    vignette_strength=0.2,
    color_warmth=0.1,
    jpeg_quality=88,
)


COMPACT_PROFILE: CameraProfile = CameraProfile(
    kind="compact",
    base_prnu_strength=0.004,
    base_fpn_row=0.003,
    base_fpn_col=0.003,
    base_read_noise=0.003,   # slightly higher
    base_shot_noise=0.012,
    sharpening_amount=0.4,
    vignette_strength=0.1,
    color_warmth=0.05,
    jpeg_quality=90,
)


CAMERA_PROFILES: dict[CameraKind, CameraProfile] = MappingProxyType({
    "smartphone": SMARTPHONE_PROFILE,
    "compact"   : COMPACT_PROFILE,
})


def get_camera_profile(kind: CameraKind) -> CameraProfile:
    camera_profile = CAMERA_PROFILES.get(kind.strip().lower())
    if camera_profile is None:
        raise ValueError(f"Unknown camera kind: {kind}")
    return camera_profile


def uint8_from_float32(img_f: ImageRGBF | ImageBGRF) -> ImageRGB | ImageBGR:
    """Convert RGB/BGR float32 in [0, 1] to RGB/BGR uint8."""
    return (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lens Model Injection: radial distortion + rolling shutter
# ---------------------------------------------------------------------------
def radial_distortion(
        img: ImageRGBF,
        k1: float,
        k2: float = 0.0,
    ) -> ImageRGBF:
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
    yy, xx = np.indices((h, w))
    x = (xx - w / 2) / (w / 2)
    y = (yy - h / 2) / (h / 2)
    r2 = x * x + y * y

    radial = 1 + k1 * r2 + k2 * r2 * r2
    map_x = ((x * radial + 1) * (w / 2)).astype(np.float32)
    map_y = ((y * radial + 1) * (h / 2)).astype(np.float32)
  
    # remap expects BGR/whatever, but we only care about spatial mapping
    out = cv2.remap((img * 255.0 + 0.5).astype(np.uint8), map_x, map_y,
               interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
          ).astype(np.float32) / 255.0

    return out


def rolling_shutter_skew(
        img: ImageRGBF,
        strength: float = 0.0,
    ) -> ImageRGBF:
    """Apply simple rolling-shutter-like skew in RGB float.
    
    Disable effect by setting strength=0.
    
    Args:
        img: RGB float image in [0, 1].
        strength: Horizontal skew fraction, e.g. 0.03. Positive -> skew right.
    
    Returns:
        Skewed RGB float image.
    """
    if abs(strength) < EPSILON:
        return img

    h, w = img.shape[:2]

    # Generate per-row fractional shift (float, not int)
    # amount(y) = strength * (y/h) * w   (like your loop)
    y = np.linspace(0, 1, h, dtype=np.float32)
    shift_x = (strength * y * w).astype(np.float32)  # shape (H,)

    # Build mapping grid for cv2.remap
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # shift each row by fractional amount
    map_x = xx + shift_x[:, None]      # broadcast shift per row
    map_y = yy

    # Border interpolation: reflect is realistic for rolling shutter artifacts
    warped = cv2.remap(uint8_from_float32(img), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

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
    R, G, B = img_u8[..., 0], img_u8[..., 1], img_u8[..., 2]
    
    # RGGB CFA pattern
    mosaic = np.zeros((h, w), dtype=np.uint8)
    # R at (0,0) modulo 2
    mosaic[0::2, 0::2] = R[0::2, 0::2]  # R
    # G at (0,1) and (1,0)
    mosaic[0::2, 1::2] = G[0::2, 1::2]  # G
    mosaic[1::2, 0::2] = G[1::2, 0::2]  # G
    # B at (1,1)
    mosaic[1::2, 1::2] = B[1::2, 1::2]  # B
    
    bgr_dm = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2BGR)
    rgb_dm_u8 = cv2.cvtColor(bgr_dm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb_dm_u8


# ---------------------------------------------------------------------------
# Sensor noise: PRNU + FPN + shot/read noise (ISO-dependent)
# ---------------------------------------------------------------------------
def sensor_noise(
        img       : ImageRGBF,
        profile   : CameraProfile,
        iso_level : ISOLevel | Real,
        rng       : np.random.Generator,
    ) -> ImageRGBF:
    """Add sensor-like noise (PRNU, FPN, shot, read) in RGB float space."""
    h, w, _ = img.shape

    # ISO scaling factors
    if isinstance(iso_level, Real):
        iso_factor = abs(iso_level)
    else:
        iso_factor = ISO_SF.get(iso_level.strip().lower(), 1.0)

    if iso_factor < EPSILON:
        return img

    # PRNU - pixel-level gain variation
    if profile.base_prnu_strength > EPSILON:
        prnu_strength = profile.base_prnu_strength * iso_factor
        prnu_map = rng.normal(loc=1.0, scale=prnu_strength, size=(h, w, 1)
                             ).astype(np.float32)
    else:
        prnu_map = np.ones((h, w, 1), dtype=np.float32)

    # Row FPN - row-wise additive banding
    if profile.base_fpn_row > EPSILON:
        fpn_row = profile.base_fpn_row * iso_factor
        row_pattern = rng.normal(loc=0.0, scale=fpn_row, size=(h, 1, 1)
                                ).astype(np.float32)
    else:
        row_pattern = np.zeros((h, 1, 1), dtype=np.float32)    
    
    # Column FPN - column-wise additive banding
    if profile.base_fpn_col > EPSILON:
        fpn_col = profile.base_fpn_col * iso_factor
        col_pattern = rng.normal(loc=0.0, scale=fpn_col, size=(1, w, 1)
                                ).astype(np.float32)
    else:
        col_pattern = np.zeros((1, w, 1), dtype=np.float32)

    fpn = 1.0 + row_pattern + col_pattern
    base = img * prnu_map * fpn

    # Shot noise (brightness dependent)
    if profile.base_shot_noise > EPSILON:
        shot_sigma = profile.base_shot_noise * iso_factor
        shot_noise = (np.sqrt(base) * 
                      rng.normal(0.0, shot_sigma, size=img.shape).astype(np.float32))
    else:
        shot_noise = np.zeros_like(img)

    # Read noise (additive)
    if profile.base_read_noise > EPSILON:
        read_sigma = profile.base_read_noise * iso_factor
        read_noise = rng.normal(0.0, read_sigma, size=img.shape).astype(np.float32)
    else:
        read_noise = np.zeros_like(img)

    noisy = np.clip(base + shot_noise + read_noise, 0.0, 1.0)
    return noisy


# ---------------------------------------------------------------------------
# ISP denoise + sharpening
# ---------------------------------------------------------------------------
def isp_denoise_and_sharpen(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Approximate ISP denoising + sharpening."""   
    # Bilateral denoise (simple ISP-like noise reduction)
    img_u8 = uint8_from_float32(img)
    den = cv2.bilateralFilter(img_u8, d=5, sigmaColor=20, sigmaSpace=5)
    den_f = den.astype(np.float32) / 255.0
    
    # Unsharp mask
    blur = cv2.GaussianBlur(den_f, (0, 0), sigmaX=1.0)
    sharp = np.clip(den_f + profile.sharpening_amount * (den_f - blur), 0.0, 1.0)
    return sharp


# ---------------------------------------------------------------------------
# Vignette + color warmth
# ---------------------------------------------------------------------------
def vignette_and_color(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Apply vignetting and mild color bias in RGB float space."""
    h, w, _ = img.shape
    yy, xx = np.indices((h, w))
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / (r.max() + EPSILON)
    
    vign = (1.0 - profile.vignette_strength * r_norm**2).astype(np.float32)[..., None]
    
    img_v = img * vign
    
    warm = profile.color_warmth
    if warm != 0.0:
        img_v[..., 0] = np.clip(img_v[..., 0] + warm * 0.03, 0.0, 1.0)  # R
        img_v[..., 2] = np.clip(img_v[..., 2] - warm * 0.03, 0.0, 1.0)  # B
    
    return np.clip(img_v, 0.0, 1.0)


# ---------------------------------------------------------------------------
# JPEG round-trip - injects realistic DCT/blockiness
# ---------------------------------------------------------------------------
def jpeg_roundtrip(img: ImageRGBF, quality: int) -> ImageRGBF:
    """Run image through JPEG encode/decode to add DCT artifacts.
    
    Args:
        img: RGB float [0, 1].
        quality: JPEG quality (e.g., 85–95).
    
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
        img              : ImageRGBF,
        camera_kind      : CameraKind          = "smartphone",
        iso_level        : ISOLevel            = "mid",
        lens_k1          : float               = -0.15,
        lens_k2          : float               = 0.02,
        rolling_strength : float               = 0.03,
        apply_jpeg       : bool                = True,
        rng              : np.random.Generator = None,
    ) -> ImageRGBF:
    """Camera-like correction engine.

    Args:
        img: Input RGB float image in [0, 1].
        camera_kind: 'smartphone' or 'compact'.
        iso_level: 'low', 'mid', or 'high' ISO behavior.
        enable_lens_distortion: Whether to apply radial distortion.
        lens_k1: Primary radial distortion coefficient.
        lens_k2: Secondary radial distortion coefficient.
        rolling_strength: Skew magnitude, e.g. ~0.02–0.05.
        apply_jpeg: Whether to run a JPEG roundtrip for DCT artifacts.
        rng: Optional NumPy Generator. If None, a default RNG is created.

    Returns:
        RGB float image in [0, 1] with camera-like artifacts.
    """
    if rng is None:
        rng = np.random.default_rng()

    profile = get_camera_profile(camera_kind)

    img = np.clip(img.astype(np.float32), 0.0, 1.0)

    # 1) Lens geometry in linear RGB
    img = radial_distortion(img, k1=lens_k1, k2=lens_k2)
    img = rolling_shutter_skew(img, strength=rolling_strength)

    # 2) CFA + demosaic
    img = cfa_and_demosaic(img)

    # 3) Sensor noise (PRNU + FPN + shot / read)
    img = sensor_noise(img, profile=profile, iso_level=iso_level, rng=rng)

    # 4) ISP denoise + sharpening
    img = isp_denoise_and_sharpen(img, profile=profile)

    # 5) Vignette + color bias
    img = vignette_and_color(img, profile=profile)

    # 6) Optional JPEG roundtrip
    if apply_jpeg:
        img = jpeg_roundtrip(img, quality=profile.jpeg_quality)

    return np.clip(img, 0.0, 1.0)
