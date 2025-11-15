"""
spt_correction_engine.py
------------------------

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
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Local imports (type aliases from your mpl_utils)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

from mpl_utils import ImageBGR, ImageBGRF, ImageRGB, ImageRGBF, ImageRGBA 


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

CameraKind = Literal["smartphone", "compact"]
ToneMode = Literal["reinhard", "filmic"]


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

    kind: CameraKind
    base_prnu_strength: float
    base_fpn_row: float
    base_fpn_col: float
    base_read_noise: float
    base_shot_noise: float
    denoise_strength: float
    blur_sigma: float
    sharpening_amount: float
    tone_strength: float = 0.5
    scurve_strength: float = 0.2
    tone_mode: ToneMode = "reinhard"
    vignette_strength: float
    color_warmth: float
    jpeg_quality: int


SMARTPHONE_PROFILE: CameraProfile = CameraProfile(
    kind="smartphone",
    base_prnu_strength=0.003,
    base_fpn_row=0.002,
    base_fpn_col=0.002,
    base_read_noise=0.002,  # low
    base_shot_noise=0.01,  # medium
    denoise_strength0.6,
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
    denoise_strength0.4,
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


def get_camera_profile(kind: CameraKind | str) -> CameraProfile:
    """Return camera profile for a given camera kind."""
    key = str(kind).strip().lower()
    camera_profile = CAMERA_PROFILES.get(key)
    if camera_profile is None:
        raise ValueError(f"Unknown camera kind: {kind!r}")
    return camera_profile


# ---------------------------------------------------------------------------
# Utility conversion
# ---------------------------------------------------------------------------

def uint8_from_float32(img_f: ImageRGBF | ImageBGRF) -> ImageRGB | ImageBGR:
    """Convert RGB/BGR float32 in [0, 1] to RGB/BGR uint8 with rounding."""
    return (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


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
                  borderMode=cv2.BORDER_REFLECT,
             )
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

    bgr_dm = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2BGR)
    rgb_dm = cv2.cvtColor(bgr_dm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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
        iso_factor = float(abs(iso_level))
    else:
        key = str(iso_level).strip().lower()
        iso_factor = ISO_SF.get(key, 1.0)

    if iso_factor < EPSILON:
        # All noise effectively off
        return img

    img = np.clip(img.astype(np.float32), 0.0, 1.0)

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
        shot_std = shot_sigma * np.sqrt(np.clip(base, 0.0, 1.0))
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
    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ISP Denoising & Sharpening
# ---------------------------------------------------------------------------

def isp_denoise_and_sharpen(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """
    ISP processing stage: denoising (bilateral) + unsharp-mask sharpening.

    Controls:
        denoise_strength: 0..1 – more = stronger noise reduction & smoothing
        blur_sigma:       0.5..3 – radius of unsharp-mask blur kernel
        sharpening_amount:0..1 – strength of sharpening (halos)
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
        return np.clip(img_f, 0.0, 1.0)

    # Gaussian blur radius (sigma)
    blur_sigma = max(0.1, float(profile.blur_sigma))

    # Unsharp mask
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=blur_sigma)
    sharp = img_f + sharp_amt * (img_f - blur)

    return np.clip(sharp, 0.0, 1.0)


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
    tone = np.clip(tone, 0.0, 1.0)
    return img * (1.0 - strength) + tone * strength


def tone_scurve(img: ImageRGBF, strength: float) -> ImageRGBF:
    """Local contrast S-curve using tanh. Gives typical 'HDR' pop."""
    if strength <= EPSILON:
        return img
    x = img * 2.0 - 1.0
    y = np.tanh(x * (1.0 + strength * 2.0))
    return np.clip((y + 1.0) * 0.5, 0.0, 1.0)


def tone_mapping(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Combined tone-mapping stage."""
    tone_strength = profile.tone_strength
    scurve_strength = profile.scurve_strength
    tone_mode = profile.scurve_strength
    if tone_mode == "filmic":
        img = tone_map_filmic(img, tone_strength)
    else:
        img = tone_map_reinhard(img, tone_strength)

    img = tone_scurve(img, scurve_strength)
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Vignette + color warmth
# ---------------------------------------------------------------------------

def vignette_and_color(img: ImageRGBF, profile: CameraProfile) -> ImageRGBF:
    """Apply vignetting and mild color bias in RGB float space."""
    h, w, _ = img.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / (r.max() + EPSILON)

    vig_strength = max(profile.vignette_strength, 0.0)
    vign = 1.0 - vig_strength * (r_norm ** 2)
    vign = vign.astype(np.float32)[..., None]

    img_v = img * vign

    warm = profile.color_warmth
    if abs(warm) > EPSILON:
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
        camera_kind: CameraKind = "smartphone",
        iso_level: ISOLevel | Real = "mid",
        lens_k1: float = -0.15,
        lens_k2: float = 0.02,
        rolling_strength: float = 0.03,
        apply_jpeg: bool = True,
        rng: np.random.Generator = None,
    ) -> ImageRGBF:
    """Camera-like correction engine.

    Args:
        img: Input RGB float image in [0, 1].
        camera_kind: 'smartphone' or 'compact'.
        iso_level: 'low', 'mid', 'high' or a numeric ISO factor.
        lens_k1: Primary radial distortion coefficient.
        lens_k2: Secondary radial distortion coefficient.
        rolling_strength: Skew magnitude, e.g. ~0.02-0.05.
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

    # 4.5) Tone mapping
    img = tone_mapping(
        img,
        base_strength=profile.tone_strength,
        scurve_strength=profile.scurve_strength,
        mode=profile.tone_mode,
    )

    # 5) Vignette + color bias
    img = vignette_and_color(img, profile=profile)

    # 6) Optional JPEG roundtrip
    if apply_jpeg:
        img = jpeg_roundtrip(img, quality=profile.jpeg_quality)

    return np.clip(img, 0.0, 1.0)
