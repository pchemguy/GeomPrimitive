"""
spt_correction_engine.py
-----------

## Camera-like “Correction Engine” - concrete implementation

Self-contained Correction Engine:

 - RGB float [0,1] in/out

simulates:  
 1. lens distortion (radial) + optional rolling shutter skew
 2. CFA + demosaicing artifacts (using OpenCV Bayer demosaicing)
 
 3. PRNU and row/column FPN + ISO adaptive noise shaping
 4. brightness- & ISO-dependent shot/read noise

 5. ISP-style denoise + sharpening
 7. vignette & slight color bias (smartphone / lab camera)
 8. JPEG roundtrip (mild compression, configurable)


### Correction engine steps:

1. **Lens Model Injection**  
    * chromatic aberration
    * realistic PSF
    * slight field curvature blur
    * rolling shutter skew
2. **CFA & Demosaicing Simulation**
    * convert to virtual-Bayer RAW
    * apply CFA
    * demosaic using bilinear or VNG
    * introduces CFA artifacts

3. **PRNU & Fixed-Pattern Noise + ISO adaptive noise shaping**
    * row/column banding
    * pixel response non-uniformity
    * noise greater in shadows
    * noise smaller in highlights
4. **Shot + Read Noise Model**
    * brightness-dependent Poisson
    * Gaussian read noise
    * color-correlated noise

5. **ISP Denoiser Simulation**
    * wavelet-like
    * patch-dependent
6. **Sharpening + Halos**
    * DOG-based halo
    * edge overshoot/undershoot
7. **Vignette & Tone Curve**
    * highlight rolloff
    * mild channel color shift
8. **JPEG Simulation**
    * DCT quantization
    * block grid
    * subtle quantization ghosts
9. **Metadata removal or injection** (optional)

"""

from __future__ import annotations

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from types import MappingProxyType

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

from mpl_utils import ImageBGR, ImageBGRF, ImageRGB, ImageRGBF, ImageRGBA


ISOLevel = Literal["low", "mid", "high"]
ISO_SF: dict[ISOLevel, float] = {
    "low"  : 0.6, 
    "mid"  : 1.0, 
    "high" : 1.6,
}

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
    vignette_strength=0.3,
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
    vignette_strength=0.2,
    color_warmth=0.05,
    jpeg_quality=90,
)


CAMERA_PROFILES: dict[CameraKind, CameraProfile] = {
    "smartphone": SMARTPHONE_PROFILE,
    "compact"   : COMPACT_PROFILE,
}


def get_camera_profile(kind: CameraKind) -> CameraProfile:
    camera_profile = CAMERA_PROFILES.get(kind.strip().lower())
    if camera_profile is None
        raise ValueError(f"Unknown camera kind: {kind}")
    return camera_profile


def uint8_from_float32(img_f: ImageRGBF | ImageBGRF) -> ImageRGB | ImageBGR:
    """Convert RGB/BGR float32 in [0, 1] to RGB/BGR uint8."""
    return (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lens Model Injection: radial distortion + rolling shutter
# ---------------------------------------------------------------------------
def apply_radial_distortion(
        img: ImageRGBF,
        k1: float,
        k2: float = 0.0,
    ) -> ImageRGBF:
    """Apply simple radial lens distortion in RGB float space.
  
    Args:
        img: Input image in RGB float [0, 1], shape (H, W, 3).
        k1: Quadratic radial distortion coefficient.
        k2: Quartic radial distortion coefficient.
  
    Returns:
        Distorted image, RGB float [0, 1].
    """
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w))
    x = (xx - w / 2) / (w / 2)
    y = (yy - h / 2) / (h / 2)
    r2 = x * x + y * y

    radial = 1 + k1 * r2 + k2 * r2 * r2
    x_dist = x * radial
    y_dist = y * radial

    map_x = (x_dist * (w / 2) + w / 2).astype(np.float32)
    map_y = (y_dist * (h / 2) + h / 2).astype(np.float32)
  
    # remap expects BGR/whatever, but we only care about spatial mapping
    out = cv2.remap((img * 255.0).astype(np.uint8), map_x, map_y,
               interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
          ).astype(np.float32) / 255.0

    return out


def apply_rolling_shutter_skew(
        img: ImageRGBF,
        strength: float = 0.0,
    ) -> ImageRGBF:
    """Apply simple rolling-shutter-like skew in RGB float.
    
    Args:
        img: RGB float image in [0, 1].
        strength: Horizontal skew fraction, e.g. 0.03. Positive -> skew right.
    
    Returns:
        Skewed RGB float image.
    """
    if abs(strength) < 1e-5:
        return img

    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for y in range(h):
        shift = int(strength * (y / h) * w)
        out[y] = np.roll(img[y], shift, axis=0)
    return out


# ---------------------------------------------------------------------------
# CFA & Demosaicing
# ---------------------------------------------------------------------------
def simulate_cfa_and_demosaic(img: ImageRGBF) -> ImageRGBF:
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
    return rgb_dm
