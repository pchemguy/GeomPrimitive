"""
spt_correction_engine.py
-----------
"""

from __future__ import annotations

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

from mpl_utils import ImageBGR, ImageBGRF, ImageRGB, ImageRGBF, ImageRGBA


CameraKind = Literal["smartphone", "compact"]
ISOLevel = Literal["low", "mid", "high"]


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


# ---------------------------------------------------------------------------
# Lens distortion + rolling shutter
# ---------------------------------------------------------------------------
def apply_radial_distortion(
    img: np.ndarray,
    k1: float,
    k2: float = 0.0,
) -> np.ndarray:
    h, w = img.shape[:2]
    # Normalized coordinates: [-1,1]
    yy, xx = np.indices((h, w))
    x = (xx - w / 2) / (w / 2)
    y = (yy - h / 2) / (h / 2)
    r2 = x * x + y * y

    radial = 1 + k1 * r2 + k2 * r2 * r2
    x_dist = x * radial
    y_dist = y * radial

    map_x = (x_dist * (w / 2) + w / 2).astype(np.float32)
    map_y = (y_dist * (h / 2) + h / 2).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_rolling_shutter_skew(
    img: np.ndarray,
    strength: float = 0.0,
) -> np.ndarray:
    """Simple rolling-shutter horizontal skew.
    strength > 0 -> right skew; strength < 0 -> left."""
    if abs(strength) < 1e-4:
        return img

    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for y in range(h):
        shift = int(strength * (y / h) * w)
        out[y] = np.roll(img[y], shift, axis=0)
    return out
