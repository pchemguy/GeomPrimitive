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


CameraKind = Literal["smartphone", "compact"]
ISOLevel = Literal["low", "mid", "high"]


@dataclass
class CameraProfile:
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


SMARTPHONE_PROFILE = CameraProfile(
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

COMPACT_PROFILE = CameraProfile(
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

CAMERA_PROFILES = {
    "smartphone": SMARTPHONE_PROFILE,
    "compact"   : COMPACT_PROFILE,
}


def get_camera_profile(kind: CameraKind) -> CameraProfile:
    camera_profile = CAMERA_PROFILES.get(kind.strip().lower())
    if camera_profile is None
        raise ValueError(f"Unknown camera kind: {kind}")
    return camera_profile

