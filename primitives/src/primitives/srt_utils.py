"""
srt_utils.py
------------

https://chatgpt.com/c/690af605-a038-832a-aff2-ab08190adb02

NOTE: Early not functioning draft.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Optional, Union, Tuple


RNGType = Union[random.Random, np.random.Generator]
numeric = Union[int, float]
PointXY = tuple[numeric, numeric]
CoordRange = tuple[numeric, numeric]


def random_srt_params(bbox_x1x2: CoordRange, bbox_y1y2: CoordRange,
                      rot_range: Tuple[float, float] = (-15, 15),
                      trans_range: Tuple[float, float] = (-0.1, 0.1),
                      rng: Optional[RNGType] = random.Random(),
                     ) -> Tuple[float, float, float, float, np.ndarray]:
    """Generate random scale, rotation, translation, and affine matrix for unit box.
    
    The source is assumed to be the unit box ([-1, 1]).
    """
    s = rng.uniform(*scale_range)
    theta = np.deg2rad(rng.uniform(*rot_range))
    tx, ty = rng.uniform(*trans_range, size=2)
    mat = _affine_matrix(s, theta, tx, ty)
    return s, theta, tx, ty, mat


def affine_matrix(sf: float, theta: float, tx: float, ty: float) -> np.ndarray:
    """Return a 3x3 homogeneous transform matrix."""
    c, sn = np.cos(theta), np.sin(theta)
    return np.array(
        [[sf * c,  -sf * sn, tx],
         [sf * sn,  sf * c,  ty],
         [0.0,      0.0      1.0]],
        dtype=np.float32
    )


def apply_srt(
    path: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    """Apply the affine transform to a set of 2D points."""
    pts = np.c_[path, np.ones(len(path))]
    return (mat @ pts.T).T[:, :2]
