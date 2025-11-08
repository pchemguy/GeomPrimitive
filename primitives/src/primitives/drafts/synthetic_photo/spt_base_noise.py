"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import util, exposure, filters
from skimage.util import random_noise
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spt_base import *


def apply_noise(img: ImageBGR,
                gaussian: float = 0.2,     # 0-1 (x10)
                poisson: bool = True,      # True / False
                sp_amount: float = 0.0,    # 0-1 (x10)
                speckle_var: float = 0.0,  # 0-1 (x10)
                blur_sigma: float = 0.8,   # 0-1 (x0.1)
               ) -> ImageBGR:
    """Apply combined sensor-domain noises and blur.

    Each noise model can be independently controlled.
    Poisson   (poisson): photon shot noise (bool)
    Gaussian  (gaussian): additive white noise [0 -> 0.05]
    Salt&Pepper (sp_amount): impulse noise fraction [0 -> 0.05]
    Speckle   (speckle_var): multiplicative variance [0 -> 0.05]
    """
    img_f = util.img_as_float(img)

    if poisson:         img_f = random_noise(img_f, mode="poisson")                            # Poisson
    if gaussian > 0:    img_f = random_noise(img_f, mode="gaussian", var=(gaussian*0.1)**2)    # Gaussian
    if sp_amount > 0:   img_f = random_noise(img_f, mode="s&p", amount=sp_amount*0.1)          # Salt & pepper
    if speckle_var > 0: img_f = random_noise(img_f, mode="speckle", var=speckle_var*0.1)       # Speckle
    if blur_sigma > 0:  img_f = filters.gaussian(img_f, sigma=blur_sigma*10, channel_axis=2)   # Blur

    return util.img_as_ubyte(exposure.rescale_intensity(img_f))


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_noise(
                               img=base_bgr,
                               gaussian = 0.2,
                               poisson = True,
                               sp_amount = 0.0,
                               speckle_var = 0.0,
                               blur_sigma = 0.8,
                           )

    noise_off = {
        "img": base_bgr,
        "poisson": False,
        "gaussian": 0,
        "sp_amount": 0,
        "speckle_var": 0,
        "blur_sigma": 0,
    }
    default_props = {
        "img": base_bgr,
        "poisson": True,
        "gaussian": 0.1,
        "sp_amount": 0.1,
        "speckle_var": 0.1,
        "blur_sigma": 0.1,
    }
    dx2 = {
        "img": base_bgr,
        "poisson": True,
        "gaussian": 0.2,
        "sp_amount": 0.2,
        "speckle_var": 0.2,
        "blur_sigma": 0.2,
    }
    dx5 = {
        "img": base_bgr,
        "poisson": True,
        "gaussian": 0.5,
        "sp_amount": 0.5,
        "speckle_var": 0.5,
        "blur_sigma": 0.5,
    }
    demos = {
        "Noise OFF": apply_noise(**noise_off),
        "Basic": apply_noise(**default_props),
        "dx2": apply_noise(**dx2),
        "dx5": apply_noise(**dx5),
    }


    show_RGBx_grid(demos, n_columns=4)


if __name__ == "__main__":
    main()




"""
        {"poisson": True,                    "blur_sigma": 0},
        {"gaussian":  0.1,                   "blur_sigma": 0},
        {"gaussian":  0.2,                   "blur_sigma": 0},
        {"gaussian":  0.4,                   "blur_sigma": 0},
        {"gaussian":  0.8,                   "blur_sigma": 0},
        {"gaussian":  0.5,                   "blur_sigma": 1},
        {"gaussian":  0.5,                   "blur_sigma": 2},
        {"gaussian":  0.5,                   "blur_sigma": 4},
        {"gaussian":  0.5,                   "blur_sigma": 8},
        {"gaussian":  1.6,                   "blur_sigma": 0},
        {"sp_amount": 0.1, "speckle_var": 8, "blur_sigma": 0},
"""
