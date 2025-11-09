"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
from skimage import util, exposure, filters
from skimage.util import random_noise
import cv2

import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spt_config
if __name__ == "__main__":
    spt_config.BATCH_MODE = False
else:
    if spt_config.BATCH_MODE:
        # Use a non-interactive backend (safe for multiprocessing workers)
        mpl.use("Agg")
import matplotlib.pyplot as plt

from spt_base import *


def apply_noise(img: ImageBGR,
                poisson: bool = True,      # apply Poisson photon noise
                gaussian: float = 0.2,     # normalized [0-1], - var=(sigma*0.1)^2
                sp_amount: float = 0.0,    # normalized [0-1], - amount=x*0.1
                speckle_var: float = 0.0,  # normalized [0-1], - var=x*0.1
                blur_sigma: float = 0.8,   # normalized [0-1], - sigma=x*10 px
               ) -> ImageBGR:
    """
    Apply combined sensor-domain noise and optical blur models.
    
    The algorithm converts the input image to [0, 1] float and sequentially
    applies several noise processes, each of which can be independently enabled
    or scaled:
    
      - **Poisson (shot noise):** Discrete photon-count fluctuations applied when
        `poisson=True`.
      - **Gaussian (read noise):** Additive white noise with variance
        `(gaussian * 0.1)^2`, giving a practical intensity range of 0-0.05 RMS.
      - **Salt & Pepper:** Impulse noise affecting a random fraction
        `sp_amount * 0.1` of pixels.
      - **Speckle:** Multiplicative noise of variance `speckle_var * 0.1`, modeling
        sensor gain variation.
      - **Blur:** Gaussian blur with sigma = `blur_sigma * 10` pixels applied
        per-channel to simulate optical PSF or focus spread.
    
    All noise operations are performed in the floating [0, 1] domain and the
    result is rescaled and quantized back to uint8 [0, 255].
    
    Args:
        img:         Input image in uint8 BGR format.
        poisson:     Whether to apply photon shot noise.
        gaussian:    Strength of additive white noise (0-1 - var up to 0.01).
        sp_amount:   Fraction of impulse-corrupted pixels (0-1 - up to 0.1).
        speckle_var: Multiplicative noise variance (0-1 - up to 0.1).
        blur_sigma:  Normalized Gaussian blur radius (0-1 - sigma up to 10 px).
    
    Returns:
        ImageBGR: Noisy and blurred image, same shape as input.
    """
    img_f = util.img_as_float(img)
    
    # Sequential noise composition
    if poisson:
        img_f = random_noise(img_f, mode="poisson")
    if gaussian > 0:
        img_f = random_noise(img_f, mode="gaussian", var=(gaussian * 0.1) ** 2)
    if sp_amount > 0:
        img_f = random_noise(img_f, mode="s&p", amount=sp_amount * 0.1)
    if speckle_var > 0:
        img_f = random_noise(img_f, mode="speckle", var=speckle_var * 0.1)
    if blur_sigma > 0:
        img_f = filters.gaussian(img_f, sigma=blur_sigma * 10.0, channel_axis=2)
  
    return util.img_as_ubyte(exposure.rescale_intensity(img_f))


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    proc_bgr:  ImageBGR  = apply_noise(
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
