"""
fqc.py
----------------

Complete forensic evaluation suite for distinguishing synthetic images
from real laboratory photos or scans using reference-free methods.

Focus:
  - No metadata or RAW processing
  - Works on:
        - file path (PNG or JPEG)
        - OpenCV BGR ndarray
  - Provides:
        - noise residual & PRNU-like metrics
        - per-channel noise statistics
        - noise spectral slope (1/f^beta)
        - FFT radial / angular energy profiles
        - JPEG block grid detection for any input
        - patch-based variance + kurtosis maps
        - edge profile consistency metrics
        - row/column banding detection
        - color correlation statistics
        - local self-consistency tests

Dependencies:
    numpy
    opencv-python
    scipy
    scikit-image

All output is returned as a Python dict.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
from skimage.filters import sobel, scharr, laplace
from skimage.util import view_as_blocks, img_as_float
from skimage.restoration import estimate_sigma


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _load_image(img_input: Any) -> np.ndarray:
    """
    Accepts:
        - path to file (jpg/png)
        - OpenCV ndarray (BGR)
    Returns:
        float32 RGB in [0,1]
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(img_input)
        img = cv2.imread(img_input, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_input}")
    else:
        # assume OpenCV BGR ndarray
        img = img_input

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_as_float(img.astype(np.float32))


# ---------------------------------------------------------------------------
# A. Noise Residual / PRNU-like analysis
# ---------------------------------------------------------------------------

def analyze_noise_residual(img: np.ndarray) -> Dict[str, Any]:
    """
    Extracts a denoised baseline and computes noise residual.
    Provides PRNU-like metrics: correlation, RMS, energy spectrum.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = img_as_float(gray)

    # Baseline denoising (acts like crude wavelet denoise)
    denoised = gaussian_filter(gray_f, sigma=1.0)
    residual = gray_f - denoised

    rms = float(np.sqrt(np.mean(residual**2)))

    # Correlation of residual with itself shifted (quick PRNU proxy)
    shifted = np.roll(residual, 1, axis=1)
    prnu_corr = float(np.corrcoef(residual.flatten(), shifted.flatten())[0, 1])

    # Spectral slope estimation
    R = fftshift(fft2(residual))
    mag = np.abs(R)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    radial_profile = []
    for radius in range(1, min(cy, cx)):
        mask = (r >= radius) & (r < radius + 1)
        radial_profile.append(np.mean(mag[mask]))
    radial_profile = np.array(radial_profile)

    return {
        "rms_noise": rms,
        "prnu_corr": prnu_corr,
        "noise_residual": residual,
        "fft_radial_profile": radial_profile,
    }


# ---------------------------------------------------------------------------
# B. JPEG grid artifact analysis
# ---------------------------------------------------------------------------

def analyze_jpeg_artifacts(img: np.ndarray) -> Dict[str, Any]:
    """
    Robust JPEG-grid artifact detection.
    Works even when image dims are not divisible by 8.
    Pads reflectively so view_as_blocks() always works.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Need at least 16x16 to say anything useful
    if h < 16 or w < 16:
        return {
            "jpeg_blockiness": None,
            "block_var_map": None,
            "note": "image too small for JPEG artifact analysis",
        }

    # Padding to nearest multiple of 8
    pad_h = (8 - h % 8) if (h % 8) else 0
    pad_w = (8 - w % 8) if (w % 8) else 0

    if pad_h or pad_w:
        gray_padded = np.pad(
            gray,
            ((0, pad_h), (0, pad_w)),
            mode="reflect"
        )
    else:
        gray_padded = gray

    # Now safe for block slicing
    blocks = view_as_blocks(gray_padded, block_shape=(8, 8))
    variances = np.var(blocks, axis=(2, 3))

    # Differences across block grid lines
    vert_diff = np.mean(np.abs(np.diff(variances, axis=1)))
    horiz_diff = np.mean(np.abs(np.diff(variances, axis=0)))
    blockiness = float(vert_diff + horiz_diff)

    return {
        "jpeg_blockiness": blockiness,
        "block_var_map": variances,
        "note": f"padded h={pad_h}, w={pad_w}" if (pad_h or pad_w) else "no padding",
    }


# ---------------------------------------------------------------------------
# C. FFT-based structural analysis
# ---------------------------------------------------------------------------

def analyze_fft_structure(img: np.ndarray) -> Dict[str, Any]:
    """
    Full-image FFT: radial + angular profiles.
    Good for detecting synthetic smoothness, missing high-frequency content,
    or unnatural symmetry.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    F = fftshift(fft2(gray))
    mag = np.abs(F)

    # Radial profile (already computed similarly above)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(np.int32)
    radial = np.bincount(r.ravel(), mag.ravel())[:min(cy, cx)]

    # Angular profile (0-360 degrees)
    theta = np.degrees(np.arctan2(yy - cy, xx - cx))
    theta_idx = ((theta + 360) % 360).astype(np.int32)
    angular = np.bincount(theta_idx.ravel(), mag.ravel())
    angular = angular[:360]

    return {
        "fft_magnitude": mag,
        "fft_radial": radial,
        "fft_angular": angular,
    }


# ---------------------------------------------------------------------------
# D. Patch variance, kurtosis, local statistics
# ---------------------------------------------------------------------------

def analyze_patch_stats(img: np.ndarray, patch_size: int = 16) -> Dict[str, Any]:
    """Patch-level variance and kurtosis maps."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ph, pw = patch_size, patch_size

    # Crop to patch-grid
    h2 = (h // ph) * ph
    w2 = (w // pw) * pw
    gray = gray[:h2, :w2]

    blocks = view_as_blocks(gray, block_shape=(ph, pw))
    var_map = np.var(blocks, axis=(2, 3))

    # Excess kurtosis
    mean = np.mean(blocks, axis=(2, 3), keepdims=True)
    diff = blocks - mean
    kurt = np.mean(diff**4, axis=(2, 3)) / (np.var(blocks, axis=(2, 3)) + 1e-8)**2
    # Subtract 3 for excess kurtosis (normal distribution baseline)
    kurtosis_map = kurt - 3.0

    return {
        "patch_variance_map": var_map,
        "patch_kurtosis_map": kurtosis_map,
    }


# ---------------------------------------------------------------------------
# Extended Patch Analysis
# ---------------------------------------------------------------------------

def analyze_patch_full(img: np.ndarray, patch_size: int = 16) -> Dict[str, Any]:
    """
    Detailed patch-level forensic analysis.
    Computes variance, kurtosis, entropy, noise estimates,
    brightness/noise correlation, patch spectral slopes, etc.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = img_as_float(gray)

    h, w = gray_f.shape
    ph = pw = patch_size

    # crop to grid
    h2 = (h // ph) * ph
    w2 = (w // pw) * pw
    gray_f = gray_f[:h2, :w2]

    # shape: (nH, nW, ph, pw)
    blocks = view_as_blocks(gray_f, block_shape=(ph, pw))
    nH, nW, _, _ = blocks.shape
    N = nH * nW
    B = blocks.reshape(N, ph, pw)

    # Patch-level metrics
    patch_mean = B.mean(axis=(1, 2))
    patch_var  = B.var(axis=(1, 2))

    # Kurtosis: (E[(x-mu)^4] / sigma^4) - 3
    patch_kurt = (((B - patch_mean[:,None,None])**4).mean(axis=(1,2)) /
                  (patch_var + 1e-8)**2) - 3.0

    # Entropy per patch

    # Patch entropy (scalar per patch)
    patch_entropy = []
    for b in B:
        hist, _ = np.histogram(b, bins=64, range=(0,1), density=True)
        hist = hist + 1e-12  # avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        patch_entropy.append(entropy)
    patch_entropy = np.array(patch_entropy)    
    
    # Quick noise estimate by high-pass filtering
    hp_kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float32)
    patch_noise = np.array([
        np.mean(np.abs(convolve2d(b, hp_kernel, mode="same")))
    for b in B])

    # Per patch FFT slope (detect unnatural textures)
    patch_fft_slope = []
    for b in B:
        F = np.abs(fftshift(fft2(b)))
        h2b, w2b = F.shape
        cy, cx = h2b//2, w2b//2
        yy, xx = np.indices(F.shape)
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2).astype(int)
        radial = np.bincount(r.ravel(), F.ravel())
        idx = np.arange(1, len(radial))
        if len(idx) < 4:
            patch_fft_slope.append(np.nan)
            continue
        log_r = np.log(idx+1)
        log_p = np.log(radial[1:len(idx)+1] + 1e-8)
        slope, _ = np.polyfit(log_r, log_p, 1)
        patch_fft_slope.append(slope)
    patch_fft_slope = np.array(patch_fft_slope)

    # Reference-free checks:
    # Real images usually have:
    # - broad distribution of patch variance
    # - correlation between brightness and noise
    # - heterogeneous kurtosis
    # - heterogeneous entropy
    # - FFT slopes with dispersion
    var_std = float(np.std(patch_var))
    kurt_std = float(np.std(patch_kurt))
    noise_std = float(np.std(patch_noise))
    entropy_std = float(np.std([np.sum(e) for e in patch_entropy]))
    fft_std = float(np.nanstd(patch_fft_slope))

    # brightness-variance correlation
    bv_corr = float(np.corrcoef(patch_mean, patch_var)[0,1]) if var_std > 0 else 0.0
    # brightness-noise correlation
    bn_corr = float(np.corrcoef(patch_mean, patch_noise)[0,1]) if noise_std > 0 else 0.0

    return {
        "patch_mean": patch_mean,
        "patch_var": patch_var,
        "patch_kurtosis": patch_kurt,
        "patch_noise": patch_noise,
        "patch_fft_slope": patch_fft_slope,
        "var_std": var_std,
        "kurt_std": kurt_std,
        "noise_std": noise_std,
        "entropy_std": entropy_std,
        "fft_std": fft_std,
        "brightness_variance_corr": bv_corr,
        "brightness_noise_corr": bn_corr,
    }


# ---------------------------------------------------------------------------
# E. Row/column banding
# ---------------------------------------------------------------------------

def analyze_banding(img: np.ndarray) -> Dict[str, Any]:
    """
    Detects row/column periodic artifacts, typical for:
        - CMOS readout
        - scanner rails
        - rolling shutter anomalies
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = img_as_float(gray)

    row_std = np.std(gray_f, axis=1)
    col_std = np.std(gray_f, axis=0)

    row_fft = np.abs(fftshift(fft2(row_std[np.newaxis, :])))
    col_fft = np.abs(fftshift(fft2(col_std[:, np.newaxis])))

    return {
        "row_std": row_std,
        "col_std": col_std,
        "row_fft": row_fft,
        "col_fft": col_fft,
    }


# ---------------------------------------------------------------------------
# F. Edge profile consistency
# ---------------------------------------------------------------------------

def analyze_edge_profiles(img: np.ndarray) -> Dict[str, Any]:
    """
    Detects unnatural smoothness or excessive sharpening on edges.
    Uses Sobel/Laplace sharpness + edge noise ratio.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = img_as_float(gray)

    sob = sobel(gray_f)
    lap = laplace(gray_f)

    edge_strength = np.mean(np.abs(sob))
    lap_energy = np.mean(np.abs(lap))

    # Estimate noise level from skimage: modern param is channel_axis
    try:
        noise_est = estimate_sigma(gray_f, channel_axis=None)
    except TypeError:
        # fallback for old versions
        noise_est = estimate_sigma(gray_f, multichannel=False)

    edge_noise_ratio = float(edge_strength / (noise_est + 1e-8))

    return {
        "edge_strength": float(edge_strength),
        "laplacian_energy": float(lap_energy),
        "edge_noise_ratio": edge_noise_ratio,
    }


# ---------------------------------------------------------------------------
# G. Color-channel correlation (synthetic images often deviate)
# ---------------------------------------------------------------------------

def analyze_color_stats(img: np.ndarray) -> Dict[str, Any]:
    """
    Computes correlation between RGB channels.
    Synthetic images often show:
        - too high or too low channel correlation
        - unrealistic chroma relationships
    """
    R = img[..., 0].ravel()
    G = img[..., 1].ravel()
    B = img[..., 2].ravel()

    rg = float(np.corrcoef(R, G)[0, 1])
    rb = float(np.corrcoef(R, B)[0, 1])
    gb = float(np.corrcoef(G, B)[0, 1])

    return {
        "corr_RG": rg,
        "corr_RB": rb,
        "corr_GB": gb,
    }


# ---------------------------------------------------------------------------
# Helper: clamp and safe linear scoring
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_band(x: float, lo: float, hi: float) -> float:
    """
    Score in [0,1] where [lo, hi] is ideal band.
    Linearly falls to 0 beyond that.
    """
    if x <= lo:
        return _clamp01((x - 0.0) / (lo - 0.0)) if lo > 0 else 0.0
    if x >= hi:
        return _clamp01((hi - x) / (hi - 0.0)) if hi > 0 else 0.0
    # inside band: best = 1
    return 1.0


# ---------------------------------------------------------------------------
# Realism scoring (reference-free, heuristic)
# ---------------------------------------------------------------------------

def compute_realism_scores(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute reference-free realism subscores and overall score [0,100].

    Heuristics are based on typical digital camera behavior, not on any
    specific dataset. This is what you can do even when neither side
    has reference images.
    """
    nr = res["noise_residual"]
    jp = res["jpeg_artifacts"]
    fft = res["fft_structure"]
    ps = res["patch_stats"]
    pf = res["patch_full"]
    band = res["banding"]
    edge = res["edge_profiles"]
    col = res["color_stats"]

    # -------------------------
    # 1) Noise & PRNU realism
    # -------------------------
    rms = float(nr["rms_noise"])
    prnu = float(nr["prnu_corr"])

    # RMS in [0.005, 0.03] is "camera-like", lower = too clean, higher = too noisy
    s_noise_rms = _score_band(rms, 0.005, 0.03)

    # PRNU correlation in [0.02, 0.06] is "reasonable sensor fingerprint"
    s_noise_prnu = _score_band(abs(prnu), 0.02, 0.06)

    s_noise = 0.6 * s_noise_rms + 0.4 * s_noise_prnu

    # -------------------------
    # 2) JPEG trace realism
    # -------------------------
    blockiness = jp.get("jpeg_blockiness", None)
    if blockiness is None:
        s_jpeg = 0.5  # unknown; neither penalize nor reward strongly
    else:
        # Very small blockiness -> maybe lossless or synthetic;
        # moderate blockiness [1,4] -> typical JPEG; huge -> overcompressed / weird
        s_jpeg = 0.7 * _score_band(blockiness, 1.0, 4.0) + 0.3 * _score_band(
            blockiness, 0.5, 6.0
        )

    # -------------------------
    # 3) Edge behavior realism
    # -------------------------
    es = float(edge["edge_strength"])
    enr = float(edge["edge_noise_ratio"])

    # Edge strength in [0.02, 0.08] is typical (too low = blurry, too high = oversharp)
    s_edge_strength = _score_band(es, 0.02, 0.08)

    # Edge/noise ratio in [2, 7] typically; too low = edges too clean, too high = noisy
    s_edge_enr = _score_band(enr, 2.0, 7.0)

    s_edges = 0.5 * s_edge_strength + 0.5 * s_edge_enr

    # -------------------------
    # 4) Banding realism
    # -------------------------
    row_std = float(band["row_std"].std())
    col_std = float(band["col_std"].std())

    # We allow both very low and moderate banding. Extreme banding is penalized.
    # Typical low-level sensor banding std ~ 0.001-0.004 in normalized space.
    s_row = _score_band(row_std, 0.0005, 0.004)
    s_col = _score_band(col_std, 0.0005, 0.004)
    s_banding = 0.5 * s_row + 0.5 * s_col

    # -------------------------
    # 5) Color coupling realism
    # -------------------------
    rg = float(col["corr_RG"])
    rb = float(col["corr_RB"])
    gb = float(col["corr_GB"])

    # Correlations ~ [0.7, 0.95] are natural; <0.6 weird; >0.98 grayscale-ish/sus
    s_rg = _score_band(abs(rg), 0.7, 0.95)
    s_rb = _score_band(abs(rb), 0.7, 0.95)
    s_gb = _score_band(abs(gb), 0.7, 0.95)

    s_color = (s_rg + s_rb + s_gb) / 3.0

    # -------------------------
    # 6) Spectral naturalness (FFT radial slope)
    # -------------------------
    radial = fft["fft_radial"]
    # Avoid zeros and center
    idx = np.arange(1, len(radial))
    vals = np.array(radial[1:], dtype=np.float64) + 1e-8
    # log-log slope
    log_r = np.log(idx + 1.0)
    log_p = np.log(vals)
    if len(log_r) >= 5:
        slope, _ = np.polyfit(log_r, log_p, 1)
        # Natural images ~ 1/f^beta with beta ~ 2 => slope ~ -2
        # Acceptable band [-1.2, -2.8]
        s_fft = _score_band(abs(slope), 1.2, 2.8)
    else:
        slope = float("nan")
        s_fft = 0.5

    # -------------------------
    # 7) Patch statistics realism (kurtosis)
    # -------------------------
    kurt_map = ps["patch_kurtosis_map"]
    mean_kurt = float(np.mean(kurt_map))
    # For roughly normal noise, excess kurtosis ~ 0; allow small deviations.
    s_kurt = _score_band(abs(mean_kurt), 0.0, 1.0)  # abs close to 0 is best

    var_std   = pf["var_std"]
    kurt_std  = pf["kurt_std"]
    noise_std = pf["noise_std"]
    ent_std   = pf["entropy_std"]
    fft_std   = pf["fft_std"]

    bv_corr   = pf["brightness_variance_corr"]
    bn_corr   = pf["brightness_noise_corr"]



    # Patch variance distribution: want VARIETY
    s_pvar = _score_band(var_std, 0.002, 0.02)
    
    # Patch kurtosis distribution: heterogeneous is real
    s_pkurt = _score_band(kurt_std, 0.05, 0.5)
    
    # Patch noise distribution: uniform noise -> synthetic
    s_pnoise = _score_band(noise_std, 0.002, 0.02)
    
    # Patch entropy distribution: real images have wide entropy range
    s_pent = _score_band(ent_std, 0.01, 0.10)
    
    # FFT slope variability: real images vary patch to patch
    s_pfft = _score_band(fft_std, 0.15, 1.0)
    
    # Brightness-variance correlation: should be negative or weakly negative
    s_bv = 1.0 - _clamp01((bv_corr + 0.2) / 0.6)  # maps negative corr -> high score
    
    # Brightness-noise correlation: strongly negative is realistic
    s_bn = 1.0 - _clamp01((bn_corr + 0.2) / 0.6)
    
    # Combine patch realism
    s_patch = (
        0.15 * s_pvar +
        0.15 * s_pkurt +
        0.15 * s_pnoise +
        0.15 * s_pent +
        0.15 * s_pfft +
        0.125 * s_bv +
        0.125 * s_bn
    )

    
    # -------------------------
    # Combine subscores
    # -------------------------
    # Weights are heuristic; tweak if needed.
    w_noise = 0.2
    w_jpeg = 0.1
    w_edges = 0.2
    w_banding = 0.1
    w_color = 0.15
    w_fft = 0.15
    w_kurt = 0.1
    w_patch = 0.25


    overall = (
        w_noise * s_noise +
        w_jpeg * s_jpeg +
        w_edges * s_edges +
        w_banding * s_banding +
        w_color * s_color +
        w_fft * s_fft +
        w_kurt * s_kurt +
        w_patch * s_patch
    )    

    return {
        "overall_score": float(overall * 100.0),
        "noise_score": float(s_noise * 100.0),
        "jpeg_score": float(s_jpeg * 100.0),
        "edge_score": float(s_edges * 100.0),
        "banding_score": float(s_banding * 100.0),
        "color_score": float(s_color * 100.0),
        "fft_score": float(s_fft * 100.0),
        "kurtosis_score": float(s_kurt * 100.0),
        "fft_slope": float(slope) if "slope" in locals() else None,
        "mean_patch_kurtosis": mean_kurt,
        "row_std_std": row_std,
        "col_std_std": col_std,
        "patch_score": float(s_patch * 100.0),
        "patch_var_std": var_std,
        "kurt_std": kurt_std,
        "patch_noise_std": noise_std,
        "patch_entropy_std": ent_std,
        "patch_fft_std": fft_std,
        "patch_brightness_variance_corr": bv_corr,
        "patch_brightness_noise_corr": bn_corr,
        
    }


# ---------------------------------------------------------------------------
# Realism classification (reference-free heuristic)
# ---------------------------------------------------------------------------

def classify_realism(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify image realism based on overall and component scores.
    Returns:
            {
                "label": "camera-like" | "ambiguous" | "likely synthetic",
                "overall_score": float,
                "flags": [list of textual reasons]
            }
    """
    overall = scores["overall_score"]
    flags = []

    # Rough thresholds
    if overall >= 70.0:
        label = "camera-like"
    elif overall >= 40.0:
        label = "ambiguous"
    else:
        label = "likely synthetic"

    # Flag low subscores
    def check(key: str, name: str, thresh: float = 50.0):
        if scores[key] < thresh:
            flags.append(f"{name} score low ({scores[key]:.1f})")

    check("noise_score", "Noise / PRNU realism")
    check("jpeg_score", "JPEG pipeline realism")
    check("edge_score", "Edge behavior realism")
    check("banding_score", "Banding realism")
    check("color_score", "Color coupling realism")
    check("fft_score", "Frequency spectrum realism")
    check("kurtosis_score", "Patch statistics realism")
    check("patch_score", "Patch-level realism")

    return {
        "label": label,
        "overall_score": overall,
        "flags": flags,
    }


def interpret_forensic_report(res: Dict[str, Any], scores: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert raw forensic metrics into a human-readable diagnostic summary.
    No baseline needed yet - uses heuristic forensic expectations.
    """
    out = []
    nr = res["noise_residual"]
    jp = res["jpeg_artifacts"]
    fft = res["fft_structure"]
    ps = res["patch_stats"]
    band = res["banding"]
    edge = res["edge_profiles"]
    col = res["color_stats"]

    # ---------------------------------------------------------------
    # Noise Residual / PRNU
    # ---------------------------------------------------------------
    rms = nr["rms_noise"]
    prnu = nr["prnu_corr"]

    if rms < 0.005:
        out.append(f"- Noise RMS extremely low ({rms:.5f}): image likely too clean / synthetic.")
    elif rms < 0.015:
        out.append(f"- Noise RMS low ({rms:.5f}): cleaner than most lab photos.")
    else:
        out.append(f"- Noise RMS moderate ({rms:.5f}): within typical lab photo range.")

    if prnu < 0.01:
        out.append(f"- PRNU correlation very low ({prnu:.4f}): typical for synthetic images.")
    elif prnu < 0.03:
        out.append(f"- PRNU weak ({prnu:.4f}): mild sensor fingerprint.")
    else:
        out.append(f"- PRNU correlation noticeable ({prnu:.4f}): real-sensor signature likely present.")

    # ---------------------------------------------------------------
    # JPEG Artifact Level
    # ---------------------------------------------------------------
    blockiness = jp.get("jpeg_blockiness", None)
    if blockiness is None:
        out.append("- JPEG blockiness: not measurable (image too small).")
    else:
        if blockiness < 1.0:
            out.append(f"- JPEG blockiness very low ({blockiness:.2f}): looks synthetic or lossless.")
        elif blockiness < 4.0:
            out.append(f"- JPEG blockiness mild ({blockiness:.2f}): consistent with light compression.")
        else:
            out.append(f"- JPEG blockiness strong ({blockiness:.2f}): typical JPEG compression grid.")

    # ---------------------------------------------------------------
    # Edge Profiles
    # ---------------------------------------------------------------
    es = edge["edge_strength"]
    enr = edge["edge_noise_ratio"]

    if es < 0.02:
        out.append(f"- Very weak edge strength ({es:.3f}): soft or blurred image.")
    elif es > 0.08:
        out.append(f"- Strong edge response ({es:.3f}): possibly oversharpened (synthetic?).")
    else:
        out.append(f"- Edge strength ({es:.3f}) within typical range.")

    if enr < 2.0:
        out.append(f"- Edge noise ratio low ({enr:.2f}): edges unusually clean (synthetic?).")
    elif enr > 7.0:
        out.append(f"- Edge noise ratio high ({enr:.2f}): noisy or oversharpened.")
    else:
        out.append(f"- Edge noise ratio ({enr:.2f}) normal.")

    # ---------------------------------------------------------------
    # Banding
    # ---------------------------------------------------------------
    row_std = band["row_std"].std()
    col_std = band["col_std"].std()

    if row_std < 0.001 and col_std < 0.001:
        out.append("- No detectable row/column banding: synthetic or high-quality camera.")
    elif row_std > 0.003 or col_std > 0.003:
        out.append("- Row/column variability present: sensor or scanner signature.")
    else:
        out.append("- Very faint banding: normal low-level sensor pattern.")

    # ---------------------------------------------------------------
    # Color channel correlation
    # ---------------------------------------------------------------
    rg = col["corr_RG"]
    rb = col["corr_RB"]
    gb = col["corr_GB"]

    if max(rg, rb, gb) > 0.98:
        out.append("- RGB channels almost perfectly correlated: synthetic grayscale-like.")
    elif min(rg, rb, gb) < 0.6:
        out.append("- Low RGB correlation: unusual / synthetic color distribution.")
    else:
        out.append("- RGB correlation normal.")

    if scores is not None:
        out.append("")
        out.append(f"Realism scores (0-100): overall={scores['overall_score']:.1f}, "
                   f"noise={scores['noise_score']:.1f}, jpeg={scores['jpeg_score']:.1f}, "
                   f"edges={scores['edge_score']:.1f}, banding={scores['banding_score']:.1f}, "
                   f"color={scores['color_score']:.1f}, fft={scores['fft_score']:.1f}, "
                   f"kurtosis={scores['kurtosis_score']:.1f}")
    
    # ---------------------------------------------------------------
    # Patch Analysis
    # ---------------------------------------------------------------
    pf = res["patch_full"]
    var_std = pf["var_std"]
    kurt_std = pf["kurt_std"]
    noise_std = pf["noise_std"]
    ent_std = pf["entropy_std"]
    fft_std = pf["fft_std"]
    bv_corr = pf["brightness_variance_corr"]
    bn_corr = pf["brightness_noise_corr"]
    
    out.append("")
    out.append("--- Patch Analysis ---")
    out.append(f"Variance std: {var_std:.5f}")
    if var_std < 0.002:
        out.append("  -> Patch variance too uniform: synthetic-like.")
    elif var_std < 0.005:
        out.append("  -> Patch variance on low side.")
    else:
        out.append("  -> Patch variance distribution realistic.")
    
    out.append(f"Kurtosis std: {kurt_std:.5f}")
    if kurt_std < 0.05:
        out.append("  -> Patch kurtosis too uniform: synthetic noise.")
    else:
        out.append("  -> Kurtosis variability plausible.")
    
    out.append(f"Noise std: {noise_std:.5f}")
    if noise_std < 0.002:
        out.append("  -> Patch noise uniform: suspicious synthetic trait.")
    else:
        out.append("  -> Noise variation looks real.")
    
    out.append(f"Entropy std: {ent_std:.5f}")
    if ent_std < 0.01:
        out.append("  -> Entropy too uniform: repeated or artificial texture.")
    else:
        out.append("  -> Good entropy diversity.")
    
    out.append(f"FFT slope std: {fft_std:.5f}")
    if fft_std < 0.15:
        out.append("  -> Patch spectral slopes too similar: synthetic or overly smooth.")
    else:
        out.append("  -> Patch spectral diversity normal.")

    out.append(f"Brightness-variance corr: {bv_corr:.3f}")
    if bv_corr > -0.05:
        out.append("  -> Missing expected negative correlation: no camera physics.")
    else:
        out.append("  -> Brightness-variance relationship realistic.")
    
    out.append(f"Brightness-noise corr: {bn_corr:.3f}")
    if bn_corr > -0.05:
        out.append("  -> Missing brightness-noise dependency: synthetic noise.")
    else:
        out.append("  -> Brightness-dependent noise looks real.")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# High-level one-shot report
# ---------------------------------------------------------------------------


def forensic_evaluate(img_input: Any) -> Dict[str, Any]:
    """
    Main entry point.
    Accepts:
        - path to PNG / JPEG
        - OpenCV BGR ndarray
    Returns:
        Dict with all forensic metrics.
    """
    img = _load_image(img_input)

    return {
        "noise_residual": analyze_noise_residual(img),
        "jpeg_artifacts": analyze_jpeg_artifacts(img),
        "fft_structure": analyze_fft_structure(img),
        "patch_stats": analyze_patch_stats(img),
        "patch_full": analyze_patch_full(img),
        "banding": analyze_banding(img),
        "edge_profiles": analyze_edge_profiles(img),
        "color_stats": analyze_color_stats(img),
    }


def generate_forensic_report(img_input: Any) -> Dict[str, Any]:
    """
    Full pipeline:
        1) forensic_evaluate
        2) compute_realism_scores
        3) classify_realism
        4) build human-readable report string

    Returns a dict:
        {
            "metrics": <raw forensic_evaluate output>,
            "scores": <subscores + overall>,
            "classification": <label + flags>,
            "text_report": <multi-line string>
        }
    """
    metrics = forensic_evaluate(img_input)
    scores = compute_realism_scores(metrics)
    classification = classify_realism(scores)

    header = [
        f"Realism classification: {classification['label']} "
        f"(overall score {classification['overall_score']:.1f}/100)"
    ]
    if classification["flags"]:
        header.append("Flags:")
        for f in classification["flags"]:
            header.append(f"  - {f}")
    header.append("")

    details = interpret_forensic_report(metrics, scores=scores)

    text_report = "\n".join(header + [details])

    return {
        "metrics": metrics,
        "scores": scores,
        "classification": classification,
        "text_report": text_report,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    rep = generate_forensic_report(path)
    print(rep["text_report"])
