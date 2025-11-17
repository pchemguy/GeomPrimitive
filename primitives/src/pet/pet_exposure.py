"""
pet_exposure.py
----------------

Advanced exposure, white-region, and WB/contrast tools for the PET pipeline.

Features:
    - Smart white-region detection using Gaussian Mixture Models (GMM)
    - Alternative KMeans-based white-region detection
    - Pseudo-HDR style exposure merge from a single SDR image
    - Detail enhancement wrapper (OpenCV)
    - Combined WB + masked auto-levels preset tuned for graph paper backgrounds
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage import exposure as sk_exposure

from pet_debug import save_pipeline_debug, overlay_mask


LOGGER_NAME = "pet"

os.environ["LOKY_EXECUTABLE"] = sys.executable
os.environ["LOKY_WORKER"]     = sys.executable
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_PICKLER"] = "pickle"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ======================================================================
# Adjust the paper mask to include the shadowed paper
# ======================================================================

def refine_paper_mask(
    mask: np.ndarray,
    min_area_frac: float = 0.01,
    close_ksize: int = 7,
) -> np.ndarray:
    """
    Refine a binary paper mask:
      - Morphologically close to fill small gaps (grid lines, noise).
      - Keep only the largest connected component (assumed paper sheet).
      - Drop tiny masks entirely.

    Args:
        mask: uint8 mask (0/255).
        min_area_frac: Minimum fraction of image area to keep; otherwise mask=0.
        close_ksize: Kernel size for morphological closing.

    Returns:
        refined uint8 mask (0/255).
    """
    log = logging.getLogger(LOGGER_NAME)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    h, w = mask.shape[:2]
    total_area = h * w

    if mask.mean() < 1:  # almost empty
        log.warning("refine_paper_mask: mask almost empty; returning original.")
        return mask

    # Morphological closing to bridge grid lines / gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (closed > 0).astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        log.warning("refine_paper_mask: no connected components; returning closed mask.")
        return closed

    # Skip label 0 (background). Pick largest region.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_idx - 1])

    frac = largest_area / float(total_area)
    log.info(
        "refine_paper_mask: largest component area=%d (%.3f of frame), labels=%d",
        largest_area,
        frac,
        num_labels - 1,
    )

    if frac < min_area_frac:
        log.warning(
            "refine_paper_mask: largest component too small (%.3f < %.3f); returning original.",
            frac,
            min_area_frac,
        )
        return closed

    refined = (labels == largest_idx).astype(np.uint8) * 255
    return refined


# ======================================================================
# 1) SMART WHITE REGION DETECTION (GMM / KMeans)
# ======================================================================

def detect_white_regions_gmm(
    img: np.ndarray,
    n_components: int = 3,
    sample_fraction: float = 0.15,
    prob_threshold: float = 0.35,
    highlight_clip: Optional[float] = 99.8,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Shadow-aware white-region detection using GMM in Lab space.
    Much more robust for graph paper with uneven lighting.

    Clusters are based mostly on (a,b), with lightly weighted L.
    Shadowed paper still falls in same cluster as bright paper.
    """
    import logging
    import numpy as np
    from sklearn.mixture import GaussianMixture
    import cv2

    log = logging.getLogger(LOGGER_NAME)
    h, w = img.shape[:2]

    # Convert to Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[...,0:1]    #  [0, 255]
    a = lab[...,1:2]    # [0,255]
    b = lab[...,2:3]

    # Highlight clipping (optional)
    if highlight_clip is not None:
        L = np.clip(L, None, np.percentile(L, highlight_clip))

    # Reduce L influence to avoid splitting bright/shadowed paper
    L_scaled = L * 0.35     # downweight brightness
    ab_scaled = (lab[...,1:3] - 128.0) * 1.75  # emphasize chroma stability

    # Combine into feature vector
    feats = np.concatenate([L_scaled, ab_scaled], axis=-1).reshape(-1, 3)

    # Subsample
    N = feats.shape[0]
    rng = np.random.default_rng(random_state)
    if 0 < sample_fraction < 1:
        sample_size = max(3000, int(N * sample_fraction))
        idx = rng.choice(N, sample_size, replace=False)
        subset = feats[idx]
    else:
        subset = feats

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    gm.fit(subset)

    # Find cluster with "paper-like" chroma: (a,b) near zero after centering
    means = gm.means_
    # paper cluster = minimal chroma distance
    chroma_dist = np.linalg.norm(means[:,1:3], axis=1)
    paper_idx = int(np.argmin(chroma_dist))

    probs = gm.predict_proba(feats)[:, paper_idx].reshape(h, w).astype(np.float32)

    mask = (probs >= prob_threshold).astype(np.uint8)*255

    info = {
        "means": means.tolist(),
        "paper_idx": paper_idx,
        "white_frac": float(mask.mean())/255.0,
    }

    log.info(f"GMM (shadow-aware): paper_idx={paper_idx}, frac={info['white_frac']:.3f}")

    return mask, probs, info


def detect_white_regions_kmeans(
    img: np.ndarray,
    k: int = 3,
    attempts: int = 3,
) -> Tuple[np.uint8, Dict]:
    """
    Simpler alternative white-region detection using KMeans over luminance.

    Args:
        img:
            BGR uint8 image.
        k:
            Number of clusters.
        attempts:
            OpenCV KMeans attempts.

    Returns:
        mask: uint8 hard mask (0 or 255) for brightest cluster.
        info: dict with cluster centers and proportion.
    """
    log = logging.getLogger(LOGGER_NAME)

    if img.dtype != np.uint8:
        raise TypeError("detect_white_regions_kmeans expects uint8 BGR input.")

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32).reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    ret, labels, centers = cv2.kmeans(
        data=L,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=attempts,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.flatten()
    brightest_idx = int(np.argmax(centers))
    mask = (labels.flatten() == brightest_idx).astype(np.uint8).reshape(h, w) * 255
    white_frac = float(mask.mean()) / 255.0

    log.info(
        "KMeans white mask: centers=%s, brightest_idx=%d, white_frac=%.3f",
        centers,
        brightest_idx,
        white_frac,
    )

    info = {
        "centers_L": centers.tolist(),
        "brightest_idx": brightest_idx,
        "white_fraction": white_frac,
    }

    return mask, info


# ======================================================================
# 2) PSEUDO-HDR EXPOSURE MERGE FROM SINGLE IMAGE
# ======================================================================

def _adjust_gamma_float(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma adjustment using skimage.exposure.adjust_gamma on float image [0,1].
    """
    img_f = img.astype(np.float32) / 255.0
    out = sk_exposure.adjust_gamma(img_f, gamma=gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def pseudo_hdr_merge_single(
    img: np.ndarray,
    gamma_dark: float = 1.4,
    gamma_bright: float = 0.7,
    weight_sigma: float = 4.0,
) -> np.ndarray:
    """
    Pseudo-HDR-like exposure merge from a single SDR image.

    Strategy:
        - Generate three exposures: darker, original, brighter.
        - Compute a "well-exposedness" weight for each exposure.
        - Fuse exposures via weighted average.

    Args:
        img:
            BGR uint8 image.
        gamma_dark:
            Gamma > 1.0 to create a darker exposure.
        gamma_bright:
            Gamma < 1.0 to create a brighter exposure.
        weight_sigma:
            Controls how strongly we prefer midtones (0.5) in weights.

    Returns:
        Fused BGR uint8 image.
    """
    log = logging.getLogger(LOGGER_NAME)

    if img.dtype != np.uint8:
        raise TypeError("pseudo_hdr_merge_single expects uint8 BGR input.")

    log.info(
        "Pseudo-HDR merge: gamma_dark=%.2f, gamma_bright=%.2f, weight_sigma=%.2f",
        gamma_dark,
        gamma_bright,
        weight_sigma,
    )

    img_dark = _adjust_gamma_float(img, gamma=gamma_dark)
    img_bright = _adjust_gamma_float(img, gamma=gamma_bright)
    img_orig = img

    exposures = [img_dark, img_orig, img_bright]

    # Compute weights based on "well-exposedness" in grayscale
    weights = []
    for ex in exposures:
        gray = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        w = np.exp(-weight_sigma * (gray - 0.5) ** 2)
        weights.append(w)

    weights = np.stack(weights, axis=-1)  # (H, W, 3)
    weight_sum = np.sum(weights, axis=-1, keepdims=True) + 1e-6
    weights_norm = weights / weight_sum

    # Weighted sum of exposures
    out_f = (
        exposures[0].astype(np.float32) * weights_norm[..., 0:1]
        + exposures[1].astype(np.float32) * weights_norm[..., 1:2]
        + exposures[2].astype(np.float32) * weights_norm[..., 2:3]
    )

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def exposure_detail_enhance(
    img: np.ndarray,
    sigma_s: float = 12.0,
    sigma_r: float = 0.2,
) -> np.ndarray:
    """
    Wrapper for OpenCV's detailEnhance for local contrast / pseudo-HDR effects.

    Args:
        img:
            BGR uint8 image.
        sigma_s:
            Range [0, 200]. Larger values mean smoother edges.
        sigma_r:
            Range [0, 1]. Larger values reduce the strength of edges.

    Returns:
        Enhanced BGR uint8 image.
    """
    log = logging.getLogger(LOGGER_NAME)

    if img.dtype != np.uint8:
        raise TypeError("exposure_detail_enhance expects uint8 BGR input.")

    log.info("Running cv2.detailEnhance: sigma_s=%.2f, sigma_r=%.3f", sigma_s, sigma_r)
    enhanced = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
    return enhanced


# ======================================================================
# 3) GRAPH-PAPER WB + MASKED AUTO-LEVELS PRESET
# ======================================================================

def whitebalance_auto_graphpaper(
    img: np.ndarray,
    mask: np.ndarray,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Diagonal white-balance tuned to graph paper, using masked white region.

    Args:
        img:
            BGR uint8 image.
        mask:
            uint8 mask where paper is 255.
        gain_min, gain_max:
            Clamp channel gains into this range for stability.

    Returns:
        balanced:
            WB corrected BGR uint8 image.
        info:
            Dict with gains and whitepoint.
    """
    log = logging.getLogger(LOGGER_NAME)

    if img.dtype != np.uint8:
        raise TypeError("whitebalance_auto_graphpaper expects uint8 BGR input.")

    idx = mask > 0
    if idx.sum() < 100:
        log.warning("WB: white mask too small; using global stats.")
        idx = np.ones(img.shape[:2], dtype=bool)

    pixels = img[idx]
    white_bgr = pixels.mean(axis=0).astype(np.float32)  # B, G, R

    target = 255.0
    gains = target / np.maximum(white_bgr, 1e-3)
    gains = np.clip(gains, gain_min, gain_max)

    log.info(
        "Graph-paper WB: white_bgr=(%.1f, %.1f, %.1f), gains=(%.3f, %.3f, %.3f)",
        white_bgr[0],
        white_bgr[1],
        white_bgr[2],
        gains[0],
        gains[1],
        gains[2],
    )

    balanced = img.astype(np.float32) * gains
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    info = {
        "white_bgr": white_bgr.tolist(),
        "gains": gains.tolist(),
    }

    return balanced, info


def auto_levels_masked(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    low_clip: float = 0.005,
    high_clip: float = 0.995,
    lo_max: float = 80.0,
    min_range: float = 30.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Photoshop-like auto-levels with optional masking and guardrails.

    - If mask is provided, we *prefer* its stats but:
        * If dynamic range of masked region is too small, fall back to global.
        * We enforce an upper bound on `lo` (lo_max) to avoid shadow crushing.

    Args:
        img: BGR uint8 image.
        mask: Optional uint8 mask; if None, use whole image.
        low_clip: Lower percentile (0.005 = 0.5%).
        high_clip: Upper percentile (0.995 = 99.5%).
        lo_max: Maximum allowed low point; anything higher gets clamped down.
        min_range: Minimum allowed (hi - lo) in masked region BEFORE stretching.

    Returns:
        out: Auto-leveled BGR uint8 image.
        info: Dict with lo/hi used and whether global fallback was used.
    """
    log = logging.getLogger(LOGGER_NAME)

    if img.dtype != np.uint8:
        raise TypeError("auto_levels_masked expects uint8 BGR input.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals_all = gray.flatten()

    # Global stats (fallback)
    lo_g = float(np.percentile(vals_all, low_clip * 100.0))
    hi_g = float(np.percentile(vals_all, high_clip * 100.0))

    use_global = False

    if mask is not None:
        m = mask > 0
        if m.sum() < 100:
            log.warning(
                "Auto-levels: mask too small; falling back to full-frame stats."
            )
            lo, hi = lo_g, hi_g
            use_global = True
        else:
            vals = gray[m]
            lo_m = float(np.percentile(vals, low_clip * 100.0))
            hi_m = float(np.percentile(vals, high_clip * 100.0))

            if (hi_m - lo_m) < min_range:
                # Masked region is too flat; global has more structure
                log.warning(
                    "Auto-levels: masked dynamic range too small (%.1f); "
                    "using global lo/hi.",
                    hi_m - lo_m,
                )
                lo, hi = lo_g, hi_g
                use_global = True
            else:
                lo, hi = lo_m, hi_m
    else:
        lo, hi = lo_g, hi_g
        use_global = True

    # Guardrail: avoid lo being extremely high (which crushes shadows)
    if lo > lo_max:
        log.info("Auto-levels: clamping lo from %.1f down to lo_max=%.1f", lo, lo_max)
        lo = lo_max

    log.info("Auto-levels masked: low=%.1f, high=%.1f (use_global=%s)", lo, hi, use_global)

    if hi - lo < 1e-6:
        log.warning("Auto-levels: degenerate range; returning original image.")
        return img.copy(), {"low": lo, "high": hi, "use_global": use_global}

    img_f = img.astype(np.float32)
    img_norm = (img_f - lo) * (255.0 / (hi - lo))
    out = np.clip(img_norm, 0, 255).astype(np.uint8)

    info = {"low": lo, "high": hi, "use_global": use_global}
    return out, info


def exposure_pipeline_graphpaper(
    img: np.ndarray,
    use_gmm: bool = True,
    detail_enhance_stage: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    High-level preset tuned for lab photos with graph paper background.

    Steps:
        1) Detect white paper region via GMM (default) or KMeans.
        2) Apply graph-paper WB using that mask.
        3) Apply masked auto-levels (paper-based).
        4) Optionally apply local detail enhancement.

    Args:
        img:
            BGR uint8 image.
        use_gmm:
            If True, use GMM-based detection; otherwise KMeans.
        detail_enhance_stage:
            If True, run cv2.detailEnhance at the end.

    Returns:
        out:
            Corrected BGR uint8 image.
        meta:
            Dict with diagnostic info from each stage.
    """
    log = logging.getLogger(LOGGER_NAME)

    if use_gmm:
        raw_mask, probs, info_gmm = detect_white_regions_gmm(img)
        mask = refine_paper_mask(raw_mask)
        meta_mask = {"kind": "gmm", **info_gmm}
    else:
        raw_mask, info_k = detect_white_regions_kmeans(img)
        mask = refine_paper_mask(raw_mask)
        probs = None
        meta_mask = {"kind": "kmeans", **info_k}
        
    wb_img, info_wb = whitebalance_auto_graphpaper(img, mask)
    lvl_img, info_lvl = auto_levels_masked(wb_img, mask=mask)

    if detail_enhance_stage:
        out = exposure_detail_enhance(lvl_img)
    else:
        out = lvl_img

    meta: Dict = {
        "mask_info": meta_mask,
        "wb_info": info_wb,
        "levels_info": info_lvl,
        "used_gmm": use_gmm,
        "detail_enhance": detail_enhance_stage,
        "has_probs": probs is not None,
    }

    log.info("Graph-paper exposure pipeline complete.")

    raw_mask, probs, info_gmm = detect_white_regions_gmm(img)
    mask = refine_paper_mask(raw_mask)

    wb_img, info_wb   = whitebalance_auto_graphpaper(img, mask)
    lvl_img, info_lvl = auto_levels_masked(wb_img, mask=mask)
    detail_img        = exposure_detail_enhance(lvl_img)

    # Visual overlays
    mask_overlay = overlay_mask(img, mask, color=(0,255,0), alpha=0.30)

    # Debug export
    stages = {
        "original"      : img,
        "mask_raw"      : raw_mask,
        "mask_refined"  : mask,
        "mask_overlay"  : mask_overlay,
        "wb"            : wb_img,
        "levels"        : lvl_img,
        "detail"        : detail_img,
    }

    save_pipeline_debug(stages, out_dir="debug_output")    
    return out, meta


