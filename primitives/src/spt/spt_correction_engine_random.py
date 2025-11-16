"""
spt_correction_engine_random.py
--------------------------------

Ref: https://chatgpt.com/c/69172a6a-b78c-8326-b080-7b02e61b4730

Random Camera Profile Generator + Synthetic Lab Camera Wrapper
==============================================================

This module generates *physically plausible* randomized CameraProfile
objects for use with the spt_correction_engine. All parameters are sampled
within sensible ranges calibrated to real-world smartphones and compact
cameras.

Features:
- Samples *all* optical, sensor, ISP, tone-mapping, and JPEG controls.
- Produces realistic distributions that match real lab-photo statistics.
- Maintains physical interdependencies:
    * High ISO -> more noise, stronger denoise, stronger sharpening + tone
    * High distortion -> slightly stronger vignette
    * High denoise -> reduced sharpening to avoid halos
    * Smartphone profiles tend toward aggressive processing
    * Compact profiles tend toward more neutral rendering
- Enum and boolean values are sampled correctly.
- ISO sampled as continuous float in [0.5, 2.0].

---------------------------------------------------------------------

This module provides two primary high-level APIs:

1. **random_camera_profile()**
   Generates a physically plausible randomized `CameraProfile`, sampling
   all optical, sensor, ISP, tone-mapping, and JPEG parameters from
   calibrated distributions that match real smartphone/compact camera
   behavior.

2. **synthetic_lab_camera()**
   A higher-level wrapper that:
     - Generates a random camera correction_profile
     - Generates random lens parameters (with optional user overrides)
     - Applies the full correction engine (`apply_camera_model`)
     - Returns all metadata as a structured dictionary or JSON

This enables creation of large sets of synthetic laboratory images that
faithfully resemble real captured photographs.

Usage:
    from spt_correction_engine_random import random_camera_profile
    correction_profile = random_camera_profile(kind="smartphone")
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from spt_correction_engine import CameraProfile, ToneMode

__all__ = [
    "random_camera_profile",
    "random_lens_params",
    "synthetic_lab_camera",
]

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def rand_uniform(rng, a, b):
    return float(rng.uniform(a, b))

def rand_bool(rng, p_true=0.5):
    return bool(rng.random() < p_true)

def rand_enum(rng, enum_list):
    return rng.choice(enum_list)

# ---------------------------------------------------------------------------
# Ranges for all parameters (from the master table)
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "k1": (-0.20, 0.20),
    "k2": (-0.02, 0.02),
    "rolling_strength": (0.0, 0.05),

    # Sensor noise
    "prnu": (0.0, 0.010),
    "fpn_row": (0.0, 0.010),
    "fpn_col": (0.0, 0.010),
    "shot_noise": (0.0, 0.030),
    "read_noise": (0.0, 0.008),

    # ISP
    "denoise_strength": (0.0, 1.0),
    "blur_sigma": (0.5, 3.0),
    "sharpen": (0.0, 1.0),

    # Tone
    "tone_strength": (0.0, 0.80),
    "scurve_strength": (0.0, 0.40),

    # Vignette + color
    "vignette_strength": (0.0, 0.50),
    "color_warmth": (0.0, 0.20),

    # JPEG
    "jpeg_quality": (70, 98),

    # ISO
    "iso_float": (0.5, 2.0),
}

TONE_MODES = ["reinhard", "filmic"]
CAMERA_KINDS = ["smartphone", "compact"]

# ---------------------------------------------------------------------------
# Physical interaction tuning functions
# ---------------------------------------------------------------------------

def apply_physical_interactions(params, rng):
    """
    Modify sampled parameters so they exhibit realistic ISP-sensor interactions.
    This ensures distributions match real image statistics.
    """

    iso = params["iso_level"]

    # Noise scales with ISO
    params["base_prnu_strength"] *= iso
    params["base_fpn_row"] *= iso
    params["base_fpn_col"] *= iso
    params["base_shot_noise"] *= iso
    params["base_read_noise"] *= iso

    # Denoise vs ISO
    params["denoise_strength"] = np.clip(
        params["denoise_strength"] + 0.3 * (iso - 1.0), 0.0, 1.0
    )

    # Sharpen decreases if denoise is high (avoid watercolor halos)
    ds = params["denoise_strength"]
    params["sharpening_amount"] *= (1.0 - 0.4 * ds)

    # Tone mapping stronger at higher ISO
    params["tone_strength"] = np.clip(
        params["tone_strength"] + 0.2 * (iso - 1.0), 0.0, 0.8
    )

    # S-curve moderate coupling
    params["scurve_strength"] = np.clip(
        params["scurve_strength"] + 0.1 * (iso - 1.0), 0.0, 0.4
    )

    # Vignette slightly correlates with lens distortion magnitude
    k1_mag = abs(params["k1"])
    params["vignette_strength"] = np.clip(
        params["vignette_strength"] + 0.5 * k1_mag,
        0.0,
        PARAM_RANGES["vignette_strength"][1],
    )

    # Compact cameras tend to be milder overall
    if params["kind"] == "compact":
        params["tone_strength"] *= 0.8
        params["scurve_strength"] *= 0.8
        params["sharpening_amount"] *= 0.7
        params["denoise_strength"] *= 0.9

    return params

# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def random_camera_profile(kind: Optional[str] = None, seed: Optional[int] = None) -> CameraProfile:
    """
    Generate a physically plausible randomized CameraProfile for use with
    apply_camera_model().

    Args:
        kind: "smartphone" or "compact" (random if None).
        seed: Optional RNG seed.

    Returns:
        CameraProfile with randomized parameters.
    """

    rng = np.random.default_rng(seed)

    # Pick camera kind
    if kind is None:
        kind = rng.choice(CAMERA_KINDS)
    else:
        kind = str(kind).lower().strip()

    # ISO as continuous float
    iso = rand_uniform(rng, *PARAM_RANGES["iso_float"])

    # Base parameter sampling
    params = {
        "kind": kind,
        "iso_level": iso,

        # optics
        "k1": rand_uniform(rng, *PARAM_RANGES["k1"]),
        "k2": rand_uniform(rng, *PARAM_RANGES["k2"]),
        "rolling_strength": rand_uniform(rng, *PARAM_RANGES["rolling_strength"]),

        # sensor base noise
        "base_prnu_strength": rand_uniform(rng, *PARAM_RANGES["prnu"]),
        "base_fpn_row": rand_uniform(rng, *PARAM_RANGES["fpn_row"]),
        "base_fpn_col": rand_uniform(rng, *PARAM_RANGES["fpn_col"]),
        "base_shot_noise": rand_uniform(rng, *PARAM_RANGES["shot_noise"]),
        "base_read_noise": rand_uniform(rng, *PARAM_RANGES["read_noise"]),

        # ISP
        "denoise_strength": rand_uniform(rng, *PARAM_RANGES["denoise_strength"]),
        "blur_sigma": rand_uniform(rng, *PARAM_RANGES["blur_sigma"]),
        "sharpening_amount": rand_uniform(rng, *PARAM_RANGES["sharpen"]),

        # tone mapping
        "tone_strength": rand_uniform(rng, *PARAM_RANGES["tone_strength"]),
        "scurve_strength": rand_uniform(rng, *PARAM_RANGES["scurve_strength"]),
        "tone_mode": rand_enum(rng, TONE_MODES),

        # lens falloff and color
        "vignette_strength": rand_uniform(rng, *PARAM_RANGES["vignette_strength"]),
        "color_warmth": rand_uniform(rng, *PARAM_RANGES["color_warmth"]),

        # JPEG
        "jpeg_quality": int(rand_uniform(rng, *PARAM_RANGES["jpeg_quality"])),
    }

    # Apply physical dependencies
    params = apply_physical_interactions(params, rng)

    # Build CameraProfile
    prof = CameraProfile(
        kind=kind,
        base_prnu_strength=params["base_prnu_strength"],
        base_fpn_row=params["base_fpn_row"],
        base_fpn_col=params["base_fpn_col"],
        base_read_noise=params["base_read_noise"],
        base_shot_noise=params["base_shot_noise"],
        denoise_strength=params["denoise_strength"],
        blur_sigma=params["blur_sigma"],
        sharpening_amount=params["sharpening_amount"],
        tone_strength=params["tone_strength"],
        scurve_strength=params["scurve_strength"],
        tone_mode=params["tone_mode"],
        vignette_strength=params["vignette_strength"],
        color_warmth=params["color_warmth"],
        jpeg_quality=params["jpeg_quality"],
    )

    prof.iso_level = iso
  
    return prof

# ---------------------------------------------------------------------------
# Convenience: generate lens parameters separate from correction_profile
# ---------------------------------------------------------------------------

def random_lens_params(seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    return {
        "k1": rand_uniform(rng, *PARAM_RANGES["k1"]),
        "k2": rand_uniform(rng, *PARAM_RANGES["k2"]),
        "rolling_strength": rand_uniform(rng, *PARAM_RANGES["rolling_strength"]),
    }


# ---------------------------------------------------------------------------
# High-Level Synthetic Lab Camera Wrapper
# ---------------------------------------------------------------------------

# Logging setup
import logging
_logger = logging.getLogger("SPTPipeline")
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# High-Level Synthetic Lab Camera Wrapper
# ---------------------------------------------------------------------------

def synthetic_lab_camera(img, kind: str | None = None, seed: int | None = None,
                          rng: np.random.Generator | None = None,
                          metadata_format: str = "dict",
                          **lens_overrides):
    """
    High-level API: automatically generates a random camera correction_profile,
    lens parameters, and applies the correction engine.

    Args:
        img: RGB float32 image in [0,1].
        kind: Optional camera type ("smartphone" or "compact").
        seed: Optional RNG seed.
        rng: Optional NumPy Generator.
        metadata_format: "dict" (default) or "json".
        **lens_overrides: Optional overrides for k1, k2, rolling_strength.

    Returns:
        Either a dict with structured metadata or JSON string.
    """

    from spt_correction_engine import apply_camera_model

    _logger.info("Generating synthetic camera correction_profile...")

    local_rng = rng if rng is not None else np.random.default_rng(seed)
    correction_profile = random_camera_profile(kind=kind, seed=seed)
    lens_params = random_lens_params(seed=seed)

    # Overrides
    for key, val in lens_overrides.items():
        if key in lens_params:
            lens_params[key] = float(val)

    _logger.info("Applying camera model...")

    out = apply_camera_model(
        img,
        camera_kind=correction_profile.kind,
        iso_level=correction_profile.iso_level,
        lens_k1=lens_params["k1"],
        lens_k2=lens_params["k2"],
        rolling_strength=lens_params["rolling_strength"],
        apply_jpeg=True,
        rng=local_rng,
    )

    meta = {
        "iso_level": correction_profile.iso_level,
        "rolling_strength": lens_params.get("rolling_strength"),
        "camera_profile": correction_profile.__dict__,
        "lens_params": lens_params,
    }

    if metadata_format == "json":
        import json
        return out, json.dumps(meta, indent=2)

    return out, meta


if __name__ == "__main__":
    import numpy as _np
    import cv2 as _cv2
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    _outdir = _Path("_sptcam_test"); _outdir.mkdir(exist_ok=True)

    COLORS = {
        "white": (1.0,1.0,1.0),
        "cornsilk": (1.0,0.972,0.863),
        "ivory": (1.0,1.0,0.941),
        "oldlace": (0.992,0.961,0.902),
        "floralwhite": (1.0,0.98,0.94),
        "whitesmoke": (0.96,0.96,0.96),
    }

    for name, rgb in COLORS.items():
        print(f"\n=== Testing {name} ===")
        dummy = _np.zeros((256,256,3), dtype=_np.float32)
        dummy[...] = rgb

        t0 = _time.time()
        out, meta = synthetic_lab_camera(dummy, seed=123)
        dt = _time.time() - t0
        print(f"Processing time: {dt:.3f}s")

        _cv2.imwrite(str(_outdir/f"{name}_orig.jpg"), (dummy*255).astype(_np.uint8)[:,:,::-1])
        _cv2.imwrite(str(_outdir/f"{name}_proc.jpg"), (out*255).astype(_np.uint8)[:,:,::-1])

        with open(_outdir/f"{name}_meta.json","w") as f:
            _json.dump(meta,f,indent=2)

    print("\nSelf-test completed")

    # --- Forensic analysis report ---
    from fqc import generate_forensic_report  # adjust if needed

    def _patchwise_checks(image):
        H, W, _ = image.shape
        patches = []
        ps = 64
        for y in range(0, H-ps+1, ps):
            for x in range(0, W-ps+1, ps):
                patch = image[y:y+ps, x:x+ps]
                rep_patch = generate_forensic_report(patch)
                m = rep_patch["metrics"]
                patches.append(((y, x), m))
        return patches

    report_path = _outdir / "report.txt"
    with open(report_path, "w") as rep:
        rep.write("Synthetic Camera Forensic Report\n\n")
        for name in COLORS.keys():
            proc_path = _outdir / f"{name}_proc.jpg"
            rep.write(f"=== {name} ===\n")
            try:
                img = _cv2.imread(str(proc_path))[:,:,::-1].astype(_np.float32)/255.0
                rep_dict = generate_forensic_report(img)
                metrics = rep_dict["metrics"]
                rep.write("Full-image metrics:\n")
                for k, v in metrics.items():
                    rep.write(f"  {k}: {v}\n")

                rep.write("\nPatchwise metrics:\n")
                patches = _patchwise_checks(img)
                for (y, x), pm in patches:
                    rep.write(f"  Patch (y={y}, x={x}):\n")
                    for k, v in pm.items():
                        rep.write(f"    {k}: {v}\n")
            except Exception as e:
                rep.write(f"Error: {e}\n")
            rep.write("\n")
