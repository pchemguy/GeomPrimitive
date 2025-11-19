"""
pet_config.py
--------------

Central configuration object for the PET pipeline.

All modules read their parameters from this dataclass.
This allows unified tuning, reproducibility, and easy swapping of presets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PETConfig:
    """
    Master configuration for the PET graph-paper enhancement pipeline.

    Every stage (GMM, mask refinement, WB, levels, detail enhance,
    debug export) reads its tunables from this central object.

    Create and pass this object to the pipeline orchestrator:
        cfg = PETConfig()
        out, meta = run_pet_pipeline(img, cfg)
    """

    # -------------------------------------------------------------
    # White-region detection (GMM)
    # -------------------------------------------------------------
    gmm_n_components: int = 3
    gmm_sample_fraction: float = 0.15
    gmm_prob_threshold: float = 0.35
    gmm_highlight_clip: float = 99.8
    gmm_random_state: int = 0

    # -------------------------------------------------------------
    # Mask refinement
    # -------------------------------------------------------------
    mask_close_ksize: int = 7
    mask_min_area_frac: float = 0.01

    # -------------------------------------------------------------
    # White balance
    # -------------------------------------------------------------
    wb_gain_min: float = 0.5
    wb_gain_max: float = 2.0

    # -------------------------------------------------------------
    # Auto levels
    # -------------------------------------------------------------
    levels_low_clip: float = 0.005
    levels_high_clip: float = 0.995
    levels_min_range: float = 30.0
    levels_lo_max: float = 80.0

    # -------------------------------------------------------------
    # Detail enhancement
    # -------------------------------------------------------------
    detail_sigma_s: float = 12.0
    detail_sigma_r: float = 0.20
    enable_detail: bool = True

    # -------------------------------------------------------------
    # Debug output
    # -------------------------------------------------------------
    debug_enabled: bool = True
    debug_outdir: str = "debug_output"
    debug_composite_name: str = "pipeline_debug.jpg"
    debug_histograms: bool = True

    # -------------------------------------------------------------
    # Misc / Future expansions
    # -------------------------------------------------------------
    keep_intermediate: bool = False
    preset_name: Optional[str] = None
