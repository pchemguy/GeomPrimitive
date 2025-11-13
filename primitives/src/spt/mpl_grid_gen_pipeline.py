"""
mpl_grid_gen_pipeline.py
------------------------

High-level aesthetic pipeline for applying visual effects to grid
LineCollections produced by mpl_grid_utils.generate_grid_collections().

The pipeline can:
    - Apply low-frequency warping (Perlin-like)
    - Apply endpoint perturbation ("pen pressure")
    - Apply pencil-style linewidth jitter
    - Apply major/minor styling
    - Add a paper texture background
    - Use named presets for complete styles

This module depends on:
    - grid_effects.py  (for primitive effects)
    - mpl_grid_utils   (only indirectly, for user-level integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from mpl_grid_gen import GridJitterConfig, generate_grid_collections
from mpl_grid_gen_effects import (
    SinusoidalDistortionField,
    warp_line_collection,
    perturb_line_collection,
    apply_pencil_linewidths,
    generate_paper_texture,
    apply_grid_style,
    resolve_theme_from_preset,
)


@dataclass
class GridEffectsPipeline:
    """
    High-level grid post-processing pipeline.

    Each step is optional. The pipeline can be applied to:
        - the segment geometry of line collections (warp, perturb)
        - styling (colors, widths, linestyles)
        - the plot background (paper texture)

    The pipeline is intentionally stateless except for preset parameters.
    """

    # --- Distortion field ---
    distortion_field: Optional[SinusoidalDistortionField] = None
    distortion_strength: float = 1.0

    # --- Endpoint jitter ---
    endpoint_offset: float = 0.0       # world units

    # --- Linewidth modulation ---
    pencil_width_major: Optional[Tuple[float, float]] = None  # (base, jitter_frac)
    pencil_width_minor: Optional[Tuple[float, float]] = None

    # --- Background (paper) ---
    paper_texture: bool = False
    paper_texture_params: Dict[str, Any] = None

    # --- Styling (major/minor colors, widths, etc.) ---
    theme: Optional[str] = None
    styling_params: Dict[str, Any] = None

    # ------------------------------------------------------------------
    # 1) FACTORY: BUILD FROM PRESET NAME
    # ------------------------------------------------------------------
    @staticmethod
    def from_preset(name: str) -> "GridEffectsPipeline":
        """
        Create a pre-tuned pipeline for a named aesthetic.
        """

        name = name.lower().strip()

        # Notebook-style, hand-drawn look
        if name == "handdrawn_notebook":
            return GridEffectsPipeline(
                distortion_field=SinusoidalDistortionField(
                    amplitude=0.35,
                    min_freq=0.01,
                    max_freq=0.05,
                    n_modes=6,
                    seed=42,
                ),
                distortion_strength=1.0,
                endpoint_offset=0.14,
                pencil_width_major=(1.8, 0.55),
                pencil_width_minor=(0.9, 0.55),
                paper_texture=True,
                paper_texture_params=dict(
                    shape=(1600, 1600),
                    seed=777,
                    base_color=(0.97, 0.97, 0.94),
                    max_deviation=0.04,
                    n_layers=3,
                ),
                theme=resolve_theme_from_preset("handwriting_synthetic"),
                styling_params=dict(
                    zorder=-1,
                    linewidth_major=1.8,
                    linewidth_minor=0.9,
                ),
            )

        # Technical / blueprint hybrid
        if name == "blueprint_technical":
            return GridEffectsPipeline(
                distortion_field=SinusoidalDistortionField(
                    amplitude=0.15,
                    min_freq=0.02,
                    max_freq=0.08,
                    n_modes=4,
                    seed=101,
                ),
                distortion_strength=0.7,
                endpoint_offset=0.04,
                pencil_width_major=(1.2, 0.2),
                pencil_width_minor=(0.6, 0.2),
                paper_texture=False,
                theme="blueprint",
                styling_params=dict(
                    zorder=-1,
                    linewidth_major=1.2,
                    linewidth_minor=0.6,
                ),
            )

        # "Perfect print" - subtle distortions only
        if name == "printlike":
            return GridEffectsPipeline(
                distortion_field=SinusoidalDistortionField(
                    amplitude=0.05,
                    min_freq=0.02,
                    max_freq=0.06,
                    n_modes=3,
                    seed=999,
                ),
                distortion_strength=0.5,
                endpoint_offset=0.02,
                pencil_width_major=(1.2, 0.05),
                pencil_width_minor=(0.6, 0.05),
                paper_texture=False,
                theme="printlike_subtle",
                styling_params=dict(
                    zorder=0,
                    linewidth_major=1.2,
                    linewidth_minor=0.6,
                ),
            )

        # Default simple aesthetic
        return GridEffectsPipeline(
            distortion_field=None,
            endpoint_offset=0.0,
            pencil_width_major=(1.4, 0.2),
            pencil_width_minor=(0.8, 0.2),
            paper_texture=False,
            theme="default",
            styling_params=dict(zorder=0),
        )

    # ------------------------------------------------------------------
    # 2) APPLY TO LINE COLLECTIONS (geometric effects)
    # ------------------------------------------------------------------
    def apply_to_collections(
        self,
        x_major: LineCollection,
        x_minor: LineCollection,
        y_major: LineCollection,
        y_minor: LineCollection,
    ) -> None:
        """Apply all enabled geometric effects to line collections."""

        families = [x_major, x_minor, y_major, y_minor]

        # 2.1 Warp field
        if self.distortion_field is not None:
            for lc in families:
                warp_line_collection(lc, self.distortion_field, self.distortion_strength)

        # 2.2 Endpoint wobble
        if self.endpoint_offset > 0:
            for lc in families:
                perturb_line_collection(lc, self.endpoint_offset)

        # 2.3 Pencil-style linewidth jitter
        if self.pencil_width_major:
            base, jitter = self.pencil_width_major
            for lc in (x_major, y_major):
                apply_pencil_linewidths(lc, base_width=base, jitter_fraction=jitter)

        if self.pencil_width_minor:
            base, jitter = self.pencil_width_minor
            for lc in (x_minor, y_minor):
                apply_pencil_linewidths(lc, base_width=base, jitter_fraction=jitter)

    # ------------------------------------------------------------------
    # 3) APPLY STYLING (colors, alpha, linewidths, zorder)
    # ------------------------------------------------------------------
    def apply_styling(
        self,
        x_major: LineCollection,
        x_minor: LineCollection,
        y_major: LineCollection,
        y_minor: LineCollection,
    ) -> None:
        """Apply major/minor styling according to the selected theme."""

        style = self.theme or "default"
        params = self.styling_params or {}

        apply_grid_style(
            x_major,
            x_minor,
            y_major,
            y_minor,
            style=style,
            **params,
        )

    # ------------------------------------------------------------------
    # 4) APPLY BACKGROUND
    # ------------------------------------------------------------------
    def apply_background(self, ax, bbox) -> None:
        """Draw paper texture on the axes behind other layers."""

        if not self.paper_texture:
            return

        params = self.paper_texture_params or {}
        tex = generate_paper_texture(**params)

        x0, y0, x1, y1 = bbox
        ax.imshow(
            tex,
            origin="lower",
            extent=(x0, x1, y0, y1),
            zorder=-10,
        )


def main():
    #from grid_effects_pipeline import GridEffectsPipeline
    
    # Step 1 - Generate grid (already available)

    # Create jitter config from preset
    preset_name = "sketchy"
    jitter = GridJitterConfig.preset(preset_name)

    bbox=(-10, -10, 10, 10)
    
    # Grid geometry
    obliquity_deg = 60.0
    rotation_deg = 20.0

    x_major_step = 3.0
    x_minor_step = 1.0
    y_major_step = 3.0
    y_minor_step = 1.0

    xM, xm, yM, ym = generate_grid_collections(
        bbox=bbox,
        obliquity_deg=obliquity_deg,
        rotation_deg=rotation_deg,
        x_major_step=x_major_step,
        x_minor_step=x_minor_step,
        y_major_step=y_major_step,
        y_minor_step=y_minor_step,
        jitter=jitter,
    )
    
    # Step 2 - Choose style pipeline
    effects = GridEffectsPipeline.from_preset("handdrawn_notebook")
    
    # Step 3 - Apply geometry effects
    effects.apply_to_collections(xM, xm, yM, ym)
    
    # Step 4 - Create figure, add background
    fig, ax = plt.subplots(figsize=(7, 7))
    effects.apply_background(ax, bbox)
    
    # Step 5 - Add styled grid
    effects.apply_styling(xM, xm, yM, ym)
    for lc in (xM, xm, yM, ym):
        ax.add_collection(lc)
    
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    


if __name__ == "__main__":
    main()
