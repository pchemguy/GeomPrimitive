"""
mpl_grid_gen_effects.py
-----------------------

Optional post-processing effects for grid line geometry and rendering.
These effects operate on Matplotlib LineCollections and/or their backing
segment arrays.

Included effects:
    1) Low-frequency distortion fields (Perlin/Simplex-like approximation)
    2) Pencil-style linewidth modulation
    3) Organic endpoint perturbation ("pen pressure" look)
    4) Subtle paper texture generation (low-contrast, multi-scale noise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from matplotlib.collections import LineCollection

# ======================================================================
# 1) LOW-FREQUENCY DISTORTION FIELD (Perlin/Simplex-like)
# ======================================================================


@dataclass(frozen=True)
class SinusoidalDistortionField:
    """Smooth pseudo-Perlin displacement field for warping grids.

    The displacement is a mean of sinusoidal modes at randomized phases
    and low spatial frequencies. It approximates Perlin/Simplex-style
    warping sufficiently well for hand-drawn grid aesthetics.

    Args:
        amplitude:
            Maximum displacement magnitude (~3s) in world units.

        min_freq, max_freq:
            Frequency bounds for sinusoidal modes (cycles per world unit).

        n_modes:
            Number of sinusoidal modes per component.

        seed:
            RNG seed for reproducible field.
    """

    amplitude: float = 0.5
    min_freq: float = 0.02
    max_freq: float = 0.08
    n_modes: int = 5
    seed: Optional[int] = None

    # Internal cached params
    def _params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        freqs = rng.uniform(self.min_freq, self.max_freq, size=(self.n_modes, 2))
        phase_x = rng.uniform(0.0, 2 * math.pi, size=self.n_modes)
        phase_y = rng.uniform(0.0, 2 * math.pi, size=self.n_modes)
        return freqs, phase_x, phase_y

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        """Evaluate displacement field at given (x,y) points.

        Args:
            xy: array (..., 2)

        Returns:
            Array (..., 2) offsets in world coordinates.
        """
        xy = np.asarray(xy, float)
        flat = xy.reshape(-1, 2)

        freqs, px, py = self._params()

        x = flat[:, 0:1]
        y = flat[:, 1:2]

        arg = 2 * math.pi * (x * freqs[:, 0] + y * freqs[:, 1])  # (N, m)

        dx = np.sin(arg + px).mean(axis=1)
        dy = np.sin(arg + py).mean(axis=1)

        disp = np.stack([dx, dy], axis=-1) * self.amplitude
        return disp.reshape(xy.shape)


def warp_segments_with_field(
    segments: np.ndarray,
    field: SinusoidalDistortionField,
    strength: float = 1.0,
) -> np.ndarray:
    """Warp line segments by sampling a distortion field.

    Args:
        segments: array (n_segments, 2, 2)
        field: SinusoidalDistortionField
        strength: scaling factor for displacement

    Returns:
        Warped segment array of same shape.
    """
    segs = np.asarray(segments, float)
    pts = segs.reshape(-1, 2)
    pts_w = pts + field.displacement(pts) * strength
    return pts_w.reshape(segs.shape)


def warp_line_collection(
    lc: LineCollection,
    field: SinusoidalDistortionField,
    strength: float = 1.0,
) -> None:
    """Warp an existing LineCollection in-place."""
    segs = lc.get_segments()
    warped = warp_segments_with_field(segs, field, strength)
    lc.set_segments(warped)


# ======================================================================
# 2) PENCIL-SKETCH LINEWIDTH MODULATION
# ======================================================================


def apply_pencil_linewidths(
    lc: LineCollection,
    base_width: float = 1.5,
    jitter_fraction: float = 0.4,
    seed: Optional[int] = None,
) -> None:
    """Apply per-segment linewidth jitter to simulate pencil strokes.

    Line widths:
        w_i = base * (1 + jitter_fraction * N(0, 1/3)),
        clipped to [0.1*base, +inf).

    Args:
        lc: Matplotlib LineCollection
        base_width: nominal line width
        jitter_fraction: relative jitter strength
        seed: RNG seed
    """
    segs = lc.get_segments()
    n = len(segs)
    if n == 0:
        return

    rng = np.random.default_rng(seed)

    if jitter_fraction <= 0.0:
        widths = np.full(n, float(base_width))
    else:
        noise = rng.normal(loc=0.0, scale=1.0 / 3.0, size=n)
        widths = base_width * (1.0 + jitter_fraction * noise)
        widths = np.clip(widths, 0.1 * base_width, None)

    lc.set_linewidths(widths)


# ======================================================================
# 3) ENDPOINT MICRO-PERTURBATION ("pen pressure")
# ======================================================================


def perturb_segment_endpoints(
    segments: np.ndarray,
    max_offset: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Perturb endpoints along the normal to the segment.

    Args:
        segments: array (n, 2, 2)
        max_offset: 3s bound for offset magnitude
        seed: RNG seed

    Returns:
        New segment array.
    """
    segs = np.asarray(segments, dtype=float)
    n = len(segs)
    if n == 0 or max_offset <= 0:
        return segs.copy()

    rng = np.random.default_rng(seed)

    p1 = segs[:, 0, :]
    p2 = segs[:, 1, :]

    v = p2 - p1
    L = np.linalg.norm(v, axis=1, keepdims=True)
    L = np.maximum(L, 1e-12)

    normals = np.stack([-v[:, 1], v[:, 0]], axis=1) / L

    sigma = max_offset / 3.0
    off1 = rng.normal(0.0, sigma, size=(n, 1))
    off2 = rng.normal(0.0, sigma, size=(n, 1))

    p1p = p1 + off1 * normals
    p2p = p2 + off2 * normals

    out = segs.copy()
    out[:, 0, :] = p1p
    out[:, 1, :] = p2p
    return out


def perturb_line_collection(
    lc: LineCollection,
    max_offset: float = 0.15,
    seed: Optional[int] = None,
) -> None:
    """Apply endpoint micro-perturbation in-place."""
    segs = lc.get_segments()
    lc.set_segments(perturb_segment_endpoints(segs, max_offset, seed))


# ======================================================================
# 4) PAPER TEXTURE (fast, subtle, low-contrast)
# ======================================================================


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Cheap separable box filter using cumulative sums."""
    if radius <= 0:
        return img

    h, w = img.shape
    pad = radius

    # Horizontal blur
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    csum = np.cumsum(tmp, axis=1)
    left = csum[:, :-2 * pad]
    right = csum[:, 2 * pad :]
    horz = (right - left) / (2 * pad)

    # Vertical blur
    tmp = np.pad(horz, ((pad, pad), (0, 0)), mode="reflect")
    csum = np.cumsum(tmp, axis=0)
    top = csum[:-2 * pad, :]
    bot = csum[2 * pad :, :]
    vert = (bot - top) / (2 * pad)
    return vert


def generate_paper_texture(
    shape: Tuple[int, int] = (1024, 1024),
    seed: Optional[int] = None,
    base_color: Tuple[float, float, float] = (0.97, 0.97, 0.94),
    max_deviation: float = 0.05,
    n_layers: int = 3,
) -> np.ndarray:
    """Generate low-contrast procedural paper texture.

    Args:
        shape: (H, W)
        seed: RNG seed
        base_color: base RGB in [0,1]
        max_deviation: max +/- deviation from base_color
        n_layers: multi-scale noise layers

    Returns:
        Float32 array (H, W, 3) in [0,1].
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    acc = np.zeros((h, w), float)

    for i in range(n_layers):
        noise = rng.normal(0.0, 1.0, size=(h, w))
        radius = 2 ** (i + 1)
        smooth = _box_blur(noise, radius)
        acc += smooth

    acc -= acc.mean()
    acc /= (acc.std() or 1.0)
    acc *= max_deviation

    base = np.array(base_color, dtype=float).reshape(1, 1, 3)
    tex = np.clip(base + acc[..., None], 0.0, 1.0)
    return tex.astype(np.float32)


def resolve_theme_from_preset(preset_name: str) -> str:
    """Map jitter preset to the default visual theme."""
    name = preset_name.lower()

    if name in ("blueprint", "architectural_drift"):
        return "blueprint"
    if name in ("sketchy", "messy", "handwriting_synthetic"):
        return "sketchy"
    if name in ("engineering_paper", "technical"):
        return "engineering_paper"
    if name in ("printlike_subtle", "blueprint_clean"):
        return "printlike_subtle"

    return "default"


def style_lc(
    lc,
    *,
    linewidth: float | None = None,
    color: str | tuple | None = None,
    alpha: float | None = None,
    linestyle: str | None = None,
    zorder: float | None = None,
):
    """Convenience styling helper for LineCollection.

    Accepts intuitive kwargs matching matplotlib.plot() naming, converts
    them to the correct LineCollection property names, and applies them.
    """
    props = {}

    if linewidth is not None:
        props["linewidths"] = linewidth

    if color is not None:
        props["colors"] = color

    if alpha is not None:
        props["alpha"] = alpha

    if linestyle is not None:
        props["linestyles"] = linestyle

    if zorder is not None:
        props["zorder"] = zorder

    if props:
        lc.set(**props)


GRID_COLOR_PALETTES = {
    "default": {
        "major":  (0.15, 0.15, 0.15, 0.9),
        "minor":  (0.15, 0.15, 0.15, 0.4),
    },

    "blueprint": {   # classic blueprint-blue
        "major":  (0.10, 0.25, 0.70, 0.90),
        "minor":  (0.10, 0.25, 0.70, 0.45),
    },

    "engineering_paper": {   # muted graphite
        "major":  (0.00, 0.00, 0.00, 0.80),
        "minor":  (0.00, 0.00, 0.00, 0.35),
    },

    "sketchy": {    # pencil style
        "major":  (0.10, 0.10, 0.10, 0.85),
        "minor":  (0.10, 0.10, 0.10, 0.45),
    },

    "messy": {      # chaotic scribbles
        "major":  (0.05, 0.05, 0.05, 0.85),
        "minor":  (0.05, 0.05, 0.05, 0.35),
    },

    "printlike_subtle": {    # crisp laser printer
        "major":  (0.00, 0.00, 0.00, 0.95),
        "minor":  (0.00, 0.00, 0.00, 0.20),
    },

    "handwriting_synthetic": {  # notebook pencil
        "major":  (0.12, 0.12, 0.12, 0.85),
        "minor":  (0.12, 0.12, 0.12, 0.40),
    },

    "architectural_drift": {   # blueprint-like but softer
        "major":  (0.08, 0.18, 0.55, 0.90),
        "minor":  (0.08, 0.18, 0.55, 0.45),
    },
}


def apply_grid_style(
    x_major_lc,
    x_minor_lc,
    y_major_lc,
    y_minor_lc,
    *,
    style: str = "default",
    linewidth_major: float = 1.2,
    linewidth_minor: float = 0.6,
    linestyle_major: str = "-",
    linestyle_minor: str = "-",
    zorder: float = 0,
):
    """Apply a complete styling scheme to all four grid collections."""

    palette = GRID_COLOR_PALETTES.get(style, GRID_COLOR_PALETTES["default"])

    major_color = palette["major"]
    minor_color = palette["minor"]

    # Major lines
    for lc in (x_major_lc, y_major_lc):
        style_lc(
            lc,
            linewidth=linewidth_major,
            color=major_color,
            linestyle=linestyle_major,
            alpha=major_color[3],
            zorder=zorder,
        )

    # Minor lines
    for lc in (x_minor_lc, y_minor_lc):
        style_lc(
            lc,
            linewidth=linewidth_minor,
            color=minor_color,
            linestyle=linestyle_minor,
            alpha=minor_color[3],
            zorder=zorder,
        )
