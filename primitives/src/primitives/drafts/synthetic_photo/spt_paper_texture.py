"""
spt_paper_texture.py
--------------------

https://chatgpt.com/c/690ddbc5-0f1c-8328-b33b-3e9a9315d3c7

Procedural paper background generator with multiple presets.

Designed for Matplotlib figure backgrounds or synthetic data augmentation.
Implements multi-scale noise layers, optional fibers, specks, and tone control.

Dependencies: numpy, scipy (for gaussian_filter), matplotlib (optional preview).
"""

import contextlib
from functools import wraps

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Literal, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_paper_texture(
    shape: Tuple[int, int] = (1024, 1024),
    seed: Optional[int] = None,
    base_color: Tuple[float, float, float] = (0.96, 0.95, 0.92),
    low_sigma: float = 60,
    mid_sigmas: Tuple[float, ...] = (8, 20),
    high_sigma: float = 0.8,
    amp_low: float = 0.06,
    amp_mid: float = 0.03,
    amp_high: float = 0.01,
    fibers: bool = False,
    fiber_angle_deg: float = 0.0,
    fiber_strength: float = 0.02,
    specks: bool = False,
    speck_density: float = 0.0003,
    speck_size: Tuple[float, float] = (1, 4),
    speck_darkness: float = 0.2,
) -> np.ndarray:
  """Procedurally synthesize a paper-like RGB texture."""
  rng = np.random.default_rng(seed)
  H, W = shape

  # --- Low-frequency illumination
  low = gaussian_filter(rng.standard_normal((H, W)), low_sigma)
  low = (low - low.mean()) / (low.std() + 1e-8)

  # --- Mid-frequency paper grain
  mid = np.zeros((H, W))
  for s in mid_sigmas:
    n = gaussian_filter(rng.standard_normal((H, W)), s)
    n -= n.mean()
    mid += n
  mid /= len(mid_sigmas)
  mid -= gaussian_filter(mid, 3 * max(mid_sigmas))
  mid /= mid.std() + 1e-8

  # --- High-frequency sensor noise
  high = gaussian_filter(rng.standard_normal((H, W)), high_sigma)
  high = (high - high.mean()) / (high.std() + 1e-8)

  texture = amp_low * low + amp_mid * mid + amp_high * high
  texture = np.clip(1 + texture, 0, 2)

  # --- Optional fibers (anisotropic grain)
  if fibers:
    theta = np.deg2rad(fiber_angle_deg)
    sx, sy = np.abs(np.cos(theta)) * 8 + 1, np.abs(np.sin(theta)) * 8 + 1
    fib = gaussian_filter(rng.standard_normal((H, W)), (sy, sx))
    fib -= gaussian_filter(fib, 30)
    fib /= fib.std() + 1e-8
    texture += fiber_strength * fib

  # --- Optional recycled specks
  if specks:
    count = int(H * W * speck_density)
    ys = rng.integers(0, H, count)
    xs = rng.integers(0, W, count)
    mask = np.zeros((H, W), dtype=float)
    mask[ys, xs] = 1.0
    sigma = rng.uniform(*speck_size)
    mask = gaussian_filter(mask, sigma)
    mask /= mask.max() + 1e-8
    texture *= (1 - speck_darkness * mask)

  # --- Apply tone
  paper = np.empty((H, W, 3), dtype=np.float32)
  for i, c in enumerate(base_color):
    paper[..., i] = np.clip(c * texture, 0, 1)

  return paper


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
  "smooth": dict(
      base_color=(0.97, 0.97, 0.96),
      amp_low=0.04,
      amp_mid=0.015,
      amp_high=0.008,
      fibers=False,
      specks=False,
  ),
  "sketch": dict(
      base_color=(0.96, 0.95, 0.92),
      amp_low=0.06,
      amp_mid=0.035,
      fibers=True,
      fiber_angle_deg=20,
      fiber_strength=0.02,
      specks=False,
  ),
  "recycled": dict(
      base_color=(0.93, 0.92, 0.89),
      amp_low=0.05,
      amp_mid=0.03,
      fibers=False,
      specks=True,
      speck_density=0.0008,
      speck_darkness=0.3,
  ),
  "watercolor": dict(
      base_color=(0.97, 0.96, 0.91),
      amp_low=0.08,
      amp_mid=0.04,
      fibers=True,
      fiber_angle_deg=0,
      fiber_strength=0.03,
      specks=False,
  ),
  "vellum": dict(
      base_color=(0.98, 0.98, 0.96),
      amp_low=0.03,
      amp_mid=0.02,
      fibers=True,
      fiber_angle_deg=45,
      fiber_strength=0.01,
      specks=False,
  ),
  "notebook": dict(
      base_color=(0.96, 0.96, 0.94),
      amp_low=0.05,
      amp_mid=0.025,
      fibers=False,
      specks=False,
  ),
}


def generate_paper_preset(
    preset: Literal[
        "smooth", "sketch", "recycled", "watercolor", "vellum", "notebook"
    ] = "smooth",
    shape: Tuple[int, int] = (1024, 1024),
    seed: Optional[int] = None,
) -> np.ndarray:
  """Convenience wrapper selecting a preset configuration."""
  cfg = PRESETS[preset].copy()
  cfg.update(shape=shape, seed=seed)
  return generate_paper_texture(**cfg)


# ---------------------------------------------------------------------------
# Matplotlib integration utilities
# ---------------------------------------------------------------------------
def apply_paper_background(
    ax: Axes,
    preset: str = "smooth",
    seed: Optional[int] = None,
    reuse: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
  """Render a paper texture background behind a Matplotlib Axes.

  Args:
    ax: Target Matplotlib Axes.
    preset: Name of paper preset (see PRESETS).
    seed: Optional RNG seed.
    reuse: Optional pre-generated texture to reuse (avoids recomputation).
    pad: Fractional padding beyond data limits for background extent.
    alpha: Transparency factor for blending (1.0 = opaque).

  Returns:
    The generated or reused texture (np.ndarray).
  """
  # --- Determine image resolution in pixels
  fig = ax.figure
  bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  width_px = int(bbox.width * fig.dpi)
  height_px = int(bbox.height * fig.dpi)

  # --- Generate or reuse background texture
  tex = reuse if reuse is not None else generate_paper_preset(
      preset, shape=(height_px, width_px), seed=seed
  )

  # Draw the texture in axes coordinates (0�1), independent of data
  ax.imshow(
      tex,
      extent=(0, 1, 0, 1),
      transform=ax.transAxes,
      zorder=-100,
      aspect="auto",
      interpolation="bilinear",
      alpha=alpha,
    )

  ax.set_facecolor("none")  # let paper show through
  return tex


# ---------------------------------------------------------------------------
# Matplotlib global context manager
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def paper_style(
    preset: str = "smooth",
    seed: Optional[int] = None,
    alpha: float = 1.0,
    pad: float = 0.05,
):
  """Context manager applying paper backgrounds to all new Matplotlib subplots.

  Example:
      >>> with paper_style("recycled"):
      ...     fig, axs = plt.subplots(1, 2, figsize=(6, 3))
      ...     axs[0].plot(...)
      ...     axs[1].imshow(...)
      >>> plt.show()

  Args:
      preset: Paper preset name (see PRESETS).
      seed: RNG seed for deterministic texture generation.
      alpha: Transparency of paper texture.
      pad: Extra padding beyond data limits for texture extent.
  """
  original_subplots = plt.subplots

  def _patched_subplots(*args, **kwargs):
    fig, axes = original_subplots(*args, **kwargs)
    # Normalize axes into iterable
    axes_iter = np.atleast_1d(axes).ravel() if np.ndim(axes) else [axes]

    tex_cache = None
    for ax in axes_iter:
      tex_cache = apply_paper_background(
          ax,
          preset=preset,
          seed=seed,
          reuse=tex_cache,  # reuse first texture for consistency
          alpha=alpha,
      )
    return fig, axes

  plt.subplots = _patched_subplots
  try:
    yield
  finally:
    plt.subplots = original_subplots


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=150)
  for ax, name in zip(axes.ravel(), PRESETS.keys()):
    tex = generate_paper_preset(name, shape=(512, 512), seed=123)
    ax.imshow(tex, extent=[0, 1, 0, 1])
    ax.set_title(name, fontsize=9)
    ax.axis("off")
  plt.tight_layout()
  plt.show()

  # ----------------------------------------------
  
  fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

  # Apply background once
  apply_paper_background(ax, preset="recycled", seed=42)

  # Draw your plot normally
  x = np.linspace(0, 10, 100)
  ax.plot(x, np.sin(x), color="black", lw=2)
  ax.set_title("Demo on Recycled Paper", fontsize=10)
  plt.tight_layout()
  plt.show()

  with paper_style("sketch", seed=123):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    x = np.linspace(0, 10, 200)
    axs[0].plot(x, np.sin(x), color="black", lw=2)
    axs[1].plot(x, np.cos(x), color="black", lw=2)
    axs[0].set_title("Sine")
    axs[1].set_title("Cosine")
    plt.tight_layout()
  plt.show()
