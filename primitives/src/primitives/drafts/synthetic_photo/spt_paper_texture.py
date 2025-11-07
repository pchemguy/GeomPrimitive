"""
spt_paper_texture.py
--------------------

https://chatgpt.com/c/690ddbc5-0f1c-8328-b33b-3e9a9315d3c7

Procedural paper background generator for Matplotlib or image processing.

Creates realistic, camera-free 'scanned paper' textures with optional fibers
and specks. The output is an RGB float array in [0,1].

Dependencies: numpy, scipy, matplotlib (optional for preview).
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_paper_texture(
    shape=(1024, 1024),
    seed=None,
    base_color=(0.96, 0.95, 0.92),
    low_sigma=60,
    mid_sigmas=(8, 20),
    high_sigma=0.8,
    amp_low=0.06,
    amp_mid=0.03,
    amp_high=0.01,
    fibers=False,
    fiber_angle_deg=0,
    fiber_strength=0.02,
    specks=False,
    speck_density=0.0003,
    speck_size=(1, 4),
    speck_darkness=0.2,
):
  """Generate a flat 'paper-like' texture suitable for Matplotlib backgrounds.

  Args:
    shape: (H, W) in pixels.
    seed: RNG seed for reproducibility.
    base_color: Base RGB tone of paper.
    *_sigma: Gaussian blur radii controlling texture scales.
    *_amp: Relative contrast for each band.
    fibers: If True, add directional fibers (anisotropic grain).
    fiber_angle_deg: Direction of fiber streaks.
    fiber_strength: Amplitude of fiber contrast.
    specks: If True, add sparse darker specks.
    speck_density: Approx. fraction of pixels containing specks.
    speck_size: (min, max) Gaussian blur sigma range for specks.
    speck_darkness: Depth of darkening for specks.

  Returns:
    np.ndarray: (H, W, 3) RGB float image in [0, 1].
  """
  rng = np.random.default_rng(seed)
  H, W = shape

  # --- Low-frequency illumination
  low = gaussian_filter(rng.standard_normal((H, W)), low_sigma)
  low = (low - low.mean()) / (low.std() + 1e-8)

  # --- Mid-frequency grain
  mid = np.zeros((H, W))
  for s in mid_sigmas:
    n = gaussian_filter(rng.standard_normal((H, W)), s)
    n -= n.mean()
    mid += n
  mid /= len(mid_sigmas)
  mid -= gaussian_filter(mid, 3 * max(mid_sigmas))
  mid /= mid.std() + 1e-8

  # --- High-frequency scanner/sensor noise
  high = gaussian_filter(rng.standard_normal((H, W)), high_sigma)
  high = (high - high.mean()) / (high.std() + 1e-8)

  # --- Combine base noise bands
  texture = amp_low * low + amp_mid * mid + amp_high * high
  texture = np.clip(1 + texture, 0, 2)

  # --- Fiber texture (anisotropic blur along a direction)
  if fibers:
    # Directional kernel via anisotropic Gaussian blur
    theta = np.deg2rad(fiber_angle_deg)
    sx, sy = np.abs(np.cos(theta)) * 8 + 1, np.abs(np.sin(theta)) * 8 + 1
    fib = gaussian_filter(rng.standard_normal((H, W)), (sy, sx))
    fib -= gaussian_filter(fib, 30)  # high-pass for streak isolation
    fib /= fib.std() + 1e-8
    texture += fiber_strength * fib

  # --- Specks / recycled-paper look
  if specks:
    count = int(H * W * speck_density)
    ys = rng.integers(0, H, count)
    xs = rng.integers(0, W, count)
    mask = np.zeros((H, W), dtype=float)
    mask[ys, xs] = 1.0
    # Blur each speck with random Gaussian kernel
    sigma = rng.uniform(*speck_size)
    mask = gaussian_filter(mask, sigma)
    mask /= mask.max() + 1e-8
    texture *= (1 - speck_darkness * mask)

  # --- Apply base tone
  paper = np.empty((H, W, 3), dtype=np.float32)
  for i, c in enumerate(base_color):
    paper[..., i] = np.clip(c * texture, 0, 1)

  return paper


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  paper = generate_paper_texture(
      (800, 1000),
      seed=1234,
      fibers=True,
      fiber_angle_deg=30,
      specks=True,
  )

  fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
  ax.imshow(paper, extent=[0, 1, 0, 1], zorder=-10)
  ax.plot([0, 1], [0, 1], color="black", lw=2)
  ax.axis("off")
  plt.tight_layout()
  plt.show()
