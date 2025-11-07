import numpy as np
from numpy.fft import rfft2, irfft2
from matplotlib import pyplot as plt


def fast_paper_texture(
    shape=(1024, 1024),
    seed=None,
    base_color=(0.96, 0.95, 0.92),
    roughness=1.8,
    fiber_dir=None,
    fiber_strength=0.05,
    speck_density=0.0004,
    speck_darkness=0.25,
):
  """Generate realistic paper texture using frequency-domain noise."""
  rng = np.random.default_rng(seed)
  H, W = shape

  # --- Power-law spectral falloff (1/f^roughness)
  fy = np.fft.fftfreq(H)[:, None]
  fx = np.fft.rfftfreq(W)[None, :]
  radius = np.sqrt(fx**2 + fy**2)
  radius[0, 0] = 1e-6
  falloff = 1.0 / (radius ** roughness)

  # --- Random complex spectrum
  spectrum = (rng.standard_normal((H, W//2 + 1)) +
              1j * rng.standard_normal((H, W//2 + 1)))
  spectrum *= falloff

  # --- Inverse FFT to spatial domain
  noise = np.real(irfft2(spectrum))
  noise = (noise - noise.mean()) / (noise.std() + 1e-8)

  # --- Add optional directional fiber modulation
  if fiber_dir is not None:
    theta = np.deg2rad(fiber_dir)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    freq = 50  # adjust for density of fibers
    fib = np.sin(X * np.cos(theta) * freq + Y * np.sin(theta) * freq)
    noise += fiber_strength * fib

  # --- Normalize and tint
  tex = np.clip(1 + 0.05 * noise, 0, 2)

  paper = np.empty((H, W, 3), dtype=np.float32)
  for i, c in enumerate(base_color):
    paper[..., i] = np.clip(c * tex, 0, 1)

  # --- Optional specks
  if speck_density > 0:
    count = int(H * W * speck_density)
    ys = rng.integers(0, H, count)
    xs = rng.integers(0, W, count)
    paper[ys, xs, :] *= 1 - speck_darkness

  return paper

tex = fast_paper_texture(
  (800, 1000),
  base_color=(0.98, 0.98, 0.99),
  roughness=1.2,
  fiber_dir=25,
  fiber_strength=0.05,
  speck_density=0.0004,
  speck_darkness=0.4,
)
plt.imshow(tex)
plt.axis("off")
plt.show()
