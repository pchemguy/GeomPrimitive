"""
synthetic_photo_generator_physical.py
-------------------------------------
Physically ordered synthetic "lab-style photo" generator.

Forward model:
  Scene surface -> (illumination + paper texture + noise) -> Lens system
  -> (perspective + radial distortion) -> Sensor aperture (vignetting, chromatic)

Pipeline summary:
  1.  render_scene()           - ideal geometry (grid, primitives)
  2.  apply_paper_lighting()   - simulate uneven lab illumination on paper
  3.  add_paper_texture()      - fiber pattern on paper surface
  4.  apply_sensor_noise()     - sensor/film noise before optics
  5.  apply_camera_effects()   - optical projection + lens distortion
  6.  add_vignette_and_color_shift() - post-lens effects (vignetting, chromatic)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import util, exposure, filters
from skimage.util import random_noise


# =====================================================================
# 1. Scene renderer (flat ideal paper)
# =====================================================================
def render_scene(width_mm=100, height_mm=80, dpi=200):
  """Render ideal undistorted grid with primitives (mm paper)."""
  fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
  ax.set_xlim(0, width_mm)
  ax.set_ylim(0, height_mm)
  ax.set_aspect("equal")
  ax.axis("off")

  # Grid
  for x in np.arange(0, width_mm + 1, 1):
    ax.axvline(x, color="gray", lw=0.2, alpha=0.5)
  for y in np.arange(0, height_mm + 1, 1):
    ax.axhline(y, color="gray", lw=0.2, alpha=0.5)
  for x in np.arange(0, width_mm + 1, 10):
    ax.axvline(x, color="gray", lw=0.6, alpha=0.7)
  for y in np.arange(0, height_mm + 1, 10):
    ax.axhline(y, color="gray", lw=0.6, alpha=0.7)

  # Primitives
  ax.add_patch(plt.Rectangle((30, 30), 20, 20, facecolor="none",
                             edgecolor="red", lw=2))
  ax.add_patch(plt.Circle((60, 40), 10, facecolor="none",
                          edgecolor="blue", lw=2))
  tri = np.array([[10, 10], [20, 10], [15, 25]])
  ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
  plt.close(fig)
  return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# =====================================================================
# 2. Illumination gradient (pre-optical)
# =====================================================================
def apply_paper_lighting(img, top_bright=1.15, bottom_dark=0.85):
  """Simulate overhead light gradient hitting the paper before optics."""
  h, w = img.shape[:2]
  grad = np.linspace(top_bright, bottom_dark, h, dtype=np.float32).reshape(h, 1)
  grad = np.repeat(grad, w, axis=1)
  lit = np.clip(img.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)
  return lit


# =====================================================================
# 3. Paper texture (pre-optical)
# =====================================================================
def add_paper_texture(img, strength=0.15, scale=8.0):
  """Apply multiplicative paper fiber texture (before distortion)."""
  h, w = img.shape[:2]
  noise_field = np.random.randn(h, w).astype(np.float32)
  noise_field = cv2.GaussianBlur(noise_field, ksize=(0, 0), sigmaX=scale)
  # Normalize for NumPy 2.0+
  noise_min = float(noise_field.min())
  noise_ptp = float(np.ptp(noise_field)) + 1e-9
  noise_norm = (noise_field - noise_min) / noise_ptp
  texture = 1.0 + strength * (noise_norm - 0.5)
  return np.clip(img.astype(np.float32) * texture[..., None],
                 0, 255).astype(np.uint8)


# =====================================================================
# 4. Sensor noise / blur (pre-optical)
# =====================================================================
def apply_sensor_noise(img, gaussian_std=0.02, poisson=True, blur_sigma=0.8):
  """Simulate sensor or film noise on the paper image before optics."""
  img_f = util.img_as_float(img)
  noisy = random_noise(img_f, mode="gaussian", var=gaussian_std**2)
  if poisson:
    noisy = random_noise(noisy, mode="poisson")
  blurred = filters.gaussian(noisy, sigma=blur_sigma, channel_axis=2)
  return util.img_as_ubyte(exposure.rescale_intensity(blurred))


# =====================================================================
# 5. Optical effects (perspective + lens distortion)
# =====================================================================
def apply_camera_effects(img, tilt_x=0.18, tilt_y=0.1, k1=-0.25, k2=0.05):
  """Warp scene by camera perspective and radial lens distortion."""
  h, w = img.shape[:2]

  # Perspective
  dx, dy = tilt_x * w, tilt_y * h
  src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
  dst = np.float32([[dx, dy], [w - dx, dy / 2], [w, h], [0, h - dy]])
  H = cv2.getPerspectiveTransform(src, dst)
  persp = cv2.warpPerspective(img, H, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

  # Lens distortion
  cx, cy = w / 2, h / 2
  r_norm = max(cx, cy)
  xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
  x_d = (xx - cx) / r_norm
  y_d = (yy - cy) / r_norm
  r2 = x_d**2 + y_d**2
  factor = 1 + k1 * r2 + k2 * r2**2
  factor = np.where(factor == 0, 1, factor)
  x_u = x_d / factor
  y_u = y_d / factor
  map_x = (x_u * r_norm + cx).astype(np.float32)
  map_y = (y_u * r_norm + cy).astype(np.float32)
  distorted = cv2.remap(persp, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),   # white paper
  )
  return distorted


# =====================================================================
# 6. Post-optical vignetting + chromatic shift
# =====================================================================
def add_vignette_and_color_shift(img,
                                 vignette_strength=0.35,
                                 warm_strength=0.10):
  """Add post-lens vignetting and edge warmth."""
  h, w = img.shape[:2]
  cx, cy = w / 2.0, h / 2.0
  xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
  rx = (xx - cx) / cx
  ry = (yy - cy) / cy
  r = np.sqrt(rx * rx + ry * ry)
  r_norm = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

  vignette_mask = 1.0 - vignette_strength * (r_norm**2)
  warm_mask = 1.0 + warm_strength * (r_norm**1.2)
  cool_mask = 1.0 - 0.5 * warm_strength * (r_norm**1.2)

  img_f = img.astype(np.float32)
  img_f *= vignette_mask[..., None]
  img_f[..., 0] *= cool_mask   # B down
  img_f[..., 2] *= warm_mask   # R up
  return np.clip(img_f, 0, 255).astype(np.uint8)


# =====================================================================
# 7. Full pipeline (physically ordered)
# =====================================================================
def generate_synthetic_photo():
  """Run full forward model."""
  # --- Scene / surface domain ---
  base = render_scene()
  lit = apply_paper_lighting(base)        # illumination before optics
  textured = add_paper_texture(lit)       # surface texture
  sensor = apply_sensor_noise(textured)   # noise, blur (still planar)

  # --- Optical domain ---
  optical = apply_camera_effects(sensor)  # projection + lens distortion

  # --- Camera domain (after optics) ---
  final = add_vignette_and_color_shift(optical)
  return final


if __name__ == "__main__":
  img = generate_synthetic_photo()
  cv2.imshow("Synthetic Lab Photo (Physically Ordered)", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
