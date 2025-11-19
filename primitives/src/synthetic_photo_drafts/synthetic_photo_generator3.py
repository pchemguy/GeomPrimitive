"""
synthetic_photo_generator.py
--------------------------------
Physically consistent synthetic 'lab-style' photo generator:

Pipeline (your requested order):
    1. Ideal scene (graph paper + primitives, flat)
    2. Sensor space: noise + slight blur
    3. Optical space: perspective + lens distortion + lighting gradient
    4. Paper texture
    5. Vignetting + chromatic imbalance (warm edges)

Output: uint8 BGR image suitable for OpenCV pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import util, exposure, filters
from skimage.util import random_noise


# =====================================================================
# A. Ideal scene rendering (flat, undistorted)
# =====================================================================
def render_scene(width_mm: float = 100,
                 height_mm: float = 80,
                 dpi: int = 200) -> np.ndarray:
  """Render ideal graph-paper scene with geometric primitives."""
  fig, ax = plt.subplots(
      figsize=(width_mm / 25.4, height_mm / 25.4),
      dpi=dpi,
  )
  ax.set_xlim(0, width_mm)
  ax.set_ylim(0, height_mm)
  ax.set_aspect("equal")
  ax.axis("off")

  # --- Millimeter grid ---
  for x in np.arange(0, width_mm + 1, 1):
    ax.axvline(x, color="gray", lw=0.2, alpha=0.5)
  for y in np.arange(0, height_mm + 1, 1):
    ax.axhline(y, color="gray", lw=0.2, alpha=0.5)

  # Thicker lines every 10 mm
  for x in np.arange(0, width_mm + 1, 10):
    ax.axvline(x, color="gray", lw=0.6, alpha=0.7)
  for y in np.arange(0, height_mm + 1, 10):
    ax.axhline(y, color="gray", lw=0.6, alpha=0.7)

  # --- Primitives ---
  # Square
  ax.add_patch(
      plt.Rectangle(
          (30, 30),
          20,
          20,
          facecolor="none",
          edgecolor="red",
          lw=2,
      )
  )
  # Circle
  ax.add_patch(
      plt.Circle(
          (60, 40),
          10,
          facecolor="none",
          edgecolor="blue",
          lw=2,
      )
  )
  # Triangle
  tri = np.array([[10, 10], [20, 10], [15, 25]], dtype=float)
  ax.fill(
      tri[:, 0],
      tri[:, 1],
      edgecolor="green",
      fill=False,
      lw=2,
  )

  # Render to RGB (via RGBA buffer)
  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
  plt.close(fig)
  rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
  return rgb


# =====================================================================
# B. Sensor noise simulation (BEFORE optical distortions)
# =====================================================================
def apply_sensor_noise(img: np.ndarray,
                       gaussian_std: float = 0.02,
                       poisson: bool = True,
                       blur_sigma: float = 0.8) -> np.ndarray:
  """Simulate noisy, slightly defocused sensor capture."""
  img_f = util.img_as_float(img)

  # Gaussian noise
  noisy = random_noise(img_f, mode="gaussian", var=gaussian_std**2)

  # Poisson (shot) noise
  if poisson:
    noisy = random_noise(noisy, mode="poisson")

  # Slight blur for sensor diffusion / defocus
  blurred = filters.gaussian(
      noisy,
      sigma=blur_sigma,
      channel_axis=2,
  )

  return util.img_as_ubyte(exposure.rescale_intensity(blurred))


# =====================================================================
# C. Optical distortions (perspective + radial + lighting gradient)
# =====================================================================
def apply_camera_effects(img: np.ndarray,
                         tilt_x: float = 0.15,
                         tilt_y: float = 0.10,
                         k1: float = -0.3,
                         k2: float = 0.05,
                         light_grad: bool = True) -> np.ndarray:
  """Apply perspective, lens distortion, and lighting falloff."""
  h, w = img.shape[:2]

  # --- Perspective (planar homography) ---
  dx = tilt_x * w
  dy = tilt_y * h
  src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
  dst = np.float32(
      [
          [dx, dy],
          [w - dx, dy / 2],
          [w, h],
          [0, h - dy],
      ]
  )
  H = cv2.getPerspectiveTransform(src, dst)
  persp = cv2.warpPerspective(
      img,
      H,
      (w, h),
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=(255, 255, 255),
  )

  # --- Radial lens distortion (barrel / pincushion) ---
  cx, cy = w / 2, h / 2
  r_norm = float(max(cx, cy))

  xx, yy = np.meshgrid(
      np.arange(w, dtype=np.float32),
      np.arange(h, dtype=np.float32),
  )
  x_d = (xx - cx) / r_norm
  y_d = (yy - cy) / r_norm
  r2 = x_d * x_d + y_d * y_d

  factor = 1.0 + k1 * r2 + k2 * r2 * r2
  factor = np.where(factor == 0.0, 1.0, factor)

  x_u = x_d / factor
  y_u = y_d / factor

  map_x = (x_u * r_norm + cx).astype(np.float32)
  map_y = (y_u * r_norm + cy).astype(np.float32)

  distorted = cv2.remap(
      persp,
      map_x,
      map_y,
      cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REFLECT,
  )

  # --- Illumination gradient (along perspective axis) ---
  if light_grad:
    grad = np.linspace(1.15, 0.85, h, dtype=np.float32).reshape(h, 1)
    grad = np.repeat(grad, w, axis=1)
    distorted = np.clip(
        distorted.astype(np.float32) * grad[..., None],
        0,
        255,
    ).astype(np.uint8)

  return distorted


# =====================================================================
# D. Paper texture overlay (multiplicative reflectance variation)
# =====================================================================
def add_paper_texture(img: np.ndarray,
                      strength: float = 0.15,
                      scale: float = 8.0) -> np.ndarray:
  """Overlay subtle paper fiber texture using a smoothed random field."""
  h, w = img.shape[:2]

  # Coarse random field -> blurred to get large-scale fibers / patches
  noise_field = np.random.randn(h, w).astype(np.float32)
  noise_field = cv2.GaussianBlur(noise_field, ksize=(0, 0), sigmaX=scale)

  # Normalize to [0, 1]
  noise_min = float(noise_field.min())
  noise_ptp = float(np.ptp(noise_field)) + 1e-9  # fixed for NumPy 2.0+
  noise_norm = (noise_field - noise_min) / noise_ptp

  # Convert to multiplicative texture around 1.0
  texture = 1.0 + strength * (noise_norm - 0.5)

  textured = np.clip(
      img.astype(np.float32) * texture[..., None],
      0,
      255,
  ).astype(np.uint8)
  return textured


# =====================================================================
# E. Vignetting + chromatic imbalance
# =====================================================================
def add_vignette_and_color_shift(
    img: np.ndarray,
    vignette_strength: float = 0.35,
    warm_strength: float = 0.10,
) -> np.ndarray:
  """
  Add radial vignetting and subtle warm chromatic shift toward corners.

  - Vignetting: darkens corners relative to center.
  - Chromatic shift: boosts R and attenuates B toward edges
    (imitating warm corner tint from real phone optics).
  """
  h, w = img.shape[:2]
  cx, cy = w / 2.0, h / 2.0

  # Correct, symmetric coordinate grid
  xx, yy = np.meshgrid(
      np.arange(w, dtype=np.float32),
      np.arange(h, dtype=np.float32),
  )

  # Compute normalized radial distance from center (0..1)
  rx = (xx - cx) / cx
  ry = (yy - cy) / cy
  r = np.sqrt(rx * rx + ry * ry)
  r_norm = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

  # --- Vignette mask ---
  vignette_mask = 1.0 - vignette_strength * (r_norm**2)

  # --- Warm color shift masks ---
  warm_mask = 1.0 + warm_strength * (r_norm**1.2)
  cool_mask = 1.0 - 0.5 * warm_strength * (r_norm**1.2)

  # Apply to image
  img_f = img.astype(np.float32)

  # Brightness falloff
  img_f *= vignette_mask[..., None]

  # Chromatic tweak: B , R  toward corners
  img_f[..., 0] *= cool_mask    # B channel
  img_f[..., 2] *= warm_mask    # R channel

  return np.clip(img_f, 0, 255).astype(np.uint8)


# =====================================================================
# F. Full pipeline
# =====================================================================
def generate_synthetic_photo() -> np.ndarray:
  """Generate one synthetic photo with full realism stack."""
  # 1. Ideal scene (flat)
  ideal = render_scene()

  # 2. Sensor noise (BEFORE any optical distortion)
  sensor_img = apply_sensor_noise(
      ideal,
      gaussian_std=0.02,
      poisson=True,
      blur_sigma=0.7,
  )

  # 3. Optical distortion: perspective + lens + gradient
  optical_img = apply_camera_effects(
      sensor_img,
      tilt_x=0.18,
      tilt_y=0.10,
      k1=-0.25,
      k2=0.06,
      light_grad=True,
  )

  # 4. Paper texture
  textured = add_paper_texture(
      optical_img,
      strength=0.12,
      scale=8.0,
  )

  # 5. Vignetting + chromatic imbalance
  final_img = add_vignette_and_color_shift(
      textured,
      vignette_strength=0.35,
      warm_strength=0.10,
  )

  return final_img


if __name__ == "__main__":
  img = generate_synthetic_photo()
  cv2.imshow("Synthetic Lab Photo (Paper + Vignette + Warm Edges)", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
