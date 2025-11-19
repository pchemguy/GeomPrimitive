"""
synthetic_photo_tool.py
-----------------------
Reusable tool for generating realistic 'lab-style' synthetic photographs.

Structure:
  - render_scene(): produces an RGBA NumPy array via Matplotlib.
  - SyntheticPhotoProcessor: applies lighting, texture, noise, lens effects,
    vignetting, and color shifts in physically faithful order.

Usage demo at bottom reproduces previous pipeline exactly.
"""

import numpy as np
import cv2
from skimage import util, exposure, filters
from skimage.util import random_noise
import matplotlib.pyplot as plt


# =====================================================================
# Stage 1: Matplotlib scene renderer (unchanged)
# =====================================================================
def render_scene(width_mm=100, height_mm=80, dpi=200):
  """Render an ideal grid with geometric primitives onto 'paper'.

  Args:
      width_mm: Scene width in millimeters.
      height_mm: Scene height in millimeters.
      dpi: Dots per inch for rasterization.

  Returns:
      np.ndarray: RGBA image (HxWx4) as produced by Matplotlib.
  """
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
  ax.add_patch(plt.Rectangle((30, 30), 20, 20,
                             facecolor="none", edgecolor="red", lw=2))
  ax.add_patch(plt.Circle((60, 40), 10,
                          facecolor="none", edgecolor="blue", lw=2))
  tri = np.array([[10, 10], [20, 10], [15, 25]])
  ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
  plt.close(fig)
  return rgba


# =====================================================================
# Stage 2: Reusable post-Matplotlib processor
# =====================================================================
class SyntheticPhotoProcessor:
  """
  Apply physically ordered postprocessing steps to a Matplotlib RGBA scene.

  Attributes (suggested typical range):
  ------------------------------------
  # Lighting & texture (pre-optical)
  top_bright: float  (1.0-1.2)   - brightness at top of paper
  bottom_dark: float (0.6-1.0)   - brightness at bottom
  texture_strength: float (0.0-0.2) - 0 disables, >0 stronger paper fibers
  texture_scale: float (4-12)    - larger = coarser fibers

  # Sensor simulation (pre-optical)
  gaussian_std: float (0-0.05)   - Gaussian noise std; 0 disables
  poisson_noise: bool            - add Poisson shot noise
  blur_sigma: float (0-1.5)      - optical blur; 0 disables

  # Optical mapping
  tilt_x: float (0-0.3)          - horizontal camera tilt (0 = perpendicular)
  tilt_y: float (0-0.3)          - vertical camera tilt
  k1: float (x0.5-0.5)           - radial lens distortion coefficient 1
  k2: float (x0.2-0.2)           - radial lens distortion coefficient 2
  pad_px: int (0-200)            - border padding to avoid edge loss

  # Post-optical (camera artifacts)
  vignette_strength: float (0-0.5)  - 0 disables; 0.3 moderate
  warm_strength: float (0-0.2)      - red cast toward edges; 0 disables
  """

  # ------------------------------------------------------------------
  def __init__(self,
               top_bright=1.15,
               bottom_dark=0.85,
               texture_strength=0.12,
               texture_scale=8.0,
               gaussian_std=0.02,
               poisson_noise=True,
               blur_sigma=0.8,
               tilt_x=0.18,
               tilt_y=0.10,
               k1=-0.25,
               k2=0.05,
               pad_px=100,
               vignette_strength=0.35,
               warm_strength=0.10):
    self.top_bright = top_bright
    self.bottom_dark = bottom_dark
    self.texture_strength = texture_strength
    self.texture_scale = texture_scale
    self.gaussian_std = gaussian_std
    self.poisson_noise = poisson_noise
    self.blur_sigma = blur_sigma
    self.tilt_x = tilt_x
    self.tilt_y = tilt_y
    self.k1 = k1
    self.k2 = k2
    self.pad_px = pad_px
    self.vignette_strength = vignette_strength
    self.warm_strength = warm_strength

  # ------------------------------------------------------------------
  @staticmethod
  def from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA to BGR uint8 image for processing."""
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

  # ------------------------------------------------------------------
  def apply_paper_lighting(self, img):
    """Uneven lighting gradient (pre-optical)."""
    h, w = img.shape[:2]
    grad = np.linspace(self.top_bright, self.bottom_dark, h, dtype=np.float32)
    grad = np.repeat(grad[:, None], w, axis=1)
    return np.clip(img.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)

  # ------------------------------------------------------------------
  def add_paper_texture(self, img):
    """Multiplicative paper fiber texture (pre-optical)."""
    if self.texture_strength <= 0:
      return img
    h, w = img.shape[:2]
    noise_field = np.random.randn(h, w).astype(np.float32)
    noise_field = cv2.GaussianBlur(noise_field, (0, 0), self.texture_scale)
    noise_min = float(noise_field.min())
    noise_ptp = float(np.ptp(noise_field)) + 1e-9
    noise_norm = (noise_field - noise_min) / noise_ptp
    texture = 1.0 + self.texture_strength * (noise_norm - 0.5)
    return np.clip(img.astype(np.float32) * texture[..., None], 0, 255).astype(np.uint8)

  # ------------------------------------------------------------------
  def apply_sensor_noise(self, img):
    """Gaussian + Poisson noise and blur (pre-optical)."""
    if self.gaussian_std <= 0 and not self.poisson_noise and self.blur_sigma <= 0:
      return img
    img_f = util.img_as_float(img)
    if self.gaussian_std > 0:
      img_f = random_noise(img_f, mode="gaussian", var=self.gaussian_std**2)
    if self.poisson_noise:
      img_f = random_noise(img_f, mode="poisson")
    if self.blur_sigma > 0:
      img_f = filters.gaussian(img_f, sigma=self.blur_sigma, channel_axis=2)
    return util.img_as_ubyte(exposure.rescale_intensity(img_f))

  # ------------------------------------------------------------------
  def apply_camera_effects(self, img):
    """Perspective + lens distortion (optical mapping)."""
    # Pad to prevent clipping under strong warps
    if self.pad_px > 0:
      img = cv2.copyMakeBorder(img, self.pad_px, self.pad_px,
                               self.pad_px, self.pad_px,
                               cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
    h, w = img.shape[:2]
    dx, dy = self.tilt_x * w, self.tilt_y * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx, dy / 2], [w, h], [0, h - dy]])
    H = cv2.getPerspectiveTransform(src, dst)
    persp = cv2.warpPerspective(img, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # Radial distortion
    cx, cy = w / 2, h / 2
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d**2 + y_d**2
    factor = 1 + self.k1 * r2 + self.k2 * r2**2
    factor = np.where(factor == 0, 1, factor)
    x_u = x_d / factor
    y_u = y_d / factor
    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)
    return cv2.remap(persp, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255))

  # ------------------------------------------------------------------
  def add_vignette_and_color_shift(self, img):
    """Apply post-lens vignetting and chromatic imbalance."""
    if self.vignette_strength <= 0 and self.warm_strength <= 0:
      return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    rx, ry = (xx - cx) / cx, (yy - cy) / cy
    r = np.sqrt(rx * rx + ry * ry)
    r_norm = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

    vignette_mask = 1.0 - self.vignette_strength * (r_norm**2)
    warm_mask = 1.0 + self.warm_strength * (r_norm**1.2)
    cool_mask = 1.0 - 0.5 * self.warm_strength * (r_norm**1.2)

    img_f = img.astype(np.float32)
    img_f *= vignette_mask[..., None]
    img_f[..., 0] *= cool_mask   # B down
    img_f[..., 2] *= warm_mask   # R up
    return np.clip(img_f, 0, 255).astype(np.uint8)

  # ------------------------------------------------------------------
  def process(self, rgba: np.ndarray) -> np.ndarray:
    """Run full physically ordered pipeline on an RGBA Matplotlib image."""
    img = self.from_rgba(rgba)
    img = self.apply_paper_lighting(img)
    img = self.add_paper_texture(img)
    img = self.apply_sensor_noise(img)
    img = self.apply_camera_effects(img)
    img = self.add_vignette_and_color_shift(img)
    return img


# =====================================================================
# Demo reproducing previous result
# =====================================================================
if __name__ == "__main__":
  rgba = render_scene()
  processor = SyntheticPhotoProcessor()  # default = "moderate" realism
  result = processor.process(rgba)
  cv2.imshow("Synthetic Lab Photo (Reusable Tool)", result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
