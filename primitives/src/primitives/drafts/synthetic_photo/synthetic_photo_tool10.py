"""
synthetic_photo_tool.py
-----------------------
Production-ready synthetic camera pipeline with:

- Matplotlib scene renderer (render_scene)
- SyntheticPhotoProcessor:
    * lighting + paper texture
    * 4 noise types (gaussian, poisson, s&p, speckle)
    * optics (focal length, tilt, radial lens distortion)
    * post-optical vignetting + chromatic shift
    * presets
    * JSON describe / from_json / verify_hash
"""

from __future__ import annotations

import json
import hashlib
import logging
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import cv2
from skimage import util, exposure, filters
from skimage.util import random_noise
import matplotlib.pyplot as plt


# ======================================================================
# Stage 0: Matplotlib scene renderer (contract: returns RGBA)
# ======================================================================
def render_scene(width_mm: float = 100,
                 height_mm: float = 80,
                 dpi: int = 200) -> np.ndarray:
  """Render an ideal grid + primitives scene via Matplotlib.

  Returns:
      RGBA numpy array (H x W x 4), to be passed into SyntheticPhotoProcessor.
  """
  fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
  ax.set_xlim(0, width_mm)
  ax.set_ylim(0, height_mm)
  ax.set_aspect("equal")
  ax.axis("off")

  # Grid (1mm + 10mm thicker lines)
  for x in np.arange(0, width_mm + 1, 1):
    ax.axvline(x, color="gray", lw=0.2, alpha=0.5)
  for y in np.arange(0, height_mm + 1, 1):
    ax.axhline(y, color="gray", lw=0.2, alpha=0.5)
  for x in np.arange(0, width_mm + 1, 10):
    ax.axvline(x, color="gray", lw=0.6, alpha=0.7)
  for y in np.arange(0, height_mm + 1, 10):
    ax.axhline(y, color="gray", lw=0.6, alpha=0.7)

  # Primitives: square, circle, triangle
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


# ======================================================================
# SyntheticPhotoProcessor: config + full image pipeline
# ======================================================================
class SyntheticPhotoProcessor:
  """Synthetic camera simulation and manager.
  Apply physically ordered postprocessing steps to a Matplotlib RGBA scene.

  Attributes (suggested typical range):
  ------------------------------------
  
  # Lighting & texture (pre-optical)
  
  top_bright: float  (1.0-1.2)   - brightness at top of paper
  bottom_dark: float (0.6-1.0)   - brightness at bottom
  texture_strength: float (0.0-0.2) - 0 disables, >0 stronger paper fibers
  texture_scale: float (4-12)    - larger = coarser fibers

  # Sensor simulation (pre-optical)
  
  | Parameter       | 0 (off)      | 0.02 (moderate)     | 0.05 (strong)              |
  | --------------- | ------------ | ------------------- | -------------------------- |
  | `gaussian_std`  | clean sensor | light readout grain | harsh electronic noise     |
  | `poisson_noise` | off          | on                  | on (binary toggle)         |
  | `sp_amount`     | none         | few dead pixels     | visible salt/pepper spots  |
  | `speckle_var`   | none         | film-grain shimmer  | heavy multiplicative noise |

  # Optical mapping

  | Parameter     | Range      | Visual effect                                         |
  | ------------- | ---------- | ----------------------------------------------------- |
  | `focal_scale` | <1         | **Wide angle**: more paper visible, slight stretching |
  |               | 1          | **Baseline**: current appearance                      |
  |               | >1         | **Telephoto**: tighter crop, less distortion          |
  | `tilt_x`      | 0-0.3      | Horizontal skew / yaw                                 |
  | `tilt_y`      | 0-0.3      | Vertical skew / pitch                                 |
  | `k1`, `k2`    | +/-0.1-0.3 | Barrel or pincushion curvature                        |

  The combination of focal_scale, tilt_x/y, and k1/k2 gives you an expressive, camera-like
  control surface - you can replicate everything from a flat scanner-like top-down view
  (focal_scale=1, tilt=0, k1=k2=0) to a handheld smartphone photo (focal_scale~0.8,
  tilt_x~0.15, tilt_y~0.1, k1~-0.25).

  # Post-optical (camera artifacts)
  
  vignette_strength: float (0-0.5)  - 0 disables; 0.3 moderate
  warm_strength: float (0-0.2)      - red cast toward edges; 0 disables

  # Presets

  Each preset should define a physically consistent camera behavior:
  - Field of view (focal length)
  - Perspective tilt
  - Lens distortion
  - Vignetting & chromatic edges
  - Optional noise characteristics  
  
  | Preset         | FOV (`focal_scale`) | Tilt (x/y)  | Distortion (k1/k2) | Noise Level | Visual Style                    |
  | -------------- | ------------------- | ----------- | ------------------ | ----------- | ------------------------------- |
  | **flatbed**    | 1.0                 | 0           | 0                  | none        | Perfect top-down scan           |
  | **dslr_macro** | 1.5                 | 0.05        | +0.05              | very low    | Slight zoom, clean optics       |
  | **smartphone** | 0.8                 | 0.15 / 0.10 | -0.25 / +0.05      | medium      | Realistic handheld photo        |
  | **wide_lab**   | 0.6                 | 0.2 / 0.15  | -0.35 / +0.1       | low         | Wide lab shot, strong curvature |
  | **lowlight**   | 1.0                 | 0.1         | -0.15 / +0.05      | high        | Dim lighting, grainy edges      |
  
  # Parameter groups
  
  | Setter                   | Purpose                                | Validation logic                                          |
  | ------------------------ | -------------------------------------- | --------------------------------------------------------- |
  | `set_lighting_texture()` | Handles illumination & texture realism | Clamps brightness and texture strength to safe range      |
  | `set_noise()`            | Handles pre-optical sensor noise       | Prevents numeric blowout; `blur_sigma` always positive    |
  | `set_optics()`           | Camera geometry & lens                 | Prevents invalid FOV or distortion coefficients           |
  | `set_post_optical()`     | Final image effects                    | Limits vignette and chromatic shift to perceptual realism | 
  """

  # ------------------------------------------------------------------
  def __init__(self,
               preset: str | None = None,
               # --- Lighting & Texture ---
               top_bright: float = 1.15,
               bottom_dark: float = 0.85,
               texture_strength: float = 0.12,
               texture_scale: float = 8.0,
               # --- Noise ---
               gaussian_std: float = 0.02,
               poisson_noise: bool = True,
               sp_amount: float = 0.0,
               speckle_var: float = 0.0,
               blur_sigma: float = 0.8,
               # --- Optics ---
               tilt_x: float = 0.18,
               tilt_y: float = 0.10,
               focal_scale: float = 1.0,
               k1: float = -0.25,
               k2: float = 0.05,
               pad_px: int = 100,
               # --- Post-Optical ---
               vignette_strength: float = 0.35,
               warm_strength: float = 0.10):
    """Initialize SyntheticPhotoProcessor.

    Args:
        preset: Optional preset name (if provided, overrides all manual params).
        Other parameters correspond to their physical groups; see setters.
    """
    # Logger (safe default; can be replaced by user)
    self.logger = logging.getLogger(self.__class__.__name__)
    if not self.logger.handlers:
      handler = logging.StreamHandler()
      fmt = logging.Formatter("[%(levelname)s] %(message)s")
      handler.setFormatter(fmt)
      self.logger.addHandler(handler)
      self.logger.setLevel(logging.INFO)

    # Initialize via preset or manual groups
    if preset:
      self.set_preset(preset)
    else:
      self.set_lighting_texture(top_bright, bottom_dark,
                                texture_strength, texture_scale)
      self.set_noise(gaussian_std, poisson_noise,
                     sp_amount, speckle_var, blur_sigma)
      self.set_optics(tilt_x, tilt_y, focal_scale, k1, k2, pad_px)
      self.set_post_optical(vignette_strength, warm_strength)

  # ===================================================================
  # Internal helper: clipping warnings
  # ===================================================================
  def _warn_clip(self, param: str, value: float,
                 new_value: float, low: float, high: float) -> None:
    if new_value != value:
      msg = (f"\u26A0\uFE0F  Parameter '{param}' clipped from "
             f"{value:.3f} to {new_value:.3f} "
             f"(valid range {low:.3f}\u2013{high:.3f})")
      try:
        self.logger.warning(msg)
      except Exception:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

  # ===================================================================
  # Group setters (with validation + logging)
  # ===================================================================
  def set_lighting_texture(self,
                           top_bright: float,
                           bottom_dark: float,
                           texture_strength: float,
                           texture_scale: float) -> None:
    """Configure lighting gradient and paper texture parameters.

    Args:
        top_bright: Relative top brightness multiplier (~1.0-1.3).
        bottom_dark: Relative bottom brightness multiplier (~0.7-1.0).
        texture_strength: Amplitude of paper texture (0-1).
        texture_scale: Gaussian smoothing kernel for texture (>0).
    """
    tb = float(np.clip(top_bright, 0.5, 2.0))
    self._warn_clip("top_bright", top_bright, tb, 0.5, 2.0)

    bd = float(np.clip(bottom_dark, 0.5, 2.0))
    self._warn_clip("bottom_dark", bottom_dark, bd, 0.5, 2.0)

    ts = float(np.clip(texture_strength, 0.0, 1.0))
    self._warn_clip("texture_strength", texture_strength, ts, 0.0, 1.0)

    sc = max(0.1, float(texture_scale))
    if sc != texture_scale:
      self.logger.warning(
        "\u26A0\uFE0F  Parameter 'texture_scale' raised to minimum 0.1")

    self.top_bright = tb
    self.bottom_dark = bd
    self.texture_strength = ts
    self.texture_scale = sc

  # ------------------------------------------------------------------
  def set_noise(self,
                gaussian_std: float,
                poisson_noise: bool,
                sp_amount: float,
                speckle_var: float,
                blur_sigma: float) -> None:
    """Configure sensor noise model parameters.

    Args:
        gaussian_std: sigma of Gaussian noise (0-0.2 typical).
        poisson_noise: Whether to apply Poisson noise.
        sp_amount: Salt-and-pepper noise fraction (0-0.1).
        speckle_var: Variance of multiplicative speckle (0-0.1 typical).
        blur_sigma: Gaussian blur radius (0-5 typical).
    """
    gs = float(np.clip(gaussian_std, 0.0, 0.3))
    self._warn_clip("gaussian_std", gaussian_std, gs, 0.0, 0.3)

    sa = float(np.clip(sp_amount, 0.0, 0.1))
    self._warn_clip("sp_amount", sp_amount, sa, 0.0, 0.1)

    sv = float(np.clip(speckle_var, 0.0, 0.2))
    self._warn_clip("speckle_var", speckle_var, sv, 0.0, 0.2)

    bs = float(np.clip(blur_sigma, 0.0, 10.0))
    self._warn_clip("blur_sigma", blur_sigma, bs, 0.0, 10.0)

    self.gaussian_std = gs
    self.poisson_noise = bool(poisson_noise)
    self.sp_amount = sa
    self.speckle_var = sv
    self.blur_sigma = bs

  # ------------------------------------------------------------------
  def set_optics(self,
                 tilt_x: float,
                 tilt_y: float,
                 focal_scale: float,
                 k1: float,
                 k2: float,
                 pad_px: int) -> None:
    """Configure optical geometry and lens distortion parameters.

    Args:
        tilt_x: Horizontal tilt fraction (0-0.5 typical).
        tilt_y: Vertical tilt fraction (0-0.5 typical).
        focal_scale: FOV scaling (>0). <1 wide, >1 telephoto.
        k1: Primary radial distortion coefficient (~-0.5-0.5).
        k2: Secondary radial distortion coefficient (~-0.5-0.5).
        pad_px: Border padding (>=0).
    """
    tx = float(np.clip(tilt_x, 0.0, 0.5))
    self._warn_clip("tilt_x", tilt_x, tx, 0.0, 0.5)

    ty = float(np.clip(tilt_y, 0.0, 0.5))
    self._warn_clip("tilt_y", tilt_y, ty, 0.0, 0.5)

    fs = max(0.05, float(focal_scale))
    if fs != focal_scale:
      self.logger.warning(
        "\u26A0\uFE0F  Parameter 'focal_scale' raised to minimum 0.05")

    k1c = float(np.clip(k1, -1.0, 1.0))
    self._warn_clip("k1", k1, k1c, -1.0, 1.0)

    k2c = float(np.clip(k2, -1.0, 1.0))
    self._warn_clip("k2", k2, k2c, -1.0, 1.0)

    pp = max(0, int(pad_px))
    if pp != pad_px:
      self.logger.warning(
        "\u26A0\uFE0F  Parameter 'pad_px' raised to minimum 0")

    self.tilt_x = tx
    self.tilt_y = ty
    self.focal_scale = fs
    self.k1 = k1c
    self.k2 = k2c
    self.pad_px = pp

  # ------------------------------------------------------------------
  def set_post_optical(self,
                       vignette_strength: float,
                       warm_strength: float) -> None:
    """Configure post-optical vignetting and chromatic effects.

    Args:
        vignette_strength: Radial attenuation strength (0-1 typical).
        warm_strength: Channel bias simulating color temperature shift (0-0.5).
    """
    vs = float(np.clip(vignette_strength, 0.0, 1.0))
    self._warn_clip("vignette_strength", vignette_strength, vs, 0.0, 1.0)

    ws = float(np.clip(warm_strength, 0.0, 0.5))
    self._warn_clip("warm_strength", warm_strength, ws, 0.0, 0.5)

    self.vignette_strength = vs
    self.warm_strength = ws

  # ===================================================================
  # Presets
  # ===================================================================
  def set_preset(self, name: str) -> None:
    preset = name.lower().strip()
    self.logger.info(f"Applying preset '{preset}'")

    if preset == "flatbed":
      self.set_lighting_texture(1.0, 1.0, 0.0, 8.0)
      self.set_noise(0.0, False, 0.0, 0.0, 0.0)
      self.set_optics(0.0, 0.0, 1.0, 0.0, 0.0, 100)
      self.set_post_optical(0.0, 0.0)

    elif preset == "smartphone":
      self.set_lighting_texture(1.15, 0.85, 0.12, 8.0)
      self.set_noise(0.02, True, 0.005, 0.015, 0.8)
      self.set_optics(0.15, 0.10, 0.8, -0.25, 0.05, 100)
      self.set_post_optical(0.35, 0.10)

    elif preset == "dslr_macro":
      self.set_lighting_texture(1.15, 0.90, 0.10, 6.0)
      self.set_noise(0.005, True, 0.001, 0.010, 0.3)
      self.set_optics(0.05, 0.05, 1.5, 0.05, 0.0, 80)
      self.set_post_optical(0.15, 0.05)

    else:
      raise ValueError(f"Unknown preset '{name}'")

  # ===================================================================
  # Image pipeline building blocks
  # ===================================================================
  @staticmethod
  def from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA (Matplotlib) to BGR (OpenCV)."""
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

  # ------------------------------------------------------------------
  def apply_paper_lighting(self, img: np.ndarray) -> np.ndarray:
    """Uneven lighting gradient (pre-optical)."""
    h, w = img.shape[:2]
    grad = np.linspace(self.top_bright, self.bottom_dark,
                       h, dtype=np.float32).reshape(h, 1)
    grad = np.repeat(grad, w, axis=1)
    return np.clip(img.astype(np.float32) * grad[..., None],
                   0, 255).astype(np.uint8)

  # ------------------------------------------------------------------
  def add_paper_texture(self, img: np.ndarray) -> np.ndarray:
    """Multiplicative paper fiber texture (pre-optical)."""
    if self.texture_strength <= 0:
      return img
    h, w = img.shape[:2]
    noise_field = np.random.randn(h, w).astype(np.float32)
    noise_field = cv2.GaussianBlur(noise_field, (0, 0), self.texture_scale)
    nf_min = float(noise_field.min())
    nf_ptp = float(np.ptp(noise_field)) + 1e-9
    noise_norm = (noise_field - nf_min) / nf_ptp
    texture = 1.0 + self.texture_strength * (noise_norm - 0.5)
    return np.clip(img.astype(np.float32) * texture[..., None],
                   0, 255).astype(np.uint8)

  # ------------------------------------------------------------------
  def apply_sensor_noise(self, img: np.ndarray) -> np.ndarray:
    """Apply combined sensor-domain noises and blur.

    Each noise model can be independently controlled.

    Gaussian  (gaussian_std): additive white noise [0 -> 0.05]
    Poisson   (poisson_noise): photon shot noise (bool)
    Salt&Pepper (sp_amount): impulse noise fraction [0 -> 0.05]
    Speckle   (speckle_var): multiplicative variance [0 -> 0.05]
    """
    img_f = util.img_as_float(img)

    # Gaussian
    if self.gaussian_std > 0:
      img_f = random_noise(img_f, mode="gaussian",
                           var=self.gaussian_std**2)

    # Poisson
    if self.poisson_noise:
      img_f = random_noise(img_f, mode="poisson")

    # Salt & pepper
    if self.sp_amount > 0:
      img_f = random_noise(img_f, mode="s&p",
                           amount=self.sp_amount)

    # Speckle
    if self.speckle_var > 0:
      img_f = random_noise(img_f, mode="speckle",
                           var=self.speckle_var)

    # Blur
    if self.blur_sigma > 0:
      img_f = filters.gaussian(img_f,
                               sigma=self.blur_sigma,
                               channel_axis=2)

    return util.img_as_ubyte(exposure.rescale_intensity(img_f))

  # ------------------------------------------------------------------
  def apply_camera_effects(self, img: np.ndarray) -> np.ndarray:
    """Apply focal scaling, perspective tilt, and radial lens distortion.

    Args:
        img: Input BGR uint8 image (pre-optical domain).

    Returns:
        np.ndarray: Distorted image simulating optical projection.
    """
    # Padding
    if self.pad_px > 0:
      img = cv2.copyMakeBorder(img,
                               self.pad_px, self.pad_px,
                               self.pad_px, self.pad_px,
                               borderType=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
    h, w = img.shape[:2]

    # ===============================================================
    # 1. FOCAL LENGTH (Field of View)
    # ---------------------------------------------------------------
    # focal_scale = 1.0 -> baseline FOV
    # focal_scale > 1 -> telephoto (zoom-in, narrower)
    # focal_scale < 1 -> wide angle (zoom-out, wider)
    # ===============================================================
    if self.focal_scale != 1.0:
      f = self.focal_scale
      new_w = int(round(w / f))
      new_h = int(round(h / f))
      cx, cy = w // 2, h // 2
      x1 = max(cx - new_w // 2, 0)
      x2 = min(cx + new_w // 2, w)
      y1 = max(cy - new_h // 2, 0)
      y2 = min(cy + new_h // 2, h)
      cropped = img[y1:y2, x1:x2]
      img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # ===============================================================
    # 2. CAMERA TILT (Perspective)
    # ---------------------------------------------------------------
    # tilt_x, tilt_y are normalized fractions of width/height.
    # 0 = no tilt; 0.3 = strong skew.
    # ===============================================================
    dx = self.tilt_x * w
    dy = self.tilt_y * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy],
                      [w - dx, dy / 2],
                      [w, h],
                      [0, h - dy]])
    H = cv2.getPerspectiveTransform(src, dst)
    persp = cv2.warpPerspective(img, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    # ===============================================================
    # 3. RADIAL LENS DISTORTION
    # ---------------------------------------------------------------
    # k1 < 0 -> barrel (wide angle)
    # k1 > 0 -> pincushion (telephoto)
    # k2 refines curvature falloff
    # ===============================================================
    cx, cy = w / 2.0, h / 2.0
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d * x_d + y_d * y_d
    factor = 1.0 + self.k1 * r2 + self.k2 * (r2**2)
    factor = np.where(factor == 0.0, 1.0, factor)
    x_u = x_d / factor
    y_u = y_d / factor
    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)

    distorted = cv2.remap(persp, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return distorted

  # ------------------------------------------------------------------
  def add_vignette_and_color_shift(self, img: np.ndarray) -> np.ndarray:
    """Apply post-lens vignetting and chromatic imbalance."""
    if self.vignette_strength <= 0 and self.warm_strength <= 0:
      return img

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    rx = (xx - cx) / cx
    ry = (yy - cy) / cy
    r = np.sqrt(rx * rx + ry * ry)
    r_norm = np.clip(r / (r.max() + 1e-9), 0.0, 1.0)

    vignette_mask = 1.0 - self.vignette_strength * (r_norm**2)
    warm_mask = 1.0 + self.warm_strength * (r_norm**1.2)
    cool_mask = 1.0 - 0.5 * self.warm_strength * (r_norm**1.2)

    img_f = img.astype(np.float32)
    img_f *= vignette_mask[..., None]
    img_f[..., 0] *= cool_mask   # B
    img_f[..., 2] *= warm_mask   # R

    return np.clip(img_f, 0, 255).astype(np.uint8)

  # ===================================================================
  # Full pipeline
  # ===================================================================
  def process(self, rgba: np.ndarray) -> np.ndarray:
    """Run full forward model: RGBA (Matplotlib) -> BGR uint8 (synthetic photo)."""
    img = self.from_rgba(rgba)
    img = self.apply_paper_lighting(img)
    img = self.add_paper_texture(img)
    img = self.apply_sensor_noise(img)
    img = self.apply_camera_effects(img)
    img = self.add_vignette_and_color_shift(img)
    return img

  # ------------------------------------------------------------------
  def process_stages(self, rgba: np.ndarray) -> dict[str, np.ndarray]:
    """Run full pipeline and return intermediate stages as dict.

    Returns:
        dict with keys:
          'base', 'lit', 'textured', 'noisy', 'optical', 'final'
    """
    stages: dict[str, np.ndarray] = {}
    stages["base"] = self.from_rgba(rgba)
    stages["lit"] = self.apply_paper_lighting(stages["base"])
    stages["textured"] = self.add_paper_texture(stages["lit"])
    stages["noisy"] = self.apply_sensor_noise(stages["textured"])
    stages["optical"] = self.apply_camera_effects(stages["noisy"])
    stages["final"] = self.add_vignette_and_color_shift(stages["optical"])
    return stages

  # ===================================================================
  # Provenance / hashing / JSON
  # ===================================================================
  def _parameter_dict(self) -> dict[str, Any]:
    return {
      "lighting": {
        "top_bright": self.top_bright,
        "bottom_dark": self.bottom_dark,
        "texture_strength": self.texture_strength,
        "texture_scale": self.texture_scale,
      },
      "noise": {
        "gaussian_std": self.gaussian_std,
        "poisson_noise": self.poisson_noise,
        "sp_amount": self.sp_amount,
        "speckle_var": self.speckle_var,
        "blur_sigma": self.blur_sigma,
      },
      "optics": {
        "tilt_x": self.tilt_x,
        "tilt_y": self.tilt_y,
        "focal_scale": self.focal_scale,
        "k1": self.k1,
        "k2": self.k2,
        "pad_px": self.pad_px,
      },
      "post_optical": {
        "vignette_strength": self.vignette_strength,
        "warm_strength": self.warm_strength,
      },
    }

  # ------------------------------------------------------------------
  def _pretty_text(self, sha: str | None) -> str:
    p = self._parameter_dict()
    pad = "  "
    lines = ["SyntheticPhotoProcessor configuration:"]
    for group, vals in p.items():
      lines.append(f"{pad}--- {group.capitalize()} ---")
      for k, v in vals.items():
        lines.append(f"{pad}{k:18s}= {v}")
      lines.append("")
    if sha:
      lines.append(f"{pad}hash (SHA256) = {sha}")
    return "\n".join(lines)

  # ------------------------------------------------------------------
  def describe(self,
               format: str = "text",
               path: str | None = None,
               include_hash: bool = False,
               return_str: bool = False) -> str | None:
    """Describe or export the current processor configuration.

    Args:
        format: "text" (default) or "json" - controls return/output format.
        path: Optional file path to save configuration (.txt or .json).
        return_str: If True, returns the formatted representation instead of printing.
        include_hash: If True, include a SHA256 hash of parameters for reproducibility.

    Returns:
        str | None: The formatted configuration string or JSON text.
    """
    meta = {
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "class": "SyntheticPhotoProcessor",
      "parameters": self._parameter_dict(),
    }

    sha = None
    if include_hash:
      sha = hashlib.sha256(
        json.dumps(meta["parameters"], sort_keys=True).encode("utf-8")
      ).hexdigest()
      meta["hash"] = sha

    text = json.dumps(meta, indent=2) if format == "json" else self._pretty_text(sha)

    if path:
      with open(path, "w", encoding="utf-8") as f:
        f.write(text)
      self.logger.info(f"\u2705  Configuration saved \u2192 {path}")

    if return_str:
      return text
    print(text)
    return None

  # ------------------------------------------------------------------
  @classmethod
  def from_json(cls, path: str) -> SyntheticPhotoProcessor:
    """Reconstruct a processor instance from a saved JSON configuration."""
    with open(path, "r", encoding="utf-8") as f:
      meta = json.load(f)
    if "parameters" not in meta:
      raise ValueError("Invalid JSON config (no 'parameters').")
    params = meta["parameters"]
    kwargs: dict[str, Any] = {}
    for section in params.values():
      kwargs.update(section)
    obj = cls(**kwargs)
    obj.logger.info(f"\u2705  Configuration loaded from {path}")
    return obj

  # ------------------------------------------------------------------
  def verify_hash(self, path: str, verbose: bool = True) -> bool:
    """Verify that the current processor parameters match a saved JSON hash.

    Args:
        path: Path to a previously saved configuration JSON (with 'hash' field).
        verbose: If True, prints verification results.

    Returns:
        bool: True if hash matches (identical configuration), False otherwise.
    """
    with open(path, "r", encoding="utf-8") as f:
      meta = json.load(f)
    if "hash" not in meta:
      if verbose:
        self.logger.warning("No stored hash in config for verification.")
      return False
    current_hash = hashlib.sha256(
      json.dumps(self._parameter_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    match = current_hash == meta["hash"]
    if verbose:
      if match:
        self.logger.info("\u2705  Configuration verified: hash matches.")
      else:
        self.logger.warning("\u26A0\uFE0F  Hash mismatch!")
    return match


# ======================================================================
# Demo / comparison entrypoint
# ======================================================================
if __name__ == "__main__":
  rgba = render_scene()
  processor = SyntheticPhotoProcessor(preset="smartphone")
  img = processor.process(rgba)

  rgba = render_scene()
  proc = SyntheticPhotoProcessor()
  proc.set_preset("smartphone")
  
  # Log configuration
  proc.describe(include_hash=True)

  # Retrieve JSON text
  cfg_json = proc.describe(format="json", include_hash=True, return_str=True)

  # Save both human-readable and machine-readable configs
  proc.describe(path="smartphone_config.txt")
  proc.describe(format="json", path="smartphone_config.json", include_hash=True)
  
  # Reconstruct processor from saved JSON
  proc_clone = SyntheticPhotoProcessor.from_json("smartphone_config.json")
  
  # Verify equality
  print(proc_clone.describe(return_str=True))
  
  # Generate identical image
  result = proc_clone.process(rgba)
  cv2.imshow("Reconstructed Processor Output", result)
  cv2.waitKey(0)
