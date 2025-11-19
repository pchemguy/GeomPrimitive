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

import json
import hashlib
from datetime import datetime
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
               # Optional quick preset
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
    if preset:
      # Apply preset if given (overrides all)
      self.set_preset(preset)
    else:
      # Store groups via unified setters for consistency
      self.set_lighting_texture(top_bright, bottom_dark, texture_strength, texture_scale)
      self.set_noise(gaussian_std, poisson_noise, sp_amount, speckle_var, blur_sigma)
      self.set_optics(tilt_x, tilt_y, focal_scale, k1, k2, pad_px)
      self.set_post_optical(vignette_strength, warm_strength)

  # ------------------------------------------------------------------
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
    import numpy as np
    self.top_bright = float(np.clip(top_bright, 0.5, 2.0))
    self.bottom_dark = float(np.clip(bottom_dark, 0.5, 2.0))
    self.texture_strength = float(np.clip(texture_strength, 0.0, 1.0))
    self.texture_scale = max(0.1, float(texture_scale))

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
    import numpy as np
    self.gaussian_std = float(np.clip(gaussian_std, 0.0, 0.3))
    self.poisson_noise = bool(poisson_noise)
    self.sp_amount = float(np.clip(sp_amount, 0.0, 0.1))
    self.speckle_var = float(np.clip(speckle_var, 0.0, 0.2))
    self.blur_sigma = float(np.clip(blur_sigma, 0.0, 10.0))

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
    import numpy as np
    self.tilt_x = float(np.clip(tilt_x, 0.0, 0.5))
    self.tilt_y = float(np.clip(tilt_y, 0.0, 0.5))
    self.focal_scale = max(0.05, float(focal_scale))
    self.k1 = float(np.clip(k1, -1.0, 1.0))
    self.k2 = float(np.clip(k2, -1.0, 1.0))
    self.pad_px = max(0, int(pad_px))

  # ------------------------------------------------------------------
  def set_post_optical(self,
                       vignette_strength: float,
                       warm_strength: float) -> None:
    """Configure post-optical vignetting and chromatic effects.

    Args:
        vignette_strength: Radial attenuation strength (0-1 typical).
        warm_strength: Channel bias simulating color temperature shift (0-0.5).
    """
    import numpy as np
    self.vignette_strength = float(np.clip(vignette_strength, 0.0, 1.0))
    self.warm_strength = float(np.clip(warm_strength, 0.0, 0.5))
  
  # ------------------------------------------------------------------
  @classmethod
  def from_json(cls, path: str) -> "SyntheticPhotoProcessor":
    """Reconstruct a processor instance from a saved JSON configuration."""
    import json
    with open(path, "r", encoding="utf-8") as f:
      meta = json.load(f)

    if "parameters" not in meta:
      raise ValueError(f"Invalid JSON file: {path}")

    p = meta["parameters"]
    kwargs = {}

    # Flatten nested sections
    for section in p.values():
      kwargs.update(section)

    instance = cls(**kwargs)
    print(f"[SyntheticPhotoProcessor] Loaded configuration from {path}")
    return instance

  # ------------------------------------------------------------------
  def verify_hash(self, path: str, verbose: bool = True) -> bool:
    """Verify that the current processor parameters match a saved JSON hash.

    Args:
        path: Path to a previously saved configuration JSON (with 'hash' field).
        verbose: If True, prints verification results.

    Returns:
        bool: True if hash matches (identical configuration), False otherwise.
    """
    import json, hashlib

    try:
      with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    except Exception as e:
      if verbose:
        print(f"[SyntheticPhotoProcessor] Failed to read {path}: {e}")
      return False

    if "parameters" not in meta:
      if verbose:
        print(f"[SyntheticPhotoProcessor] File {path} missing 'parameters' field.")
      return False

    if "hash" not in meta:
      if verbose:
        print(f"[SyntheticPhotoProcessor] File {path} has no stored hash to verify.")
      return False

    # Compute hash of current parameters
    params_current = {
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

    params_json = json.dumps(params_current, sort_keys=True)
    hash_now = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
    hash_ref = meta.get("hash")

    match = hash_now == hash_ref

    if verbose:
      if match:
        print(f"[SyntheticPhotoProcessor] V Configuration verified: hash matches.")
        print(f"  Hash: {hash_ref}")
      else:
        print(f"[SyntheticPhotoProcessor] X Hash mismatch!")
        print(f"  Expected: {hash_ref}")
        print(f"  Current : {hash_now}")

    return match

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
    """Apply combined sensor-domain noises and blur.

    Each noise model can be independently controlled.

    Gaussian  (gaussian_std): additive white noise [0 -> 0.05]
    Poisson   (poisson_noise): photon shot noise (bool)
    Salt&Pepper (sp_amount): impulse noise fraction [0 -> 0.05]
    Speckle   (speckle_var): multiplicative variance [0 -> 0.05]
    """
    img_f = util.img_as_float(img)

    # --- Gaussian ---
    if self.gaussian_std > 0:
      img_f = random_noise(img_f, mode="gaussian", var=self.gaussian_std**2)

    # --- Poisson ---
    if self.poisson_noise:
      img_f = random_noise(img_f, mode="poisson")

    # --- Salt & Pepper ---
    if self.sp_amount > 0:
      img_f = random_noise(img_f, mode="s&p", amount=self.sp_amount)

    # --- Speckle ---
    if self.speckle_var > 0:
      img_f = random_noise(img_f, mode="speckle", var=self.speckle_var)

    # --- Optional sensor blur ---
    if self.blur_sigma > 0:
      img_f = filters.gaussian(img_f, sigma=self.blur_sigma, channel_axis=2)

    return util.img_as_ubyte(exposure.rescale_intensity(img_f))

  # ------------------------------------------------------------------
  def apply_camera_effects(self, img):
    """Apply focal scaling, perspective, and lens distortion.

    Args:
        img: Input BGR uint8 image (pre-optical domain).

    Returns:
        np.ndarray: Distorted image simulating optical projection.
    """
    # --- Padding (optional) ---
    if self.pad_px > 0:
      img = cv2.copyMakeBorder(
          img, self.pad_px, self.pad_px, self.pad_px, self.pad_px,
          borderType=cv2.BORDER_CONSTANT,
          value=(255, 255, 255),
      )

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
    dst = np.float32([
        [dx, dy],
        [w - dx, dy / 2],
        [w, h],
        [0, h - dy],
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    persp = cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # ===============================================================
    # 3. RADIAL LENS DISTORTION
    # ---------------------------------------------------------------
    # k1 < 0 -> barrel (wide angle)
    # k1 > 0 -> pincushion (telephoto)
    # k2 refines curvature falloff
    # ===============================================================
    cx, cy = w / 2, h / 2
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d**2 + y_d**2

    factor = 1 + self.k1 * r2 + self.k2 * (r2**2)
    factor = np.where(factor == 0, 1, factor)

    x_u = x_d / factor
    y_u = y_d / factor

    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)

    distorted = cv2.remap(
        persp,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return distorted

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

  # ------------------------------------------------------------------
  def set_preset(self, name: str):
    """Configure the processor to mimic a realistic camera type.

    Available presets:
      - 'flatbed'     : Perfect orthographic scan (no optics, no noise)
      - 'dslr_macro'  : Slight telephoto, mild vignette, clean optics
      - 'smartphone'  : Wide lens, mild barrel distortion, visible vignetting
      - 'wide_lab'    : Very wide FOV, pronounced barrel distortion
      - 'lowlight'    : Noisy, strong vignetting, reduced contrast

    After calling, you can still fine-tune attributes manually.

    Args:
        name: One of the preset names (case-insensitive).
    """
    preset = name.lower().strip()
    if preset == "flatbed":
      self.focal_scale = 1.0
      self.tilt_x = 0.0
      self.tilt_y = 0.0
      self.k1 = 0.0
      self.k2 = 0.0
      self.gaussian_std = 0.0
      self.poisson_noise = False
      self.sp_amount = 0.0
      self.speckle_var = 0.0
      self.vignette_strength = 0.0
      self.warm_strength = 0.0

    elif preset == "dslr_macro":
      self.focal_scale = 1.5
      self.tilt_x = 0.05
      self.tilt_y = 0.05
      self.k1 = 0.05
      self.k2 = 0.0
      self.gaussian_std = 0.005
      self.poisson_noise = True
      self.sp_amount = 0.001
      self.speckle_var = 0.01
      self.vignette_strength = 0.15
      self.warm_strength = 0.05

    elif preset == "smartphone":
      self.focal_scale = 0.8
      self.tilt_x = 0.15
      self.tilt_y = 0.1
      self.k1 = -0.25
      self.k2 = 0.05
      self.gaussian_std = 0.02
      self.poisson_noise = True
      self.sp_amount = 0.005
      self.speckle_var = 0.015
      self.vignette_strength = 0.35
      self.warm_strength = 0.1

    elif preset == "wide_lab":
      self.focal_scale = 0.6
      self.tilt_x = 0.2
      self.tilt_y = 0.15
      self.k1 = -0.35
      self.k2 = 0.1
      self.gaussian_std = 0.01
      self.poisson_noise = True
      self.sp_amount = 0.002
      self.speckle_var = 0.01
      self.vignette_strength = 0.3
      self.warm_strength = 0.05

    elif preset == "lowlight":
      self.focal_scale = 1.0
      self.tilt_x = 0.1
      self.tilt_y = 0.1
      self.k1 = -0.15
      self.k2 = 0.05
      self.gaussian_std = 0.04
      self.poisson_noise = True
      self.sp_amount = 0.01
      self.speckle_var = 0.03
      self.vignette_strength = 0.45
      self.warm_strength = 0.1
      self.blur_sigma = 1.0

    else:
      raise ValueError(f"Unknown preset '{name}'. "
                       "Choose from: flatbed, dslr_macro, smartphone, "
                       "wide_lab, lowlight.")

    print(f"[SyntheticPhotoProcessor] Camera preset applied: {preset}")

  # ------------------------------------------------------------------
  def describe(
      self,
      format: str = "text",
      indent: int = 2,
      path: str | None = None,
      return_str: bool = False,
      include_hash: bool = False,
  ) -> str | None:
    """Describe or export the current processor configuration.

    Args:
        format: "text" (default) or "json" - controls return/output format.
        indent: Indentation for pretty printing.
        path: Optional file path to save configuration (.txt or .json).
        return_str: If True, returns the formatted representation instead of printing.
        include_hash: If True, include a SHA256 hash of parameters for reproducibility.

    Returns:
        str | None: The formatted configuration string or JSON text.
    """
    # ==========================================================
    # 1. Structured metadata (canonical ground truth)
    # ==========================================================
    meta = {
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "class": "SyntheticPhotoProcessor",
      "parameters": {
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
      },
    }

    # ==========================================================
    # 2. Optional reproducibility hash (based on parameters only)
    # ==========================================================
    if include_hash:
      params_json = json.dumps(meta["parameters"], sort_keys=True)
      sha = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
      meta["hash"] = sha
    else:
      sha = None

    # ==========================================================
    # 3. Build human-readable text summary
    # ==========================================================
    pad = " " * indent
    lines = [
      "SyntheticPhotoProcessor configuration:",
      f"{pad}--- Lighting & Texture ---",
      f"{pad}top_bright        = {self.top_bright:.3f}",
      f"{pad}bottom_dark       = {self.bottom_dark:.3f}",
      f"{pad}texture_strength  = {self.texture_strength:.3f}",
      f"{pad}texture_scale     = {self.texture_scale:.2f}",
      "",
      f"{pad}--- Noise ---",
      f"{pad}gaussian_std      = {self.gaussian_std:.3f}",
      f"{pad}poisson_noise     = {self.poisson_noise}",
      f"{pad}sp_amount         = {self.sp_amount:.3f}",
      f"{pad}speckle_var       = {self.speckle_var:.3f}",
      f"{pad}blur_sigma        = {self.blur_sigma:.3f}",
      "",
      f"{pad}--- Optics ---",
      f"{pad}tilt_x            = {self.tilt_x:.3f}",
      f"{pad}tilt_y            = {self.tilt_y:.3f}",
      f"{pad}focal_scale       = {self.focal_scale:.3f}",
      f"{pad}k1                = {self.k1:.3f}",
      f"{pad}k2                = {self.k2:.3f}",
      f"{pad}pad_px            = {self.pad_px}",
      "",
      f"{pad}--- Post-Optical ---",
      f"{pad}vignette_strength = {self.vignette_strength:.3f}",
      f"{pad}warm_strength     = {self.warm_strength:.3f}",
    ]
    if include_hash and sha:
      lines.append("")
      lines.append(f"{pad}hash (SHA256)     = {sha}")

    text_summary = "\n".join(lines)
    json_summary = json.dumps(meta, indent=2)

    # ==========================================================
    # 4. Output handling (print / save / return)
    # ==========================================================
    result = text_summary if format == "text" else json_summary

    if path:
      with open(path, "w", encoding="utf-8") as f:
        f.write(result)
      print(f"[SyntheticPhotoProcessor] Configuration saved -> {path}")

    if return_str:
      return result
    print(result)


# =====================================================================
# Demo reproducing previous result
# =====================================================================
if __name__ == "__main__":
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
