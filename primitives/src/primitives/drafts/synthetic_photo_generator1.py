"""
synthetic_photo_generator.py
----------------------------
Physically accurate synthetic photo generator:
    1. Ideal scene (grid + primitives)
    2. Sensor-space noise + blur
    3. Lens + perspective distortions + lighting gradient
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import util, exposure, filters
from skimage.util import random_noise


# =====================================================================
# A. Ideal scene rendering (flat, undistorted)
# =====================================================================
def render_scene(width_mm=100, height_mm=80, dpi=200):
    """Render ideal graph-paper scene with primitives."""
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Grid ---
    for x in np.arange(0, width_mm + 1, 1):
        ax.axvline(x, color="gray", lw=0.2, alpha=0.5)
    for y in np.arange(0, height_mm + 1, 1):
        ax.axhline(y, color="gray", lw=0.2, alpha=0.5)
    for x in np.arange(0, width_mm + 1, 10):
        ax.axvline(x, color="gray", lw=0.6, alpha=0.7)
    for y in np.arange(0, height_mm + 1, 10):
        ax.axhline(y, color="gray", lw=0.6, alpha=0.7)

    # --- Primitives ---
    ax.add_patch(plt.Rectangle((30, 30), 20, 20, facecolor="none", edgecolor="red", lw=2))
    ax.add_patch(plt.Circle((60, 40), 10, facecolor="none", edgecolor="blue", lw=2))
    tri = np.array([[10, 10], [20, 10], [15, 25]])
    ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

    # Render to BGR NumPy array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# =====================================================================
# B. Sensor simulation (noise + blur)
# =====================================================================

#   random_noise(image, mode="gaussian", var=0.01)  # Gaussian noise
#   random_noise(image, mode="poisson")             # Poisson shot noise
#   random_noise(image, mode="s&p", amount=0.02)    # Salt & pepper noise
#   random_noise(image, mode="speckle", var=0.05)   # Multiplicative speckle

def apply_sensor_noise(img, gaussian_std=0.02, poisson=True, blur_sigma=0.8):
    """Simulate the noisy raw sensor capture before optics."""
    img_f = util.img_as_float(img)

    # Gaussian noise (sensor read + photon statistics)
    noisy = random_noise(img_f, mode="gaussian", var=gaussian_std**2)

    # Poisson (shot) noise
    if poisson:
        noisy = random_noise(noisy, mode="poisson")

    # Small blur for focus/sensor diffusion
    blurred = filters.gaussian(noisy, sigma=blur_sigma, channel_axis=2)

    return util.img_as_ubyte(exposure.rescale_intensity(blurred))


# =====================================================================
# C. Camera distortions (optical)
# =====================================================================
def apply_camera_effects(img, tilt_x=0.15, tilt_y=0.10,
                         k1=-0.3, k2=0.05, light_grad=True):
    """Apply optical projection, lens distortion, and lighting falloff."""
    h, w = img.shape[:2]

    # --- Perspective projection (homography) ---
    dx = tilt_x * w
    dy = tilt_y * h
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

    # --- Radial lens distortion ---
    cx, cy = w / 2, h / 2
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d * x_d + y_d * y_d
    factor = 1 + k1 * r2 + k2 * r2 * r2
    factor = np.where(factor == 0.0, 1.0, factor)
    x_u = x_d / factor
    y_u = y_d / factor
    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)
    distorted = cv2.remap(persp, map_x, map_y,
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # --- Illumination gradient (simulated light falloff) ---
    if light_grad:
        grad = np.linspace(1.15, 0.85, h, dtype=np.float32).reshape(h, 1)
        grad = np.repeat(grad, w, axis=1)
        distorted = np.clip(distorted.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)

    return distorted


# =====================================================================
# D. Full synthetic photo generator
# =====================================================================
def generate_synthetic_photo():
    """Produce one physically faithful synthetic image."""
    ideal = render_scene()
    sensor_img = apply_sensor_noise(ideal, gaussian_std=0.02, poisson=True, blur_sigma=0.8)
    final_img = apply_camera_effects(sensor_img, tilt_x=0.18, tilt_y=0.1, k1=-0.25, k2=0.05)
    return final_img


if __name__ == "__main__":
    img = generate_synthetic_photo()
    cv2.imshow("Synthetic Lab Photo (Correct Sensor-First Model)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
