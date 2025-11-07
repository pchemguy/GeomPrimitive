"""
spt_base_gradient.py
-----------
"""

from __future__ import annotations

import math
from typing import TypeAlias, Sequence, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ImageBGR:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) BGR order
ImageRGB:  TypeAlias = NDArray[np.uint8]  # (H, W, 3) RGB order
ImageRGBA: TypeAlias = NDArray[np.uint8]  # (H, W, 4) RGBA order
ImageRGBx: TypeAlias = Union[ImageRGB, ImageRGBA] # Either RGB or RGBA


def bgr_from_rgba(rgba: ImageRGBA) -> ImageBGR:
    """Convert RGBA (Matplotlib) to BGR (OpenCV)."""
    return rgba[..., :3][..., ::-1]


def rgb_from_bgr(bgr: ImageBGR) -> ImageRGB:
    """Convert BGR (OpenCV) to RGB (Matplotlib)."""
    return bgr[..., ::-1]


def show_RGBx_grid(images: dict[str, ImageRGBx],
                   title_style: dict = None, figsize_scale: float = 5) -> None:
    """
    Display multiple images in an automatically balanced rectangular grid.

    Layout rule:
        cols = ceil(sqrt(N))
        rows = ceil(N / cols)
    (Keeps the layout close to square, with longer side horizontal.)

    Args:
        images: Dictionary of <Title>:<Image>; Image - NumPy array (RGB or RGBA).
        title_style: Optional dict for Matplotlib title styling.
        figsize_scale: Multiplier for overall figure size.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images to display.")

    # --- Compute balanced grid ---
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig_w = cols * figsize_scale
    fig_h = rows * figsize_scale * 0.9
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)  # flatten axes array

    # --- Title style defaults ---
    style = dict(fontsize=16, fontweight="bold", color="green")
    if title_style:
        style.update(title_style)

    # --- Draw each image ---
    for (title, img), ax in zip(images.items(), axes):
        ax.imshow(img)
        ax.set_title(title, **style)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def add_grid(ax, width_mm=100, height_mm=80) -> None:
    """Add fine (1 mm) and major (10 mm) gridlines using LineCollection."""
    # Fine grid every 1 mm
    x_fine = np.arange(0, width_mm + 1, 1)
    y_fine = np.arange(0, height_mm + 1, 1)
    lines_fine = (
        [((x, 0), (x, height_mm)) for x in x_fine] +
        [((0, y), (width_mm, y)) for y in y_fine]
    )
    lc_fine = LineCollection(lines_fine, colors="gray", linewidths=0.2, alpha=0.5)
    ax.add_collection(lc_fine)

    # Major grid every 10 mm
    x_major = np.arange(0, width_mm + 1, 10)
    y_major = np.arange(0, height_mm + 1, 10)
    lines_major = (
        [((x, 0), (x, height_mm)) for x in x_major] +
        [((0, y), (width_mm, y)) for y in y_major]
    )
    lc_major = LineCollection(lines_major, colors="gray", linewidths=0.6, alpha=0.7)
    ax.add_collection(lc_major)


def render_scene(width_mm: float = 100, 
                 height_mm: float = 80,
                 dpi: int = 200) -> ImageRGBA:
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

    add_grid(ax, width_mm=100, height_mm=80)

    # Primitives: square, circle, triangle

    ax.add_patch(plt.Rectangle((30, 30), 20, 20, edgecolor="red", fill=False, lw=2))
    ax.add_patch(plt.Circle((70, 40), 10, edgecolor="blue", fill=False, lw=2))
    tri = np.array([[10, 10], [20, 10], [15, 25]])
    ax.fill(tri[:, 0], tri[:, 1], edgecolor="green", fill=False, lw=2)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return rgba


def apply_lighting_gradient(img: ImageBGR,
                            top_bright: float = 1.1,
                            bottom_dark: float = 0.9,
                            lighting_mode: str = "linear",
                            lighting_strength: float = 5,
                            gradient_angle: float = 90,
                            grad_cx: float = 0,
                            grad_cy: float = 0,
                           ) -> ImageBGR:
    """Apply lighting gradient (linear or radial, normalized)."""
    if (not lighting_mode) or (lighting_strength <= 1e-6):
        return img

    angle_rad = math.radians(gradient_angle)

    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    # --- Base gradient shape ---
    if lighting_mode == "linear":
        u = (np.cos(angle_rad) * (x / w) +
             np.sin(angle_rad) * (y / h))
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
    else:  # Radial
        cx = (0.5 + grad_cx) * w
        cy = (0.5 + grad_cy) * h
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        r_max = np.max(np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2))
        u = np.clip(r / (r_max + 1e-9), 0.0, 1.0)

    # --- Compute lighting map ---
    grad = top_bright + (bottom_dark - top_bright) * u
    # Interpolate between flat (1.0) and full gradient
    lighting = 1.0 + lighting_strength * (grad - 1.0)

    img_f = np.clip(img.astype(np.float32) * lighting[..., None], 0, 255)

    return img_f.astype(np.uint8)


def main():
    # ----------------------------------------------------------------------
    base_rgba: ImageRGBA = render_scene()
    base_bgr:  ImageBGR  = bgr_from_rgba(base_rgba)
    grad_bgr:  ImageBGR  = apply_lighting_gradient(
                               img=base_bgr,
                               top_bright=1.1,
                               bottom_dark=0.9,
                               lighting_mode="linear",
                               lighting_strength=4,
                               gradient_angle=90,
                               grad_cx=0,
                               grad_cy=0,
                           )
    default_props = {
        "img":               base_bgr,
        "top_bright":        1.1,
        "bottom_dark":       0.9,
        "lighting_mode":     "linear",
        "lighting_strength": 1,
        "gradient_angle":    90,
        "grad_cx":           0,
        "grad_cy":           0,
    }
    
    demos = {
        "Linear 90deg x 0": {"lighting_strength": 0, "gradient_angle": 90},
        "Linear 90deg x 1": {"lighting_strength": 1, "gradient_angle": 90},
        "Linear 90deg x 5": {"lighting_strength": 5, "gradient_angle": 90},
        "Linear 45deg x 5": {"lighting_strength": 5, "gradient_angle": 45},
    }

    for (title, custom_props) in demos.items():
        demos[title] = rgb_from_bgr(
            apply_lighting_gradient(**{**default_props, **custom_props})
        )

    show_RGBx_grid(demos)


if __name__ == "__main__":
    main()
