"""
perspective_lens_demo.py
------------------------
Demonstrate combined perspective and lens distortion
using scikit-image's warp and projective transform.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, draw, util


# ---------------------------------------------------------------------
# 1. Create synthetic "graph paper" base
# ---------------------------------------------------------------------
def make_grid_image(W=400, H=300, d_major=40, d_minor=10):
    img = np.ones((H, W, 3), dtype=float)
    for y in range(0, H, d_minor):
        color = 0.8 if y % d_major else 0.5
        img[y, :, :] = color
    for x in range(0, W, d_minor):
        color = 0.8 if x % d_major else 0.5
        img[:, x, :] = color
    return img


# ---------------------------------------------------------------------
# 2. Draw rectangle and circle on top
# ---------------------------------------------------------------------
def draw_shapes(image):
    H, W, _ = image.shape
    rr, cc = draw.rectangle(
        start=(H // 4, W // 4),
        end=(3 * H // 4, 3 * W // 4),
        shape=image.shape,
    )
    image[rr, cc, 0] = 0.1  # red rectangle overlay

    rr, cc = draw.disk(
        center=(H // 2, W // 2),
        radius=min(H, W) // 6,
        shape=image.shape,
    )
    image[rr, cc, 1] = 0.1  # green circle overlay
    return image


# ---------------------------------------------------------------------
# 3. Perspective transform
# ---------------------------------------------------------------------
def make_perspective(H, W):
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.array([
        [60, 20],
        [W - 40, 0],
        [W - 10, H - 30],
        [30, H - 10],
    ])
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    return tform


# ---------------------------------------------------------------------
# 4. Radial (lens) distortion mapping
# ---------------------------------------------------------------------
def radial_distortion(coords, k1=-1.2e-6, k2=0.0):
    """Barrel distortion: negative k1 bulges outward."""
    x, y = coords.T
    cx, cy = 0.5, 0.5
    x = x - cx
    y = y - cy
    r2 = x**2 + y**2
    factor = 1 + k1 * r2 + k2 * r2**2
    x *= factor
    y *= factor
    x += cx
    y += cy
    return np.column_stack((x, y))


# ---------------------------------------------------------------------
# 5. Compose both transforms
# ---------------------------------------------------------------------
def combined_transform(coords, persp_tform, k1=-1.2e-6, k2=0.0):
    coords = persp_tform(coords)
    return radial_distortion(coords, k1, k2)


# ---------------------------------------------------------------------
# 6. Demo run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    grid = make_grid_image()
    grid = draw_shapes(grid)
    grid = util.img_as_float(grid)

    H, W, _ = grid.shape
    persp_tform = make_perspective(H, W)

    warped = transform.warp(
        grid,
        lambda c: combined_transform(c, persp_tform, k1=-2e-6),
        output_shape=(H, W),
        order=1,
        mode="edge",
    )

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(grid)
    axes[0].set_title("Original grid + shapes")
    axes[1].imshow(warped)
    axes[1].set_title("Perspective + lens distortion")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
