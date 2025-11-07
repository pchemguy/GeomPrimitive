"""
perspective_lens_opencv_demo.py
-------------------------------
Simulate perspective (camera not perpendicular) + radial lens distortion
on a flat grid with a rectangle and a circle, using OpenCV.

This version:
- Uses a proper 2D homography for perspective.
- Then applies an approximate radial distortion via cv2.remap.
- Distorts BOTH the grid and overlaid shapes.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# 1. Base image: grid + rectangle + circle
# ---------------------------------------------------------------------
def make_grid_image(width: int = 800,
                    height: int = 600,
                    step: int = 50) -> np.ndarray:
    """Create white background with gray grid, red rectangle, green circle."""
    img = np.ones((height, width, 3), np.uint8) * 255

    # Grid
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 1)

    # Rectangle (red)
    cv2.rectangle(img, (200, 150), (600, 450), (0, 0, 255), 2)

    # Circle (green)
    cv2.circle(img, (400, 300), 100, (0, 255, 0), 2)

    return img


# ---------------------------------------------------------------------
# 2. Perspective warp via homography
# ---------------------------------------------------------------------
def apply_perspective(img: np.ndarray,
                      tilt_x: float = 0.25,
                      tilt_y: float = 0.20) -> np.ndarray:
    """
    Apply a planar perspective transform (homography) to mimic camera tilt.

    tilt_x, tilt_y ~ 0.0 .. 0.4
      - tilt_x controls how much the top edge converges horizontally.
      - tilt_y controls how much the top is 'farther' (smaller vertically).
    """
    h, w = img.shape[:2]

    # Source corners (full image)
    src = np.float32([
        [0,       0      ],
        [w - 1.0, 0      ],
        [w - 1.0, h - 1.0],
        [0,       h - 1.0],
    ])

    # Destination corners: trapezoid inside the frame
    dx = tilt_x * w
    dy = tilt_y * h

    dst = np.float32([
        [0.0 + dx,    0.0 + dy],     # top-left pulled in & up
        [w - 1.0 - dx, 0.0 + dy / 2],  # top-right pulled in & slightly up
        [w - 1.0,     h - 1.0],      # bottom-right stays near corner
        [0.0,         h - 1.0 - dy], # bottom-left slightly up
    ])

    H = cv2.getPerspectiveTransform(src, dst)

    # Warp with same output size -> no clipping
    warped = cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped


# ---------------------------------------------------------------------
# 3. Radial lens distortion (approximate forward model via remap)
# ---------------------------------------------------------------------
def apply_radial_distortion(img: np.ndarray,
                            k1: float = -0.25,
                            k2: float = 0.0) -> np.ndarray:
    """
    Apply barrel (k1 < 0) or pincushion (k1 > 0) distortion
    using an approximate inverse radial mapping + cv2.remap.

    This treats the given image as the UNDISTORTED one and generates
    a distorted view.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Use the larger half-dimension to normalize radius
    r_norm = max(cx, cy)

    # Destination pixel grid (the distorted image coordinate system)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # Normalize to [-1, 1] around the center
    x_d = (xx - cx) / r_norm
    y_d = (yy - cy) / r_norm
    r2 = x_d * x_d + y_d * y_d

    # Forward radial model (undistorted -> distorted) is:
    #   x_d = x_u * (1 + k1*r^2 + k2*r^4)
    # For remap, we want dest -> source: we approximate inverse as:
    #   x_u ~ x_d / (1 + k1*r^2 + k2*r^4)
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    # Avoid division by zero in extreme cases
    factor = np.where(factor == 0.0, 1.0, factor)

    x_u = x_d / factor
    y_u = y_d / factor

    # Back to pixel coordinates in the *source* (undistorted) image
    map_x = (x_u * r_norm + cx).astype(np.float32)
    map_y = (y_u * r_norm + cy).astype(np.float32)

    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return distorted


# ---------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------
def main() -> None:
    base = make_grid_image()

    # Step 1: perspective (camera not perpendicular)
    persp = apply_perspective(base, tilt_x=0.25, tilt_y=0.20)

    # Step 2: lens distortion
    #   k1 < 0 -> barrel, k1 > 0 -> pincushion
    persp_barrel = apply_radial_distortion(persp, k1=-0.30, k2=0.05)
    persp_pincushion = apply_radial_distortion(persp, k1=+0.30, k2=-0.05)

    # Stack for visual comparison
    row1 = np.hstack((base, persp))
    row2 = np.hstack((persp_barrel, persp_pincushion))
    stacked = np.vstack((row1, row2))

    cv2.imshow(
        "Top: original | perspective   |   Bottom: perspective+barrel | perspective+pincushion",
        stacked,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
