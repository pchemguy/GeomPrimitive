"""
interactive_perspective_lens.py
-------------------------------
Live interactive demo: perspective (camera tilt) + lens distortion.
Uses correct homography + radial distortion pipeline with safe refit.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Grid + shapes
# ---------------------------------------------------------------------
def make_grid_image(width=800, height=600, step=50):
    img = np.ones((height, width, 3), np.uint8) * 255
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 1)
    cv2.rectangle(img, (200, 150), (600, 450), (0, 0, 255), 2)
    cv2.circle(img, (400, 300), 100, (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------------
# Perspective transform
# ---------------------------------------------------------------------
def apply_perspective(img, tilt_x=0.2, tilt_y=0.2):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = tilt_x * w
    dy = tilt_y * h
    dst = np.float32([
        [0 + dx, 0 + dy],
        [w - 1 - dx, 0 + dy / 2],
        [w - 1, h - 1],
        [0, h - 1 - dy],
    ])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped


# ---------------------------------------------------------------------
# Radial distortion
# ---------------------------------------------------------------------
def apply_radial_distortion(img, k1=-0.3, k2=0.05):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r_norm = max(cx, cy)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
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
        img, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return distorted


# ---------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------
def main():
    base = make_grid_image()
    cv2.namedWindow("Camera Simulator", cv2.WINDOW_NORMAL)

    # Sliders (trackbars)
    def add_slider(name, minv, maxv, start):
        cv2.createTrackbar(name, "Camera Simulator", start, maxv, lambda v: None)
        return minv

    # tilt_x, tilt_y scaled 0-40 -> 0.00-0.40

    cv2.createTrackbar("Tilt X (x0.01)", "Camera Simulator", 20, 40, lambda v: None)
    cv2.createTrackbar("Tilt Y (x0.01)", "Camera Simulator", 20, 40, lambda v: None)
    # k1: 100..100 -> 1.00..1.00
    cv2.createTrackbar("k1 (x0.01)", "Camera Simulator", 70, 200, lambda v: None)
    cv2.createTrackbar("k2 (x0.01)", "Camera Simulator", 105, 200, lambda v: None)

    print("Adjust sliders: Tilt X/Y (perspective) and k1/k2 (lens). ESC to exit.")

    while True:
        tx = (cv2.getTrackbarPos("Tilt X (x0.01)", "Camera Simulator")) / 100.0
        ty = (cv2.getTrackbarPos("Tilt Y (x0.01)", "Camera Simulator")) / 100.0
        k1 = (cv2.getTrackbarPos("k1 (x0.01)", "Camera Simulator") - 100) / 100.0
        k2 = (cv2.getTrackbarPos("k2 (0.01)", "Camera Simulator") - 100) / 100.0

        persp = apply_perspective(base, tilt_x=tx, tilt_y=ty)
        warped = apply_radial_distortion(persp, k1=k1, k2=k2)

        cv2.imshow("Camera Simulator", warped)
        if cv2.waitKey(15) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
