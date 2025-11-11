"""
camera_simulation_demo.py
-------------------------
Simulate camera tilt, perspective, and radial lens distortion
on a flat 2D grid with overlaid rectangle and circle.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# 1. Create synthetic grid + shapes
# ---------------------------------------------------------------------
def make_grid_image(width=800, height=600, step=50):
    img = np.ones((height, width, 3), np.uint8) * 255
    for x in range(0, width, step):
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(0, height, step):
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 1)

    # Red rectangle
    cv2.rectangle(img, (200, 150), (600, 450), (0, 0, 255), 2)
    # Green circle
    cv2.circle(img, (400, 300), 100, (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------------
# 2. Project 3D points to 2D with rotation, translation, distortion
# ---------------------------------------------------------------------
def project_image(img, yaw_deg=20, pitch_deg=15, roll_deg=0,
                  k1=-0.25, k2=0.10, p1=0.0, p2=0.0, k3=0.0):
    h, w = img.shape[:2]

    # Camera intrinsic matrix
    fx = fy = 800
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    # Rotation vectors (convert from degrees)
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

    # Compose rotation matrix (Rz * Ry * Rx)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Translation (camera moving back from plane)
    tvec = np.array([[0], [0], [800]], dtype=np.float32)

    # Define 3D points corresponding to every pixel in source image
    yy, xx = np.indices((h, w), dtype=np.float32)
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()], axis=-1)

    # Project points using cv2.projectPoints
    pts2d, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], tvec, K, dist)
    pts2d = pts2d.reshape(h, w, 2)

    # Map original image to new coordinates
    mapx = pts2d[..., 0].astype(np.float32)
    mapy = pts2d[..., 1].astype(np.float32)
    warped = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped


# ---------------------------------------------------------------------
# 3. Run the demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    img = make_grid_image()
    warped = project_image(
        img,
        yaw_deg=25,   # horizontal rotation (Y axis)
        pitch_deg=15, # vertical rotation (X axis)
        roll_deg=5,   # rotation around camera axis
        k1=-0.25, k2=0.10  # barrel distortion
    )

    stacked = np.hstack((img, warped))
    cv2.imshow("Original | Tilt + Perspective + Lens Distortion", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
