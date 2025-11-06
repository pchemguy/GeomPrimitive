"""
interactive_camera_simulation.py
--------------------------------
Interactive demo: simulate camera tilt + perspective + lens distortion
with automatic bounding and OpenCV trackbars.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# 1. Generate grid + shapes
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
# 2. Compute camera projection with safe normalization
# ---------------------------------------------------------------------
def project_image(img, yaw_deg, pitch_deg, roll_deg, k1, k2):
  h, w = img.shape[:2]
  fx = fy = 800
  cx, cy = w / 2, h / 2
  K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
  dist = np.array([k1, k2, 0.0, 0.0, 0.0], np.float32)

  yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
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

  tvec = np.array([[0], [0], [800]], np.float32)

  yy, xx = np.indices((h, w), np.float32)
  pts = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()], axis=-1)
  pts2d, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], tvec, K, dist)
  pts2d = pts2d.reshape(h, w, 2)

  # --- Compute bounding box of projected coordinates
  xmin, xmax = np.nanmin(pts2d[..., 0]), np.nanmax(pts2d[..., 0])
  ymin, ymax = np.nanmin(pts2d[..., 1]), np.nanmax(pts2d[..., 1])
  sx = w / (xmax - xmin)
  sy = h / (ymax - ymin)
  scale = min(sx, sy)
  pts2d[..., 0] = (pts2d[..., 0] - xmin) * scale
  pts2d[..., 1] = (pts2d[..., 1] - ymin) * scale

  mapx = np.clip(pts2d[..., 0].astype(np.float32), 0, w - 1)
  mapy = np.clip(pts2d[..., 1].astype(np.float32), 0, h - 1)
  warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
  return warped


# ---------------------------------------------------------------------
# 3. Interactive control loop
# ---------------------------------------------------------------------
def main():
  img = make_grid_image()
  cv2.namedWindow("Camera Simulation", cv2.WINDOW_NORMAL)

  def add_slider(name, minv, maxv, start):
    cv2.createTrackbar(name, "Camera Simulation", start, maxv - minv, lambda v: None)
    # store offset so we can map to signed range
    return minv

  yaw_off = add_slider("Yaw", 0, 90, 25)
  pitch_off = add_slider("Pitch", 0, 90, 15)
  roll_off = add_slider("Roll", 0, 90, 5)
  k1_off = add_slider("k1 (x1e-3)", 0, 100, 30)
  k2_off = add_slider("k2 (x1e-5)", 0, 100, 10)

  print("Adjust sliders: Yaw, Pitch, Roll, k1, k2 (press ESC to exit)")

  while True:
    yaw = cv2.getTrackbarPos("Yaw", "Camera Simulation") + yaw_off
    pitch = cv2.getTrackbarPos("Pitch", "Camera Simulation") + pitch_off
    roll = cv2.getTrackbarPos("Roll", "Camera Simulation") + roll_off
    k1 = (cv2.getTrackbarPos("k1 (x1e-5)", "Camera Simulation") - 50) / 1e5
    k2 = (cv2.getTrackbarPos("k2 (x1e-2)", "Camera Simulation") - 50) / 1e10

    warped = project_image(img, yaw, pitch, roll, k1, k2)
    stacked = np.hstack((img, warped))
    cv2.imshow("Camera Simulation", stacked)

    if cv2.waitKey(10) & 0xFF == 27:
      break

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
