"""
interactive_camera_full.py
--------------------------
Interactive simulation of a tilted, perspective camera
with focal length (FOV) and radial lens distortion control.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# 1. Synthetic grid with shapes
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
# 2. Project flat image under camera model
# ---------------------------------------------------------------------
def project_image(img, yaw_deg, pitch_deg, roll_deg,
                  focal, k1, k2, fit_output=True):
  h, w = img.shape[:2]
  cx, cy = w / 2, h / 2
  fx = fy = focal
  K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]], np.float32)
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

  if fit_output:
    # Auto-fit normalization so projection stays inside frame
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
# 3. Interactive UI
# ---------------------------------------------------------------------
def main():
  img = make_grid_image()
  cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)

  # Helper for symmetric sliders
  def make_slider(name, minv, maxv, start):
    cv2.createTrackbar(name, "Virtual Camera", int(start - minv),
                       int(maxv - minv), lambda v: None)
    return minv

  yaw_off = make_slider("Yaw (deg)", -45, 45, 15)
  pitch_off = make_slider("Pitch (deg)", -45, 45, 10)
  roll_off = make_slider("Roll (deg)", -45, 45, 5)
  f_off = make_slider("Focal (px)", 200, 1600, 800)
  k1_off = make_slider("k1 (x1e-2)", -100, 100, -25)
  k2_off = make_slider("k2 (x1e-2)", -100, 100, 10)

  print("Use sliders to adjust camera parameters. Press ESC to exit.")

  while True:
    yaw = cv2.getTrackbarPos("Yaw (deg)", "Virtual Camera") + yaw_off
    pitch = cv2.getTrackbarPos("Pitch (deg)", "Virtual Camera") + pitch_off
    roll = cv2.getTrackbarPos("Roll (deg)", "Virtual Camera") + roll_off
    focal = cv2.getTrackbarPos("Focal (px)", "Virtual Camera") + f_off
    k1 = (cv2.getTrackbarPos("k1 (x1e-2)", "Virtual Camera") + k1_off) / 100.0
    k2 = (cv2.getTrackbarPos("k2 (x1e-2)", "Virtual Camera") + k2_off) / 100.0

    warped = project_image(img, yaw, pitch, roll, focal, k1, k2)
    stacked = np.hstack((img, warped))
    cv2.imshow("Virtual Camera", stacked)

    if cv2.waitKey(10) & 0xFF == 27:
      break

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
