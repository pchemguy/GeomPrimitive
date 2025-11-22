"""
```
pet_grid_auto_crop.py
"""
import os
import math
import cv2
import numpy as np


def detect_grid_area_density(source_image, output_dir="output"):
    """
    1. Detects grid via Macro-Texture Density (Blur + Otsu).
    2. Saves debug steps (Sobel, CLAHE, Density Cloud).
    3. Rounds BBox UP to nearest 100px, centering the expansion 
       and shifting if boundaries are hit.
    """
    if source_image is None: raise ValueError("Source image is None.")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    h_img, w_img = source_image.shape[:2]

    # =========================================================
    # PHASE 1: PRE-PROCESSING (Sobel -> CLAHE)
    # =========================================================
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Sobel Edge Detection
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(os.path.join(output_dir, "step_1_sobel.jpg"), mag)
    print("Saved step_1_sobel.jpg")
    
    # 2. CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(mag)
    
    cv2.imwrite(os.path.join(output_dir, "step_2_clahe.jpg"), enhanced)
    print("Saved step_2_clahe.jpg")

    # =========================================================
    # PHASE 2: DENSITY ANALYSIS (Otsu -> Blur -> Threshold)
    # =========================================================
    
    # 3. Initial Binary Mask (High freq noise is white)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "step_3_otsu_raw.jpg"), binary)

    # 4. Macro-Density Blur (The "Cloud" Step)
    # We average over a large area (~5% of image width)
    # This turns grid lines into a solid gray block and noise into black.
    k_size = int(w_img * 0.05)
    density_map = cv2.blur(binary, (k_size, k_size))
    
    # Save visualization of the "Cloud"
    density_vis = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "step_4_density_cloud.jpg"), density_vis)
    
    # 5. Threshold the Cloud
    # Isolate the high-density grid region
    _, density_mask = cv2.threshold(density_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "step_5_density_mask.jpg"), density_mask)

    # =========================================================
    # PHASE 3: CONTOUR & INTELLIGENT EXPANSION
    # =========================================================
    contours, _ = cv2.findContours(density_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No density region found. Returning full image.")
        return 0, 0, w_img, h_img
        
    # Find largest region
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_raw, h_raw = cv2.boundingRect(largest_contour)
    
    # --- LOGIC: ROUND UP TO NEAREST 100 ---
    # Calculate Target Size
    target_w = int(math.ceil(w_raw / 100.0) * 100)
    target_h = int(math.ceil(h_raw / 100.0) * 100)
    
    # Calculate required padding total
    diff_w = target_w - w_raw
    diff_h = target_h - h_raw
    
    # Calculate new Top-Left (attempting to center the expansion)
    # We subtract half the difference from the current x, y
    new_x = x - (diff_w // 2)
    new_y = y - (diff_h // 2)
    
    # --- LOGIC: BOUNDARY CHECK & SLIDE ---
    # If we hit the left edge, we must stay at 0 (effectively adding all padding to the right)
    # If we hit the right edge, we must slide left to maintain the target width
    
    # X-Axis Check
    if new_x < 0:
        new_x = 0
    elif (new_x + target_w) > w_img:
        new_x = w_img - target_w
        if new_x < 0: new_x = 0 # Safety if image < 100px wide
        
    # Y-Axis Check
    if new_y < 0:
        new_y = 0
    elif (new_y + target_h) > h_img:
        new_y = h_img - target_h
        if new_y < 0: new_y = 0

    # Final Coordinates
    x_final = int(new_x)
    y_final = int(new_y)
    w_final = int(target_w)
    h_final = int(target_h)
    
    # Enforce boundaries one last time (truncation) in case target > image size
    x_max = min(w_img, x_final + w_final)
    y_max = min(h_img, y_final + h_final)
    
    # =========================================================
    # PHASE 4: DEBUG VISUALIZATION
    # =========================================================
    debug_img = source_image.copy()
    
    # Draw Raw Detection (Blue, Thin)
    cv2.rectangle(debug_img, (x, y), (x+w_raw, y+h_raw), (255, 0, 0), 2)
    
    # Draw Final Expanded Box (Green, Thick)
    cv2.rectangle(debug_img, (x_final, y_final), (x_max, y_max), (0, 255, 0), 3)
    
    # Text Labels
    label_raw = f"Raw: {w_raw}x{h_raw}"
    label_final = f"Expanded: {x_max-x_final}x{y_max-y_final}"
    
    cv2.putText(debug_img, label_raw, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(debug_img, label_final, (x_final, y_final - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, "step_6_final_rounded_crop.jpg"), debug_img)
    print(f"Saved step_6_final_rounded_crop.jpg")
    
    return x_final, y_final, x_max, y_max


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Load your image
    img_path = "photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00.jpg" # Update this path
    img = cv2.imread(img_path)
    
    if img is not None:
        bbox = detect_grid_area_density(img, output_dir="output")
        print(f"Final BBox: {bbox}")
    else:
        print("Image not found.")

"""
```
"""