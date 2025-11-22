import cv2
import numpy as np
import os

def apply_imagej_find_edges(img):
    """
    Replicates ImageJ 'Process -> Find Edges' using Sobel operator.
    """
    # 1. Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Calculate Gradients (Sobel)
    # We use float64 to avoid overflow during calculation
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Calculate Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # 4. Normalize back to 0-255 (8-bit)
    # ImageJ clips values, but normalizing is usually safer for visualization
    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return norm_magnitude

def detect_grid_area_sobel(source_img, output_dir="output", padding_pct=0.02):
    """
    Robust Grid Cropper using Sobel Edge Detection (ImageJ style).
    """
    if source_img is None:
        raise ValueError("Image is None")

    h, w = source_img.shape[:2]
    
    # --- STEP 1: IMAGEJ FIND EDGES ---
    edges = apply_imagej_find_edges(source_img)
    
    # Save debug to verify it matches your expectation
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "debug_1_sobel_edges.jpg"), edges)

    # --- STEP 2: MORPHOLOGICAL CLOSING ---
    # Now we have bright lines on dark background. 
    # We smear them together to form a solid block.
    
    # Threshold slightly to remove background noise (faint textures)
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    # Use a large kernel to bridge the gaps between grid lines
    # Grid cells look ~40px wide. A 25px kernel will bridge them.
    kernel_size = (25, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # "Close" operation = Dilate (connect) then Erode (restore size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Dilate a bit more to ensure the outer boundary is captured
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    cv2.imwrite(os.path.join(output_dir, "debug_2_morphology.jpg"), dilated)

    # --- STEP 3: FIND LARGEST CONTOUR ---
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No grid found.")
        return 0, 0, w, h

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    # --- STEP 4: PADDING ---
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)
    
    x_min = max(0, x - pad_x)
    y_min = max(0, y - pad_y)
    x_max = min(w, x + bw + pad_x)
    y_max = min(h, y + bh + pad_y)
    
    # --- DEBUG OVERLAY ---
    debug_img = source_img.copy()
    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "grid_crop_final.jpg"), debug_img)
    print(f"Sobel-based crop saved to: {output_dir}/grid_crop_final.jpg")
    
    return x_min, y_min, x_max, y_max
