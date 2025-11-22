"""
```
pet_grid_node_detector.py
-------------------------
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pet_histxy import plot_interactive_histogram


def find_grid_nodes(source_img, output_dir="output"):
    """
    Detects grid intersections (nodes) using Morphological Intersection.
    Returns: (N, 2) array of [x, y] coordinates.
    """
    if source_img is None: raise ValueError("Image is None")
    
    # 1. Pre-process (Same as your auto-crop pipeline)
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    
    # Sobel + Normalize
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(mag)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # =========================================================
    # THE NODE DETECTION LOGIC
    # =========================================================
    
    # 2. Define Directional Kernels
    # Length should be roughly 1/2 the grid period (e.g. 20px)
    # This filters out small noise (text) that isn't a long line.
    k_len = 25 
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    
    # 3. Morphological Opening (Erode -> Dilate)
    # "Opening" removes anything smaller than the kernel.
    # This leaves ONLY long horizontal lines and long vertical lines.
    lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_k, iterations=1)
    lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_k, iterations=1)
    
    # 4. Logical AND (The Intersection)
    # A pixel is a node ONLY if it belongs to BOTH a horizontal AND vertical line.
    intersections = cv2.bitwise_and(lines_h, lines_v)
    
    # 5. Clean up blobs
    # Dilate slightly to merge fragmented intersections
    intersections = cv2.dilate(intersections, np.ones((3,3)), iterations=1)

    # =========================================================
    # COORDINATE EXTRACTION
    # =========================================================
    
    # Connected Components finds the centroid of each intersection blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections)
    
    # Filter weak blobs (noise) by area
    # A real intersection should be at least 5-10 pixels area
    min_area = 5
    valid_indices = np.where(stats[:, cv2.CC_STAT_AREA] > min_area)[0]
    # Exclude background (index 0)
    valid_indices = valid_indices[valid_indices != 0]
    
    clean_centroids = centroids[valid_indices]
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    debug_img = source_img.copy()
    
    # Draw detected nodes
    for (cx, cy) in clean_centroids:
        cv2.circle(debug_img, (int(cx), int(cy)), 3, (0, 255, 0), -1) # Green dots
        
    cv2.imwrite(f"{output_dir}/debug_nodes_detected.jpg", debug_img)
    cv2.imwrite(f"{output_dir}/debug_nodes_mask.jpg", intersections)
    
    print(f"Detected {len(clean_centroids)} grid nodes.")
    return clean_centroids

# Usage
# nodes = find_grid_nodes(img)


# ... (Previous code for find_grid_nodes function) ...

if __name__ == "__main__":
    # 1. Define Input Path
    # Using the filename you provided earlier
    image_path = "photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00.jpg"
    #"photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00.jpg"
    
    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit()

    print(f"Processing {image_path}...")
    source_image = cv2.imread(image_path)
    
    if source_image is None:
        print("Error: Failed to decode image.")
        sys.exit()

    # 3. Run Node Detection
    # This will generate 'debug_nodes_detected.jpg' and 'debug_nodes_mask.jpg' in output/
    nodes = find_grid_nodes(source_image, output_dir="output")
    
    plot_interactive_histogram(nodes)

    print(f"Done. Found {len(nodes)} intersections.")
    print("Check 'output/' for visualization.")

"""
```
"""