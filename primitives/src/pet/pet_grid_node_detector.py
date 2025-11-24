"""
pet_grid_node_detector.py
-------------------------

Grid Node Detector (Morphological Line-Intersection Method)
===========================================================

This module implements a robust, calibration-free detector of grid
intersections ("nodes") in laboratory photographs of square or
rectangular line grids. It uses a classical morphological approach to
isolate long horizontal and vertical line segments and then extracts
their intersection points via connected-component analysis.

The detector is designed for:
    - auto-cropped photographs of graph paper, calibration grids,
      cross-hair overlays, and printed engineering grids,
    - low to moderate geometric distortions (rotation, skew, mild radial),
    - images in which grid lines visually dominate over text, digits,
      specks, or dirt.

Preprocessing Pipeline
----------------------
The detector begins with:

    1) Convert to grayscale
    2) Compute gradient magnitude using Sobel operators
    3) Apply CLAHE for local contrast normalization
    4) Otsu threshold -> binary mask suitable for morphological filtering

Directional Morphology
----------------------
After binarization, the core of the algorithm isolates grid lines using
directional morphological *opening* with long, thin structuring
elements. Two kernels are used:

    horiz_k = (k_len x 1)   - isolates long horizontal strokes  
    vert_k  = (1 x k_len)   - isolates long vertical strokes

A binary object must contain at least `k_len` contiguous pixels in the
direction of the kernel to survive erosion; therefore:

    - Small blobs (text, digits, dust, noise) are removed.
    - Only long, coherent horizontal or vertical grid segments remain.

The parameter `k_len` represents the expected minimum line length the
kernel should detect. In practice it should be on the order of
~ 40-70 % of the grid pitch. At present this value is **manually set**
in the implementation and may require adjustment for grids with widely
varying spacing or resolution. Automatic estimation of this parameter is
an open task for future development (see project notes).

Intersection Extraction
-----------------------
Logical AND of the horizontal and vertical line maps yields a sparse map
of potential node locations. Connected-component analysis is then used
to compute centroids of the surviving intersection blobs. These are
returned as an (N, 2) array of (x, y) coordinates in image space.

Debug Output
------------
Two diagnostic visualizations are automatically saved to the output
directory:

    - debug_nodes_detected.jpg - source image annotated with detected nodes
    - debug_nodes_mask.jpg     - binary mask of intersection blobs
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_grid_nodes(source_img, output_dir="output"):
    """
    Detect grid intersection points (nodes) in a photographed line grid
    using a purely morphological line-intersection method.

    Parameters
    ----------
    source_img : np.ndarray (H, W, 3), uint8 BGR
        Input image as loaded by OpenCV (BGR order). The image must be
        non-empty and already cropped to the grid region, or at least the
        grid must dominate the frame.
    output_dir : str, optional
        Directory where debug visualizations will be written:
            - debug_nodes_detected.jpg
            - debug_nodes_mask.jpg

    Returns
    -------
    nodes : np.ndarray of shape (N, 2), float64
        Array of (x, y) centroids for all detected grid intersections.
        Coordinates are in the image coordinate system (OpenCV convention:
        x = column, y = row).

    Pipeline Overview
    -----------------
    1. **Preprocessing**
       - Convert to grayscale  
       - Sobel gradients -> gradient magnitude  
       - CLAHE contrast enhancement  
       - Otsu global thresholding -> binary mask  

       This reduces lighting variations and enhances line continuity.

    2. **Directional Morphological Opening**
       Two line-shaped structuring elements isolate the dominant grid:
           k_len ~ 20-30 pixels  (~ half grid spacing)
           horiz_k = (k_len x 1)
           vert_k  = (1 x k_len)

       - Morphological opening removes anything shorter than k_len,
         effectively suppressing text, stains, digits, and specks.
       - Produces:
            lines_h  - horizontal features
            lines_v  - vertical features

    3. **Logical Intersection**
       A pixel belongs to a grid node only if it is simultaneously part
       of a horizontal and a vertical line:
           intersections = AND(lines_h, lines_v)

       A light dilation merges fragmented intersections into single blobs.

    4. **Connected Components -> Centroids**
       - Extract connected components
       - Filter out blobs smaller than a hard threshold (min_area = 5)
       - Compute geometric centroids -> final node positions

    5. **Debug Visualization**
       - Draw each centroid as a green circle on the original image
       - Save the annotated result and the intersection mask

    Notes
    -----
    - This method is robust to moderate rotation, perspective skew,
      non-uniform illumination, and typical smartphone noise.
    - For extremely distorted grids (strong perspective / curved lines),
      prefer LSD-based or Hough-family detectors.

    Examples
    --------
    >>> img = cv2.imread("grid.jpg")
    >>> nodes = find_grid_nodes(img, output_dir="debug")
    >>> print(nodes.shape)
    (412, 2)
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


if __name__ == "__main__":
    from pet_histxy import plot_interactive_histogram
    from pet_kde_interactive import plot_kde_interactive
    from pet_grid_optimizer import GridOptimizer

    from pet_grid_nodes_bbox import (
        get_grid_bbox, plot_grid_bbox, diagnose_and_fix_eps, get_histogram_pitch_ex,
        reject_outliers, get_bbox_angle, rotate_points_ccw,
    )


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

    # eps = diagnose_and_fix_eps(nodes)

    nodes_rotated = rotate_points_ccw(nodes, 45)
    get_histogram_pitch_ex(nodes_rotated)
    bbox, _, _, labels = get_grid_bbox(nodes_rotated)
    angle = get_bbox_angle(bbox)
    print(f"bbox angle: {angle}")
    plot_grid_bbox(nodes_rotated, bbox, labels)

    cleaned_nodes = reject_outliers(nodes_rotated, bbox, labels)

    grid_opt = GridOptimizer(cleaned_nodes, bw=None)
    # grid_opt.kde_table(angle_center=43, sweep_width=10.0, angle_steps=50, filename="grid_kde_sweep.csv")
    grid_opt.plot_360_landscape()
    grid_opt.plot_360_landscape_std()
    angles, locs, scores = grid_opt.analyze_twist_profile(41.0, 45.0, step=0.2)
    x_locs, angles, scores = grid_opt.analyze_spatial_twist(angle_center=43.0, search_width=10.0, num_slices=8)
    grid_opt.analyze_spatial_profile_std(angle_center=43.0, sweep_width=10.0, num_slices=4, angle_steps=20)
    
    results = grid_opt.plot_quartile_optimization_report(bbox_aux_angle=False, plot=True)
    print(results)
    
    
    angle_q1, entropy_q1, gini_q1 = grid_opt.optimize_quartile(0, initial_angle=angle, search_width=20, debug=True)
    angle_q4, entropy_q4, gini_q4 = grid_opt.optimize_quartile(3, initial_angle=angle, search_width=20, debug=True)
    print(f"Q1 Opt Angle: {angle_q1:.1f} | Entropy: {entropy_q1:.1f} | Gini: {gini_q1:.1f}")
    print(f"Q4 Opt Angle: {angle_q4:.1f} | Entropy: {entropy_q4:.1f} | Gini: {gini_q4:.1f}")

    angle_q1, stddev_q1, gini_q1 = grid_opt.optimize_quartile_std(0, initial_angle=angle, search_width=20, debug=True)
    angle_q4, stddev_q4, gini_q4 = grid_opt.optimize_quartile_std(3, initial_angle=angle, search_width=20, debug=True)
    print(f"Q1 Opt Angle: {angle_q1:.1f} | StdDev * 1K: {stddev_q1 * 1000:.2f} | Gini: {gini_q1:.1f}")
    print(f"Q4 Opt Angle: {angle_q4:.1f} | StdDev * 1K: {stddev_q4 * 1000:.2f} | Gini: {gini_q4:.1f}")

    #plot_interactive_histogram(nodes)
    # Important: set KDE bandwidth to 5%-10% of estimated
    #            pitch (2nd, 3rd neighbor distance, 90th percentile)
    plot_kde_interactive(cleaned_nodes, bw=2)

    print(f"Done. Found {len(nodes_rotated)} intersections.")
    print("Check 'output/' for visualization.")

"""
```
"""
