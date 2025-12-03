"""
pet_segmentation_composite2.py
-----------------------------

https://gemini.google.com/app/2021246f4a867b0d
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_with_min_bbox(image_path):
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # =================================================================
    # STEP 1: Initial Mask (Lab-'a')
    # =================================================================
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # =================================================================
    # STEP 2: GrabCut Refinement
    # =================================================================
    gc_mask_init = np.zeros(img_bgr.shape[:2], np.uint8)
    gc_mask_init[rough_mask > 0] = cv2.GC_PR_FGD
    
    kernel_anchor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(rough_mask, kernel_anchor, iterations=4)
    gc_mask_init[sure_fg > 0] = cv2.GC_FGD 

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    gc_result_mask = np.where((gc_mask_init == 2) | (gc_mask_init == 0), 0, 255).astype('uint8')

    # =================================================================
    # STEP 3: Morphological Cleanup
    # =================================================================
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(gc_result_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # =================================================================
    # STEP 4: Dominant Region & Minimal BBox
    # =================================================================
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_dominant_mask = np.zeros_like(cleaned_mask)
    
    # Store the minimal rect points for visualization
    box_points = None
    rect_area = 0
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw it filled on the final mask
        cv2.drawContours(final_dominant_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # --- NEW: Calculate Minimal Area Rectangle (Rotated) ---
        # returns (center(x,y), (width, height), angle of rotation)
        rect = cv2.minAreaRect(largest_contour)
        rect_area = rect[1][0] * rect[1][1]
        
        # Convert to 4 corner points
        box_points = cv2.boxPoints(rect)
        # Convert float coordinates to integer for drawing
        box_points = np.int32(box_points)
    else:
        print("Warning: No objects found.")

    # =================================================================
    # Visualization
    # =================================================================
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("Segmentation + Minimal Area BBox", fontsize=16)

    # Intermediate Steps
    ax1 = plt.subplot(2, 3, 1); ax1.imshow(rough_mask, cmap='gray'); ax1.set_title("1. Lab-a Mask"); ax1.axis('off')
    ax2 = plt.subplot(2, 3, 2); ax2.imshow(gc_result_mask, cmap='gray'); ax2.set_title("2. GrabCut"); ax2.axis('off')
    ax3 = plt.subplot(2, 3, 3); ax3.imshow(cleaned_mask, cmap='gray'); ax3.set_title("3. Cleanup"); ax3.axis('off')

    # Final Mask
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(final_dominant_mask, cmap='gray')
    ax4.set_title("4. Final Dominant Mask")
    ax4.axis('off')

    # Overlay with BBox
    ax5 = plt.subplot(2, 3, 5)
    vis_overlay = img_rgb.copy()
    
    # Green Overlay
    mask_indices = final_dominant_mask > 0
    if np.any(mask_indices):
        roi = vis_overlay[mask_indices]
        green_block = np.zeros_like(roi)
        green_block[:] = [0, 255, 0] 
        vis_overlay[mask_indices] = cv2.addWeighted(roi, 0.6, green_block, 0.4, 0)

    # Yellow Contour (The Organ Boundary)
    if contours:
        cv2.drawContours(vis_overlay, [largest_contour], -1, (255, 255, 0), 2)

    # --- NEW: Draw the Rotated Bounding Box (Blue) ---
    if box_points is not None:
        cv2.drawContours(vis_overlay, [box_points], 0, (0, 0, 255), 2)
        # Label the area
        # Use top-left corner of the box for text
        start_point = tuple(box_points[1]) 
        cv2.putText(vis_overlay, f"MinArea: {int(rect_area)} px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    ax5.imshow(vis_overlay)
    ax5.set_title("5. Final + Min BBox (Blue)")
    ax5.axis('off')

    # Saturation Histogram
    ax6 = plt.subplot(2, 3, 6)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    masked_pixels = s_channel[final_dominant_mask > 0]
    
    if len(masked_pixels) > 0:
        ax6.hist(masked_pixels, bins=32, range=[0, 256], color='magenta', alpha=0.7, edgecolor='black')
        ax6.axvline(np.mean(masked_pixels), color='k', linestyle='--', label='Mean')
        ax6.legend()
    
    ax6.set_title("6. Saturation Histo (Region Only)")
    ax6.set_xlim([0, 256])
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    segmentation_with_min_bbox("photo_2025-11-17_23-50-05.jpg")
