"""
pet_segmentation_composite3.py
-----------------------------

https://gemini.google.com/app/2021246f4a867b0d
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_percentile_analysis(image_path):
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # =================================================================
    # STEPS 1-3: The Robust Segmentation Pipeline
    # =================================================================
    
    # 1. Lab-'a' Initial Mask
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. GrabCut Refinement
    gc_mask_init = np.zeros(img_bgr.shape[:2], np.uint8)
    gc_mask_init[rough_mask > 0] = cv2.GC_PR_FGD
    kernel_anchor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(rough_mask, kernel_anchor, iterations=4)
    gc_mask_init[sure_fg > 0] = cv2.GC_FGD 

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    gc_result_mask = np.where((gc_mask_init == 2) | (gc_mask_init == 0), 0, 255).astype('uint8')

    # 3. Morphological Cleanup
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(gc_result_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 4. Dominant Region Selection
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_dominant_mask = np.zeros_like(cleaned_mask)
    
    rect_box = None
    rect_area = 0
    largest_contour = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_dominant_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Calculate Minimal Area Rect
        rect = cv2.minAreaRect(largest_contour)
        rect_area = rect[1][0] * rect[1][1]
        rect_box = np.int32(cv2.boxPoints(rect))
    else:
        print("Warning: No objects found.")
        return

    # =================================================================
    # STEP 5: Percentile Intersection Analysis
    # =================================================================
    # Logic: Mask = (Low Saturation) AND (Extreme Lightness)
    
    # Prepare Data
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # We already have 'lab', grab L channel (Channel 0)
    l_channel = lab[:, :, 0]

    # Extract pixels ONLY within the final mask
    mask_bool = final_dominant_mask > 0
    s_roi = s_channel[mask_bool]
    l_roi = l_channel[mask_bool]
    
    percentile_mask = np.zeros_like(final_dominant_mask)
    
    # Stats for plotting later
    s_p25, l_p10, l_p85 = 0, 0, 0

    if len(s_roi) > 0:
        # Calculate Percentiles
        s_p25 = np.percentile(s_roi, 25)
        l_p10 = np.percentile(l_roi, 10)
        l_p85 = np.percentile(l_roi, 85)

        # Create Condition Masks (Full Image size)
        # 1. Bottom 10% Saturation
        cond_s = s_channel <= s_p25
        
        # 2. Union of Bottom 10% and Top 10% Lightness
        cond_l = (l_channel <= l_p10) | (l_channel >= l_p85)
        
        # 3. Intersection of Both + Must be inside Final Mask
        combined_logic = cond_s & cond_l & mask_bool
        
        percentile_mask[combined_logic] = 255

    # =================================================================
    # Visualization
    # =================================================================
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("Segmentation + Minimal BBox + Percentile Analysis", fontsize=16)

    # --- Plot 1: Initial Mask ---
    ax1 = plt.subplot(2, 4, 1); ax1.imshow(rough_mask, cmap='gray'); ax1.set_title("1. Initial Lab-a"); ax1.axis('off')

    # --- Plot 2: GrabCut ---
    ax2 = plt.subplot(2, 4, 2); ax2.imshow(gc_result_mask, cmap='gray'); ax2.set_title("2. GrabCut"); ax2.axis('off')

    # --- Plot 3: Cleanup ---
    ax3 = plt.subplot(2, 4, 3); ax3.imshow(cleaned_mask, cmap='gray'); ax3.set_title("3. Cleanup"); ax3.axis('off')

    # --- Plot 4: Final + Min BBox (Existing Visual) ---
    ax4 = plt.subplot(2, 4, 4)
    vis_overlay = img_rgb.copy()
    
    # Green Overlay
    if np.any(mask_bool):
        roi = vis_overlay[mask_bool]
        green_block = np.zeros_like(roi)
        green_block[:] = [0, 255, 0] 
        vis_overlay[mask_bool] = cv2.addWeighted(roi, 0.6, green_block, 0.4, 0)

    # Blue Minimal Box
    if rect_box is not None:
        cv2.drawContours(vis_overlay, [rect_box], 0, (0, 0, 255), 2)
        
    ax4.imshow(vis_overlay)
    ax4.set_title(f"4. Final + Min BBox (Blue)\nArea: {int(rect_area)}px")
    ax4.axis('off')

    # --- Plot 5: The NEW Percentile Mask (Visual) ---
    ax5 = plt.subplot(2, 4, 5)
    vis_percentile = img_rgb.copy()
    
    # Draw the percentile pixels in MAGENTA (High Contrast)
    perc_indices = percentile_mask > 0
    if np.any(perc_indices):
        roi_p = vis_percentile[perc_indices]
        magenta_block = np.zeros_like(roi_p)
        magenta_block[:] = [255, 0, 255] # Magenta
        vis_percentile[perc_indices] = cv2.addWeighted(roi_p, 0.2, magenta_block, 0.8, 0)
        
    # Draw yellow contour of original object for context
    if largest_contour is not None:
        cv2.drawContours(vis_percentile, [largest_contour], -1, (255, 255, 0), 1)

    ax5.imshow(vis_percentile)
    ax5.set_title("5. Percentile Intersection\n(Magenta = Low S + Extreme L)")
    ax5.axis('off')

    # --- Plot 6: Saturation Histogram ---
    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(s_roi, bins=32, range=[0, 256], color='gray', alpha=0.5)
    # Highlight the bottom 10%
    ax6.axvline(s_p25, color='r', linestyle='--', linewidth=2, label=f'25th %: {s_p25:.1f}')
    # Shade the selected area
    ax6.axvspan(0, s_p25, color='red', alpha=0.2)
    ax6.set_title("6. Saturation (HSV)\nRed zone = Selected")
    ax6.legend(loc='upper right', fontsize='small')
    ax6.set_xlim([0, 256])

    # --- Plot 7: Lightness Histogram ---
    ax7 = plt.subplot(2, 4, 7)
    ax7.hist(l_roi, bins=32, range=[0, 256], color='gray', alpha=0.5)
    # Highlight bottom 10 and top 10
    ax7.axvline(l_p10, color='b', linestyle='--', label=f'10th: {l_p10:.1f}')
    ax7.axvline(l_p85, color='b', linestyle='--', label=f'85th: {l_p85:.1f}')
    # Shade selected areas
    ax7.axvspan(0, l_p10, color='blue', alpha=0.2)
    ax7.axvspan(l_p85, 255, color='blue', alpha=0.2)
    ax7.set_title("7. Lightness (Lab)\nBlue zone = Selected")
    ax7.legend(loc='upper center', fontsize='small')
    ax7.set_xlim([0, 256])

    # --- Plot 8: Final Text Summary ---
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    info_text = (
        f"ANALYSIS SUMMARY\n"
        f"----------------\n"
        f"Total Object Area: {cv2.countNonZero(final_dominant_mask)} px\n"
        f"Targeted Pixels:   {cv2.countNonZero(percentile_mask)} px\n\n"
        f"THRESHOLDS (Calculated):\n"
        f"S < {s_p25:.1f} (Bottom 25%)\n"
        f"AND\n"
        f"L < {l_p10:.1f} OR L > {l_p85:.1f}\n"
        f"(Extreme Darks/Lights)"
    )
    ax8.text(0.1, 0.5, info_text, fontsize=12, family='monospace', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    segmentation_percentile_analysis("photo_2025-11-17_23-50-05.jpg")
    # segmentation_percentile_analysis("Figure_1.png")
