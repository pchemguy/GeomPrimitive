"""
pet_segmentation_composite.py
-----------------------------

https://gemini.google.com/app/2021246f4a867b0d
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_pipeline_final(image_path):
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # =================================================================
    # STEP 1: Initial Mask using Lab-'a' channel
    # =================================================================
    # Rationale: 'a' channel separates green-red well. Good for rough start.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    
    # Otsu thresholding automatically finds the split between background and red things
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # =================================================================
    # STEP 2: Refine with GrabCut
    # =================================================================
    # Rationale: Uses color statistics to snap boundaries to edges, improving the rough Lab mask.
    
    # Initialize GrabCut mask image. 0 = Definite BG, 2 = Probable BG, 3 = Probable FG, 1 = Definite FG
    gc_mask_init = np.zeros(img_bgr.shape[:2], np.uint8)
    
    # Mark the rough mask area as "Probable Foreground" (3)
    gc_mask_init[rough_mask > 0] = cv2.GC_PR_FGD
    
    # Create "Sure Foreground" anchors by eroding the rough mask deeply.
    # This tells GrabCut: "These central pixels are definitely the object, learn their color."
    kernel_anchor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(rough_mask, kernel_anchor, iterations=4)
    gc_mask_init[sure_fg > 0] = cv2.GC_FGD # Mark as Definite Foreground (1)

    # Run GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Run 5 iterations in mask initialization mode
    cv2.grabCut(img_bgr, gc_mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    # Extract final mask: Keep Probable FG (3) and Definite FG (1)
    gc_result_mask = np.where((gc_mask_init == 2) | (gc_mask_init == 0), 0, 255).astype('uint8')

    # =================================================================
    # STEP 3: Morphological Cleanup (Open/Close)
    # =================================================================
    # Rationale: Smooth boundaries, fill small holes, remove small noise specs.
    
    # Use an elliptical kernel for smooth, organic shapes
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Closing: Fills small holes inside the object
    cleaned_mask = cv2.morphologyEx(gc_result_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    # Opening: Removes small noise specs outside and smooths edges
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # =================================================================
    # STEP 4: Select Dominant Region
    # =================================================================
    # Rationale: The previous steps might still leave disconnected blood splashes. 
    # We assume the organ is the largest contiguous object.
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_dominant_mask = np.zeros_like(cleaned_mask)
    
    if contours:
        # Find the single largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw only that largest contour filled (thickness=-1) onto the fresh mask
        cv2.drawContours(final_dominant_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        print("Warning: No objects detected after cleanup.")

    # =================================================================
    # Visualization Setup
    # =================================================================
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("Segmentation Pipeline & Analysis", fontsize=16)

    # Row 1: Intermediate Masks
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rough_mask, cmap='gray')
    ax1.set_title("1. Initial Lab-'a' Mask\n(Rough, catches everything red)")
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(gc_result_mask, cmap='gray')
    ax2.set_title("2. GrabCut Refinement\n(Snaps boundaries based on color)")
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(cleaned_mask, cmap='gray')
    ax3.set_title("3. Morphological Cleanup\n(Close/Open to smooth & fill)")
    ax3.axis('off')

    # Row 2: Final Results & Analysis
    
    # --- Final Mask Only ---
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(final_dominant_mask, cmap='gray')
    ax4.set_title("4. Final Dominant Region\n(Largest connected component only)")
    ax4.axis('off')

    # --- Final Overlay ---
    ax5 = plt.subplot(2, 3, 5)
    # Create visualization: Original image + transparent green overlay
    vis_overlay = img_rgb.copy()
    
    # 1. Extract ROI where mask is active
    mask_indices = final_dominant_mask > 0
    if np.any(mask_indices):
        roi = vis_overlay[mask_indices]
        # 2. Create a solid green block same size as ROI
        green_block = np.zeros_like(roi)
        green_block[:] = [0, 255, 0] # Green in RGB
        # 3. Blend them (addWeighted handles same-size arrays)
        blended = cv2.addWeighted(roi, 0.6, green_block, 0.4, 0)
        # 4. Put blended pixels back
        vis_overlay[mask_indices] = blended

    # Draw a yellow contour boundary for sharpness
    contours_dom, _ = cv2.findContours(final_dominant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_overlay, contours_dom, -1, (255, 255, 0), 2)

    ax5.imshow(vis_overlay)
    ax5.set_title("5. Final Overlay\n(Green fill + Yellow border)")
    ax5.axis('off')

    # --- Saturation Histogram of Selected Region ---
    ax6 = plt.subplot(2, 3, 6)
    
    # Convert original to HSV to get Saturation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # Extract Saturation values ONLY where the final mask is active
    # Using boolean masking, this flattens the result into a 1D array of pixels
    masked_saturation_pixels = s_channel[final_dominant_mask > 0]
    
    if len(masked_saturation_pixels) > 0:
        # Plot histogram
        ax6.hist(masked_saturation_pixels, bins=32, range=[0, 256], color='magenta', alpha=0.7, edgecolor='black')
        # Calculate mean for annotation
        mean_sat = np.mean(masked_saturation_pixels)
        ax6.axvline(mean_sat, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_sat:.1f}')
        ax6.legend()
    
    ax6.set_title("6. Saturation Histogram\n(Of Final Dominant Region Only)")
    ax6.set_xlabel("Saturation Value (0-255)")
    ax6.set_ylabel("Pixel Count")
    ax6.set_xlim([0, 256])
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == "__main__":
    # Run pipeline
    segmentation_pipeline_final("photo_2025-11-17_23-50-05.jpg")
    # Use your actual image file here:
    # segmentation_pipeline_final("Figure_1.png")
