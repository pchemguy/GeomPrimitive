"""
pet_segmentation.py
-------------------
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def create_smart_mask(image_path, target_color='red'):
    # 1. Read Image
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Convert to Lab and Extract 'a'
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)
    a_channel = lab_img[:, :, 1]

    # 3. Pre-processing (Robustness Step 1)
    # Blur to reduce high-frequency noise before thresholding
    blurred_a = cv2.GaussianBlur(a_channel, (5, 5), 0)

    # 4. Smart Thresholding (Otsu's Method)
    # Otsu calculates the optimal threshold value automatically.
    # Note: 'a' channel Red is > 128 (Bright), Green is < 128 (Dark).
    
    threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    
    # If targeting GREEN, we need to invert because Otsu looks for bright regions
    if target_color.lower() == 'green':
        threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    # ret is the calculated optimal threshold value
    ret, mask = cv2.threshold(blurred_a, 0, 255, threshold_type)
    
    print(f"Otsu's algorithm chose threshold: {ret}")

    # 5. Morphological Cleanup (Robustness Step 2)
    # Create a kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)

    # 'Opening' removes white noise (dots in the background)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 'Closing' fills black holes (spots inside the object)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. Save Mask
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}_lab_a_mask{ext}"
    cv2.imwrite(output_path, clean_mask)

    # 7. Visualization
    plt.figure(figsize=(12, 5))

    # Original 'a' channel
    plt.subplot(1, 3, 1)
    plt.imshow(a_channel, cmap='gray')
    plt.title("Raw 'a' Channel")
    plt.axis('off')

    # Raw Otsu Mask (Noisy)
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Raw Otsu Mask\n(Threshold: {ret:.1f})")
    plt.axis('off')

    # Morphological Clean Mask (Robust)
    plt.subplot(1, 3, 3)
    plt.imshow(clean_mask, cmap='gray')
    plt.title("Cleaned 'Smart' Mask\n(Blurred + Morphology)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_smart_mask("photo_2025-11-17_23-50-05.jpg", target_color='red')
