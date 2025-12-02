"""
pet_segmentation.py
-------------------
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def show_lab_a(image_path):
    # 1. Read Image
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Convert to Lab and Extract 'a'
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)
    a_channel = lab_img[:, :, 1]

    # 3. Save the 'a' layer to file
    # Split the input path to preserve extension and folder location
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}_laba{ext}"
    
    cv2.imwrite(output_path, a_channel)
    print(f"Saved 'a' channel to: {output_path}")

    # 4. Visualization (Image + Histogram)
    plt.figure(figsize=(10, 5))

    # --- Left: The 'a' Channel Image ---
    plt.subplot(1, 2, 1)
    plt.imshow(a_channel, cmap='gray')
    plt.title(f"Lab - 'a' Channel\n(Saved as {os.path.basename(output_path)})")
    plt.axis('off')

    # --- Right: The 'a' Channel Histogram ---
    plt.subplot(1, 2, 2)
    hist = cv2.calcHist([a_channel], [0], None, [256], [0, 256])
    
    plt.plot(hist, color='black')
    plt.fill_between(range(256), hist.flatten(), color='gray', alpha=0.3)
    
    plt.title("Histogram of 'a' Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    
    # Guides
    plt.axvline(x=128, color='k', linestyle='--', linewidth=1, label='Neutral')
    plt.text(10, np.max(hist)*0.8, '<- Green', color='green', fontweight='bold')
    plt.text(200, np.max(hist)*0.8, 'Red ->', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_lab_a("photo_2025-11-17_23-50-05.jpg")
