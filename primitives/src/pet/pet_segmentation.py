"""
pet_segmentation.py
-------------------
"""

import cv2
import matplotlib.pyplot as plt


def show_lab_a_channel(image_path):
    # 1. Read the image
    # OpenCV reads images in BGR format by default
    bgr_img = cv2.imread(image_path)

    # Safety check to ensure image was found
    if bgr_img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Convert to Lab Color Space
    # We use COLOR_BGR2Lab because the source was loaded via OpenCV
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)

    # 3. Extract the 'a' component
    # The lab_img is a NumPy array with shape (Height, Width, 3)
    # Channel 0 = L (Lightness)
    # Channel 1 = a (Green-Red)
    # Channel 2 = b (Blue-Yellow)
    a_channel = lab_img[:, :, 1]

    # --- Visualization Setup ---
    plt.figure(figsize=(12, 6))

    # Plot 1: Original Image (Convert BGR to RGB for correct matplotlib display)
    plt.subplot(1, 2, 1)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title("Original Image (RGB)")
    plt.axis('off')

    # Plot 2: The 'a' Channel
    plt.subplot(1, 2, 2)
    # We use a grayscale colormap. 
    # In 'a' channel: Darker pixels = Green, Lighter pixels = Red
    plt.imshow(a_channel, cmap='gray')
    plt.title("Lab - 'a' Channel (Green-Red)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04) # Adds a legend for pixel intensity

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_lab_a_channel("photo_2025-11-17_23-50-05.jpg")
