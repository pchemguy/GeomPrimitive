import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_organ_robust(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    # 2. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- STEP A: The "Redness" Filter (Hue) ---
    # We want to make sure we are looking at RED things (Organ + Blood)
    # and ignoring random background noise (Blue/Yellow specs).
    
    # Calculate distance from Red (Hue 0 or 180)
    # We treat 0 and 179 as the same point (Red)
    hue_float = h.astype(np.float32)
    hue_dist = np.minimum(np.abs(hue_float - 0), np.abs(hue_float - 180))
    
    # Create a weight: 1.0 for Red, 0.0 for Cyan
    # We are lenient (sigma=30) because the organ might be slightly orange
    hue_weight = np.exp(-0.5 * (hue_dist / 30.0)**2)

    # --- STEP B: The "Strength" Filter (Saturation) ---
    # This is the secret sauce. 
    # Normal Saturation is linear. 
    # We apply a POWER CURVE (Gamma Correction) to widen the gap.
    
    sat_float = s.astype(np.float32) / 255.0 # Normalize 0-1
    
    # Exponent > 1.0 suppresses weak signals (Blood) and keeps strong ones (Organ)
    # If Organ is Sat=0.9 and Blood is Sat=0.5:
    # Linear: Gap is 0.4
    # Cubed (s^3): 0.9^3 = 0.73, 0.5^3 = 0.125 -> Gap is now 0.6!
    sat_boosted = np.power(sat_float, 3) 

    # --- STEP C: Combine (Organness Score) ---
    # Score = Redness * Boosted_Saturation
    organ_score = hue_weight * sat_boosted
    
    # Convert back to 0-255 image
    organ_map = (organ_score * 255).astype(np.uint8)

    # --- STEP D: Thresholding ---
    # Now that we've widened the gap, Otsu will find the perfect cut-off
    # strictly between the Organ and everything else.
    ret, organ_mask = cv2.threshold(organ_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cleanup (Morphology)
    kernel = np.ones((5,5), np.uint8)
    # Close small holes inside the organ
    organ_mask = cv2.morphologyEx(organ_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove tiny specs (remaining blood droplets)
    organ_mask = cv2.morphologyEx(organ_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    # 1. Original Saturation (The Problem)
    plt.subplot(1, 3, 1)
    plt.imshow(s, cmap='gray')
    plt.title("Original Saturation\n(Blood & Organ look similar)")
    plt.axis('off')

    # 2. Boosted Score (The Solution)
    plt.subplot(1, 3, 2)
    plt.imshow(organ_map, cmap='gray')
    plt.title("Boosted 'Organ' Score\n(Gap Amplified: S^3)")
    plt.axis('off')

    # 3. Final Mask
    plt.subplot(1, 3, 3)
    # Overlay mask on original
    vis_img = img.copy()
    # Draw green contours around the detected organ
    contours, _ = cv2.findContours(organ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Final Detection (Green Outline)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    extract_organ_robust("photo_2025-11-17_23-50-05.jpg")
