import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_grabcut_with_blood_veto(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # --- STEP 1: The "Rough" Shape (Lab 'a') ---
    # This captures everything reddish (Organ + Blood)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- STEP 2: The "Blood Veto" (Saturation) ---
    # We identify areas that are Red but "Weak" (Low Saturation)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # We only look at saturation INSIDE the rough mask to find the split
    # masking=rough_mask ensures we don't calculate Otsu on the gray background
    valid_sat = s[rough_mask > 0]
    
    if len(valid_sat) > 0:
        # Calculate Otsu threshold on Saturation to separate Strong Red (Organ) from Weak Red (Blood)
        # We use a slight modifier (0.9) to be conservative-we really want to target the weak stuff.
        sat_thresh, _ = cv2.threshold(valid_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Auto-calculated Saturation Threshold: {sat_thresh}")
        
        # Create the Veto Mask: Pixels inside the Rough area but with LOW saturation
        # Logic: If (In Rough Mask) AND (Saturation < Threshold) -> It is Blood
        blood_veto_mask = cv2.bitwise_and(rough_mask, cv2.bitwise_not(cv2.inRange(s, sat_thresh, 255)))
    else:
        blood_veto_mask = np.zeros_like(rough_mask)

    # --- STEP 3: Build the GrabCut "Tri-Map" ---
    # 0 = Definite BG, 1 = Definite FG, 2 = Probable BG, 3 = Probable FG
    
    # A. Start with 0 (Definite Background)
    gc_mask = np.zeros(img.shape[:2], np.uint8)
    
    # B. Set Rough Mask to 3 (Probable Foreground)
    gc_mask[rough_mask > 0] = cv2.GC_PR_FGD
    
    # C. APPLY THE VETO: Force Blood areas back to 0 (Definite Background)
    # This handcuffs GrabCut. It CANNOT select these pixels.
    gc_mask[blood_veto_mask > 0] = cv2.GC_BGD
    
    # D. Define "Sure Foreground" (1) - The Core
    # We erode the rough mask (excluding the vetoed blood)
    clean_rough = cv2.subtract(rough_mask, blood_veto_mask)
    kernel = np.ones((5,5), np.uint8)
    sure_fg = cv2.erode(clean_rough, kernel, iterations=6)
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    # --- STEP 4: Run GrabCut ---
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # --- STEP 5: Finalize ---
    # Keep Definite FG (1) and Probable FG (3)
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8') * 255
    
    # Smooth slightly
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Visualization ---
    plt.figure(figsize=(14, 6))

    # 1. The Veto Logic
    plt.subplot(1, 3, 1)
    # Show Saturation channel with the Veto mask overlay
    vis_sat = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    # Color the Vetoed Blood areas RED
    vis_sat[blood_veto_mask > 0] = [0, 0, 255] # Red overlay
    plt.imshow(vis_sat)
    plt.title(f"Saturation Channel\nRed = Vetoed (S < {sat_thresh:.0f})")
    plt.axis('off')

    # 2. The Handcuffed GrabCut Input
    plt.subplot(1, 3, 2)
    # Visualize the constraints: Black=Bg, Gray=Probable, White=Definite
    vis_gc = gc_mask.copy() * 80
    plt.imshow(vis_gc, cmap='gray')
    plt.title("GrabCut Constraints\n(Notice the cutout blood!)")
    plt.axis('off')

    # 3. Final Result
    plt.subplot(1, 3, 3)
    vis_final = img.copy()
    
    # Overlay Green
    mask_indices = final_mask > 0
    if np.any(mask_indices):
        roi = vis_final[mask_indices]
        green = np.zeros_like(roi)
        green[:] = [0, 255, 0]
        vis_final[mask_indices] = cv2.addWeighted(roi, 0.7, green, 0.3, 0)
        
    cv2.drawContours(vis_final, contours, -1, (0, 255, 255), 2)
    plt.imshow(cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB))
    plt.title("Final Result\n(Smoothed Border + Blood Removed)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use your file
    segmentation_grabcut_with_blood_veto("photo_2025-11-17_23-50-05.jpg")
