import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentation_grabcut_robust(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # --- STEP 1: Create the "Rough Hint" (Lab 'a') ---
    # We use the method you liked: Lab 'a' to find the general red areas.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    
    # Threshold to get the rough blob (Organ + Blood + Leaks)
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- STEP 2: Prepare the "Tri-Map" for GrabCut ---
    # GrabCut needs 3 regions:
    # 0: DEFINITELY Background (Black)
    # 2: PROBABLE Background (Light Gray) - The "Safety Zone"
    # 3: PROBABLE Foreground (White) - The Organ + Leaks
    # 1: DEFINITELY Foreground (Center) - The core of the organ
    
    # Start with all 0 (Definite Background)
    gc_mask = np.zeros(img.shape[:2], np.uint8)
    
    # Mark the "Rough Mask" area as "Probable Foreground" (3)
    gc_mask[rough_mask > 0] = cv2.GC_PR_FGD
    
    # Now, find the "Sure Foreground" (The core center) to teach the algorithm what the organ looks like
    # We erode the rough mask heavily to stay away from the edges
    kernel = np.ones((5,5), np.uint8)
    sure_fg = cv2.erode(rough_mask, kernel, iterations=6) # Erode deeply
    gc_mask[sure_fg > 0] = cv2.GC_FGD # Mark as "Definite Foreground" (1)

    # Define the "Probable Background" (The transition zone)
    # We dilate the rough mask a bit to catch any faint edges
    sure_bg_area = cv2.dilate(rough_mask, kernel, iterations=2)
    # Anything outside this dilation is "Definite Background" (0)
    # Inside the dilation but outside the mask is "Probable Background" (2)
    # (Already handled by initialization, but good to be explicit if needed)

    # --- STEP 3: Run GrabCut ---
    # This function iteratively learns the color distribution of FG vs BG
    # and snaps the border to the correct color change, ignoring soft gradients.
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Run 5 iterations
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # --- STEP 4: Finalize Mask ---
    # GrabCut modifies gc_mask in place. 
    # We want pixels that are (1) Definite FG or (3) Probable FG
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
    final_mask = final_mask * 255

    # --- STEP 5: Smooth the Result (Optional) ---
    # GrabCut can be slightly pixelated. A tiny close/open fixes it without changing shape.
    smooth_kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=2)
    
    # Get contours for visualization
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    # 1. The Input "Hint" (Tri-Map)
    plt.subplot(1, 3, 1)
    # Visualize the tri-map:
    # 0 (Black) = Bg, 1 (White) = Sure Fg, 3 (Gray) = Probable Fg
    vis_trimap = gc_mask.copy() * 80 # Scale for visibility
    plt.imshow(vis_trimap, cmap='gray')
    plt.title("GrabCut Input (Tri-Map)\nWhite=Core, Gray=Uncertain")
    plt.axis('off')

    # 2. The Result
    plt.subplot(1, 3, 2)
    plt.imshow(final_mask, cmap='gray')
    plt.title("GrabCut Output\n(Statistically Refined)")
    plt.axis('off')

    # 3. Overlay
    plt.subplot(1, 3, 3)
    vis_img = img.copy()
    
    # Green Overlay (Safe method)
    mask_indices = final_mask > 0
    if np.any(mask_indices):
        roi = vis_img[mask_indices]
        green = np.zeros_like(roi)
        green[:] = [0, 255, 0]
        vis_img[mask_indices] = cv2.addWeighted(roi, 0.7, green, 0.3, 0)
        
    cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2) # Yellow border
    
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Final Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use your file
    segmentation_grabcut_robust("photo_2025-11-17_23-50-05.jpg")
