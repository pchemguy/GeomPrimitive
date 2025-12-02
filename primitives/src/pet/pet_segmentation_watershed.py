import cv2
import numpy as np
import matplotlib.pyplot as plt

def separation_by_gradient_path(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # 2. Get the "Working Area" (Lab 'a')
    # This captures EVERYTHING red (Organ + Blood)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    
    # Threshold 'a' to get a binary mask of "All Red Stuff"
    # Invert binary because usually, foreground is white, background is black
    # Otsu works well here to separate "Red things" from "Gray background"
    ret, thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Noise Removal (Optional but recommended)
    # Remove tiny independent specks of blood that aren't touching the organ
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- STEP A: Define "Sure Background" ---
    # We dilate the area. Anything outside of this is DEFINITELY not organ.
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # --- STEP B: Define "Sure Foreground" (The Organ Core) ---
    # This is the critical step.
    # The user noted the organ is a "Single Solid Region".
    # Distance Transform assigns a value to every pixel based on how far it is from the edge.
    # The "thickest" part of the organ will have the highest values.
    # Thin attached blood splashes will have low values.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # We keep only the "peaks" of the mountains (the center of the organ).
    # 0.5 * max is a tunable safety margin. It keeps the core, drops the edges.
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # Convert to uint8 for marker usage
    sure_fg = np.uint8(sure_fg)

    # --- STEP C: Identify the "Unknown Region" (The Gradient Path) ---
    # This is the "Battlefield" between organ and background.
    # The highest gradient (contrast change) exists somewhere in this gray band.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # --- STEP D: Create Markers ---
    # Markers are labeled: 0 (unknown), 1 (background), >1 (foreground objects)
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # --- STEP E: Run Watershed ---
    # This function grows the markers (Organ Center) outward.
    # It stops EXACTLY when it hits the "watershed ridge" (Highest Gradient Path)
    # defined by the edges in the original image.
    markers = cv2.watershed(img, markers)

    # Create the final mask (Label > 1 is the Organ)
    # Label 1 was background, Label -1 is the boundary lines drawn by watershed
    organ_mask = np.zeros_like(a_channel)
    organ_mask[markers > 1] = 255

    # --- Visualization ---
    plt.figure(figsize=(12, 8))

    # 1. Lab 'a' input
    plt.subplot(2, 3, 1)
    plt.imshow(a_channel, cmap='gray')
    plt.title("Lab 'a' Channel\n(Organ + Blood combined)")
    plt.axis('off')

    # 2. Distance Transform (Finding the core)
    plt.subplot(2, 3, 2)
    plt.imshow(dist_transform, cmap='gray')
    plt.title("Distance Transform\n(Brighter = 'Thicker' area)")
    plt.axis('off')

    # 3. The "Unknown" Band
    plt.subplot(2, 3, 3)
    plt.imshow(unknown, cmap='gray')
    plt.title("Search Area\n(Where gradient path is calculated)")
    plt.axis('off')

    # 4. Watershed Markers (The logic)
    plt.subplot(2, 3, 4)
    # Visualize markers with a jet map to see different regions
    plt.imshow(markers, cmap='jet')
    plt.title("Watershed Result\n(Blue=Bg, Red=Organ, Outline=Cut)")
    plt.axis('off')

    # 5. Final Result Overlay
    plt.subplot(2, 3, 5)
    vis_img = img.copy()

    # Mark the boundary in Yellow
    vis_img[markers == -1] = [0, 255, 255] 

    # Create a mask of the organ pixels
    mask_indices = organ_mask > 0
    
    # If we found any organ pixels...
    if np.any(mask_indices):
        # 1. Grab the pixels from the image (Shape: N x 3)
        roi = vis_img[mask_indices]
        
        # 2. Create a solid green block of the EXACT same size (Shape: N x 3)
        green_overlay = np.zeros_like(roi)
        green_overlay[:] = [0, 255, 0]
        
        # 3. Blend them together. Now sizes match (N x 3 vs N x 3)
        blended = cv2.addWeighted(roi, 0.7, green_overlay, 0.3, 0)
        
        # 4. Put the blended pixels back into the image
        vis_img[mask_indices] = blended
            
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Final Cut along Gradient")
    plt.axis('off')
    
    # 6. Final Clean Mask
    plt.subplot(2, 3, 6)
    plt.imshow(organ_mask, cmap='gray')
    plt.title("Final Binary Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    separation_by_gradient_path("photo_2025-11-17_23-50-05.jpg")
