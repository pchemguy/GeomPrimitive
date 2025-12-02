import cv2
import numpy as np
import matplotlib.pyplot as plt

def imagej_find_edges_exact(img_gray):
    """
    Exact Python port of ImageJ's 'FindEdges.java'.
    
    Logic:
    1. Apply specific Sobel kernels (Top-Bottom, Left-Right).
    2. Calculate Magnitude = Sqrt(G_x^2 + G_y^2).
    3. CLIP results to type max (255), DO NOT Normalize/Stretch.
    """
    # 1. Define the Kernels exactly as seen in Java
    # sum1: Top positive, Bottom negative
    k_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float64)
    
    # sum2: Left positive, Right negative
    k_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float64)
    
    # 2. Convolve (Use float64 to prevent overflow)
    # cv2.filter2D correlates, which is fine for symmetrical kernels like these
    # but strictly speaking we want the math to match sum1/sum2
    sum1 = cv2.filter2D(img_gray.astype(np.float64), -1, k_y)
    sum2 = cv2.filter2D(img_gray.astype(np.float64), -1, k_x)
    
    # 3. Magnitude
    magnitude = np.sqrt(sum1**2 + sum2**2)
    
    # 4. Clip (The Java Logic)
    # "if (value > typeMaxValue) value = typeMaxValue"
    magnitude = np.clip(magnitude, 0, 255)
    
    return magnitude.astype(np.uint8)

def imagej_equalize_exact(img_gray, mask=None):
    """
    Exact Python port of ImageJ's 'ContrastEnhancer.java'.
    """
    hist = cv2.calcHist([img_gray], [0], mask, [256], [0, 256]).flatten()
    weighted_hist = np.sqrt(hist)
    
    # Double Sum CDF
    total_sum = (2 * np.sum(weighted_hist)) - weighted_hist[0] - weighted_hist[-1]
    if total_sum == 0: return img_gray
    scale = 255.0 / total_sum
    
    cumsum = np.cumsum(weighted_hist)
    double_sum_at_i = (2 * cumsum) - weighted_hist[0] - weighted_hist
    lut = np.round(double_sum_at_i * scale).astype(np.uint8)
    lut[0] = 0; lut[255] = 255
    
    equalized_full = cv2.LUT(img_gray, lut)
    
    if mask is not None:
        result = img_gray.copy()
        result[mask > 0] = equalized_full[mask > 0]
        return result
    else:
        return equalized_full

def separation_absolute_exact(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    # --- STEP 1: Lab-a + GrabCut (Get ROI) ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1]
    ret, rough_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gc_mask = np.zeros(img.shape[:2], np.uint8)
    gc_mask[rough_mask > 0] = cv2.GC_PR_FGD
    kernel = np.ones((5,5), np.uint8)
    sure_fg = cv2.erode(rough_mask, kernel, iterations=5)
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    roi_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')

    # --- STEP 2: IMAGEJ EQUALIZATION (Exact Port) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_roi = imagej_equalize_exact(gray, mask=roi_mask)

    # --- STEP 3: IMAGEJ FIND EDGES (Exact Port) ---
    # We apply the custom Sobel kernels and Clipping
    edges = imagej_find_edges_exact(eq_roi)

    # --- STEP 4: FILTER & CUT ---
    # Remove outer border
    inner_zone = cv2.erode(roi_mask, kernel, iterations=3)
    internal_edges = cv2.bitwise_and(edges, edges, mask=inner_zone)
    
    # Threshold the edges to find the cut
    # Since ImageJ CLIPS instead of normalizes, weak edges might be much lower 
    # than in previous scripts. We might need a lower threshold or Otsu.
    ret_e, cut_line = cv2.threshold(internal_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(eq_roi, cmap='gray')
    plt.title("Step 2: ImageJ Equalized")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(internal_edges, cmap='gray')
    plt.title("Step 3: ImageJ Find Edges\n(Sobel + Clip)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    vis_final = img.copy()
    contours_gc, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_final, contours_gc, -1, (0, 255, 0), 2)
    
    vis_line = cv2.dilate(cut_line, np.ones((2,2), np.uint8))
    vis_final[vis_line > 0] = [0, 255, 255]

    plt.imshow(cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB))
    plt.title("Final Separation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    separation_absolute_exact("photo_2025-11-17_23-50-05.jpg")
