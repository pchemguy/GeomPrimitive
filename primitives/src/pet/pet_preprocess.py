"""
pet_preprocess.py
-----------------
"""

import cv2
import numpy as np


def clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to a color image by processing only the Luminance channel.
    Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        image_bgr: Input image in BGR format (standard OpenCV format).
        clip_limit: Threshold for contrast limiting. Higher = more contrast + more noise.
        tile_grid_size: Size of grid for histogram equalization (Input image is divided into these tiles).
    """
    # 1. Convert BGR to LAB color space
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # 2. Split into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 3. Apply CLAHE to the L-channel (Lightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)

    # 4. Merge the CLAHE enhanced L-channel with the original A and B channels
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))

    # 5. Convert back to BGR color space
    output_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return output_bgr


def normalize_local_contrast(image_bgr, block_radius_x, block_radius_y, stddev_target=1.0, stretch=True, center=True, saturate_outliers=True):
    """
    Applies Normalize Local Contrast to a BGR image by processing the Luminance channel.
    
    Args:
        image_bgr: Input image in BGR format (Opencv default).
        block_radius_x: Radius in X (kernel width will be 2*x + 1).
        block_radius_y: Radius in Y (kernel height will be 2*y + 1).
        stddev_target: The factor to scale the deviation (similar to the 'stddev' param).
        stretch: If True, stretches min/max to 0-255.
        center: If True, centers the result on 127.5.
        saturate_outliers: If True, ignores the top/bottom 0.5% of pixels when stretching.
                           This prevents noise from making the image look dark.
    """
# 1. Convert BGR to LAB (Processing L-channel only)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    img_float = l_channel.astype(np.float32)

    # 2. Kernel setup
    ksize = (2 * block_radius_x + 1, 2 * block_radius_y + 1)

    # 3. Calculate Mean and Sigma
    local_mean = cv2.boxFilter(img_float, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    
    local_mean_sq = cv2.boxFilter(img_float**2, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    local_var = local_mean_sq - (local_mean ** 2)
    local_var[local_var < 0] = 0
    local_sigma = np.sqrt(local_var) + 1e-5

    # 4. Normalization
    pixel_normalized = (img_float - local_mean) / local_sigma
    
    # Gain
    output_float = pixel_normalized * (127.5 / stddev_target)

    if center:
        output_float += 127.5

    # 5. ROBUST STRETCH
    if stretch:
        if saturate_outliers:
            # Calculate the 0.5th and 99.5th percentile
            # This ignores the extreme outliers that make images dark
            low_p, high_p = np.percentile(output_float, (0.5, 99.5))
            
            # Clip the values to this range
            output_float = np.clip(output_float, low_p, high_p)
            
            # Stretch the remaining range to 0-255
            if (high_p - low_p) > 0:
                output_float = (output_float - low_p) / (high_p - low_p) * 255.0
        else:
            # Standard linear stretch (often too dark if noise exists)
            v_min, v_max = output_float.min(), output_float.max()
            if (v_max - v_min) > 0:
                output_float = (output_float - v_min) / (v_max - v_min) * 255.0

    output_l = np.clip(output_float, 0, 255).astype(np.uint8)

    # 6. Merge and Return
    merged_lab = cv2.merge((output_l, a_channel, b_channel))

    # 7. Convert back to BGR
    output_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return output_bgr


def main():
    import os
    import sys
    # 1. Define Input Path
    # Using the filename you provided earlier
    image_path = "photo_2025-11-17_23-50-05.jpg"
    image_path_lcn = image_path[:-4] + "_lcn.jpg"
    image_path_clahe = image_path[:-4] + "_clahe.jpg"

    #"photo_2025-11-17_23-50-05_Normalize_Local_Contrast_40x40x5.00.jpg"
    
    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit()

    print(f"Processing {image_path}...")
    source_image = cv2.imread(image_path)
    
    if source_image is None:
        print("Error: Failed to decode image.")
        sys.exit()

    # Apply filter
    # Radius 20, Target StdDev 3.0 (Typical ImageJ defaults)
    out_lcn = normalize_local_contrast(source_image, block_radius_x=20, block_radius_y=20, stddev_target=4.0)
    cv2.imwrite(image_path_lcn, out_lcn)

    out_clahe = clahe(image_bgr=source_image, clip_limit=10.0, tile_grid_size=(8, 8))
    cv2.imwrite(image_path_clahe, out_clahe)


if __name__ == "__main__":
    main()
