import os
import sys
from pathlib import Path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

# ================= CONFIGURATION =================
import scyjava
# 1. SETUP MEMORY & HEADLESS MODE
scyjava.config.add_option('-Xmx4g')
scyjava.config.add_option('-Djava.awt.headless=true')

import imagej


# Path to your Fiji installation
FIJI_PATH = r"G:\ProgramsMisc\Fiji"

# Input Image Path
INPUT_IMAGE = r"photo_2025-11-17_23-50-05.jpg"

# NLC Parameters
BLOCK_RADIUS_X = 40
BLOCK_RADIUS_Y = 40
STD_DEVS = 5.0
STRETCH = True
CENTER = True
# =================================================


def imagej_init(fiji_path: str = None):
    """Initializes pyImageJ"""
    # --- STEP 0: INIT IMAGEJ ---
    print(f"Initializing ImageJ (Headless) from: {fiji_path}...")
    try:
        ij = imagej.init(fiji_path, mode='headless')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not start ImageJ.\n{e}")
        return None, None

    # Import IJ class
    IJ = scyjava.jimport('ij.IJ')

    return ij, IJ

          
def getLab(image_path: str = None):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None, None, None

    # --- OPENCV PRE-PROCESSING ---
    print(f"Loading image with OpenCV: {image_path}")
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        print("Error: OpenCV could not read the image.")
        return None, None, None

    # Convert to Lab and extract L (Luminance)
    # L channel is 0-255 (uint8)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_cv, a_cv, b_cv = cv2.split(img_lab)

    return l_cv, a_cv, b_cv


def imagej_lcn(l_cv, ij, IJ):
    # Store original shape for reshaping later
    h, w = l_cv.shape

    # --- IMAGEJ PROCESSING ---
    print("Sending L channel to ImageJ...")

    # Convert Numpy -> ImagePlus
    # We create an ImagePlus from the array.
    imp_l = ij.py.to_imageplus(l_cv)
    
    # Construct Options
    options = f"block_radius_x={BLOCK_RADIUS_X} block_radius_y={BLOCK_RADIUS_Y} standard_deviations={STD_DEVS}"
    if STRETCH: options += " stretch"
    if CENTER: options += " center"

    print(f"Running Normalize Local Contrast... Options: [{options}]")
    
    # Run the plugin on the ImagePlus
    # This modifies the internal ImageProcessor of imp_l
    IJ.run(imp_l, "Normalize Local Contrast", options)

    # --- STEP 3: ROBUST DATA RETRIEVAL ---
    print("Retrieving processed data from Java memory...")

    # 1. Get the ImageProcessor (The object that holds the modified pixels)
    ip = imp_l.getProcessor()
    
    # 2. Get the raw pixels (Returns a Java Signed Byte Array for 8-bit images)
    pixels_java = ip.getPixels()
    
    # 3. Copy Java bytes to a Numpy array (interpreted as int8: -128 to 127)
    # We use scyjava.to_python explicitly or just np.array if it exposes the buffer
    pixels_np_signed = np.array(pixels_java, dtype=np.int8)

    # 4. Reinterpret bits as uint8 (0 to 255) to match OpenCV format
    # This fixes the Java signed byte issue
    l_processed = pixels_np_signed.view(np.uint8)

    # 5. Reshape to original dimensions (Java flattens the array)
    l_processed = l_processed.reshape((h, w))

    # --- STEP 4: MERGE AND SAVE ---
    # Quick Check: Did it change?
    diff = np.mean(np.abs(l_processed.astype(float) - l_cv.astype(float)))
    print(f"Average Pixel Change: {diff:.4f} (If 0.0, plugin failed to run)")

    imp_l.close()

    return l_processed


def fiji_opencv_clahe(l_channel, block_size_px=127, max_slope=3.0):
    """
    Applies CLAHE using Fiji-style parameters (Pixel size, Slope)
    instead of OpenCV style (Grid count, ClipLimit).
    """
    h, w = l_channel.shape[:2]
    
    # Calculate Grid Size (rounding up to ensure coverage)
    grid_x = math.ceil(w / block_size_px)
    grid_y = math.ceil(h / block_size_px)
    
    # Create CLAHE
    # OpenCV's clipLimit is roughly equivalent to Fiji's Slope
    clahe = cv2.createCLAHE(clipLimit=max_slope, tileGridSize=(grid_y, grid_x))
    
    return clahe.apply(l_channel)


def fiji_auto_brightness_contrast(l_channel, saturation_percentage=0.35):
    """
    Replicates ImageJ's 'Auto' Brightness/Contrast button.
    
    Args:
        image_array: 2D numpy array (L channel).
        saturation_percentage: How many pixels to saturate at the ends (default 0.35%).
                               Increase this (e.g. to 1.0) for more aggressive contrast.
    """
    # 1. Calculate the low and high cutoffs
    # We ignore the bottom X% and top X% of pixels
    low_cutoff = np.percentile(l_channel, saturation_percentage)
    high_cutoff = np.percentile(l_channel, 100 - saturation_percentage)
    
    # 2. Prevent division by zero if image is flat
    if low_cutoff >= high_cutoff:
        return l_channel # Return original if contrast is impossible

    # 3. Stretch the histogram
    # np.interp maps values: [low, high] -> [0, 255]
    # Values outside [low, high] are automatically clamped (saturated) to 0 or 255
    normalized = np.interp(l_channel, [low_cutoff, high_cutoff], [0, 255])
    
    return normalized.astype(np.uint8)


def get_ridge_params_from_L(l_channel, detect_dark_lines=True):
    """
    Calculates Ridge Detection parameters directly from an L-channel numpy array.
    
    Args:
        l_channel: Numpy array (uint8 or float) representing Lightness.
        detect_dark_lines: True if detecting black lines on white background.
    
    Returns:
        line_width, high_contrast, low_contrast
    """
    # Convert to float for math
    img_float = l_channel.astype(float)

    # 1. Handle Polarity (Dark vs Bright)
    # If we want dark lines, we invert the image so they become mathematical "peaks"
    if detect_dark_lines:
        work_img = 255.0 - img_float
    else:
        work_img = img_float

    # 2. Smooth slightly to remove high-freq sensor noise
    img_smooth = gaussian(work_img, sigma=1.0)

    # 3. Create Binary Mask (Structure vs Background)
    # Otsu is excellent for L-channel separation
    try:
        thresh = threshold_otsu(img_smooth)
        binary_mask = img_smooth > thresh
    except ValueError:
        # Fallback if image is perfectly uniform
        return 3.0, 200, 80 

    # 4. Calculate LINE WIDTH
    # Distance from center of ridge to nearest background pixel
    dist_map = distance_transform_edt(binary_mask)
    skel = skeletonize(binary_mask)
    
    # Extract widths only at the skeleton locations
    widths = dist_map[skel] * 2.0  # radius * 2 = width
    
    if len(widths) > 0:
        est_width = np.median(widths)
    else:
        est_width = 3.0 # Default if no structure found

    # 5. Calculate CONTRAST
    foreground_vals = work_img[binary_mask]
    background_vals = work_img[~binary_mask]

    # Baseline background level
    bg_level = np.median(background_vals) if len(background_vals) > 0 else 0
    
    if len(foreground_vals) > 0:
        # High Contrast: The peaks of the ridges (90th percentile)
        high_c = np.percentile(foreground_vals, 90) - bg_level
        
        # Low Contrast: The faint parts of the ridges (20th percentile)
        low_c = np.percentile(foreground_vals, 20) - bg_level
    else:
        high_c, low_c = 200, 50

    # 6. Sanity Checks (Constraints for the Plugin)
    high_c = max(high_c, 10.0)
    low_c = max(low_c, 5.0)
    if low_c >= high_c:
        low_c = high_c * 0.5

    return round(est_width, 2), int(high_c), int(low_c)


def get_multiscale_ridge_params(l_channel, detect_dark_lines=True, variability_threshold=2.0, show_histogram=False):
    """
    Analyzes L-channel to determine if Single-Scale or Multi-Scale detection is needed.

    Prints stats, and optionally plots the width distribution.

    Args:
        l_channel: Numpy array (L from Lab).
        detect_dark_lines: Boolean.
        variability_threshold: If the std deviation of widths > this (in pixels), 
                               we trigger multi-scale.
    
    Returns:
        A LIST of parameter dictionaries. 
        Length 1 = Single Scale. Length 2 = Multi Scale.
    """
    # 1. Pre-processing
    img_float = l_channel.astype(float)
    if detect_dark_lines:
        work_img = 255.0 - img_float
    else:
        work_img = img_float

    img_smooth = gaussian(work_img, sigma=1.0)

    # 2. Binary Mask & Skeleton
    try:
        thresh = threshold_otsu(img_smooth)
        binary_mask = img_smooth > thresh
    except ValueError:
        print(">> STATISTICS: Image appears empty/uniform. Using Defaults.")
        return [{"line_width": 3.5, "high_contrast": 200, "low_contrast": 80, "darkline": detect_dark_lines, "name": "Default"}]

    # 3. Measure Widths
    dist_map = distance_transform_edt(binary_mask)
    skel = skeletonize(binary_mask)
    
    # Extract widths at skeleton (filter out tiny noise < 1px)
    raw_widths = dist_map[skel] * 2.0
    raw_widths = raw_widths[raw_widths > 0.5] 

    if len(raw_widths) == 0:
        print(">> STATISTICS: No structure detected. Using Defaults.")
        return [{"line_width": 3.5, "high_contrast": 200, "low_contrast": 80, "darkline": detect_dark_lines}]

    # 4. Statistical Analysis
    width_median = np.median(raw_widths)
    width_mean = np.mean(raw_widths)
    width_std = np.std(raw_widths)
    
    # Calculate Contrast
    foreground_vals = work_img[binary_mask]
    background_vals = work_img[~binary_mask]
    bg_level = np.median(background_vals) if len(background_vals) > 0 else 0
    p90 = np.percentile(foreground_vals, 90) if len(foreground_vals) > 0 else 200
    high_c = max(p90 - bg_level, 10)
    low_c = max(high_c * 0.4, 5)

    # --- PRINT STATISTICS (Always runs) ---
    print("-" * 40)
    print(f"IMAGE STATISTICS REPORT")
    print(f"Structure Detected: {detect_dark_lines and 'Dark Lines' or 'Bright Lines'}")
    print(f"Background Level:   {bg_level:.1f}")
    print(f"Target Contrast:    High={int(high_c)}, Low={int(low_c)}")
    print(f"Line Widths:")
    print(f"  - Median: {width_median:.2f} px")
    print(f"  - Mean:   {width_mean:.2f} px")
    print(f"  - StdDev: {width_std:.2f} px")
    print("-" * 40)
    
    # --- HISTOGRAM VISUALIZATION ---
    if show_histogram:
        # Calculate Percentiles
        p90 = np.percentile(raw_widths, 90)
        p95 = np.percentile(raw_widths, 95)
        p99 = np.percentile(raw_widths, 99)

        # Create explicit bins of size 0.5
        max_val = np.max(raw_widths)
        bins_list = np.arange(0, math.ceil(max_val) + 1, 0.5)
        
        plt.figure(figsize=(12, 6))
        
        # Plot Histogram
        plt.hist(raw_widths, bins=bins_list, color='skyblue', edgecolor='black', alpha=0.6, label='Width Counts')
        
        # Plot Key Metrics
        plt.axvline(width_median, color='black', linestyle='-', linewidth=2, label=f'Median ({width_median:.1f})')
        
        # Plot Upper Percentiles (The "Tail")
        plt.axvline(p90, color='orange', linestyle='--', linewidth=1.5, label=f'90% ({p90:.1f})')
        plt.axvline(p95, color='red', linestyle='--', linewidth=1.5, label=f'95% ({p95:.1f})')
        plt.axvline(p99, color='darkred', linestyle='--', linewidth=1.5, label=f'99% ({p99:.1f})')

        # Plot Decision Lines (if multi-scale is triggered)
        if width_std >= variability_threshold:
            p25 = np.percentile(raw_widths, 25)
            p85 = np.percentile(raw_widths, 85)
            # Use background shading or distinct markers for the "Action" items
            plt.axvline(p25, color='green', linestyle=':', linewidth=3, label=f'Thin Pass ({p25:.1f})')
            plt.axvline(p85, color='blue', linestyle=':', linewidth=3, label=f'Thick Pass ({p85:.1f})')
            
        plt.title(f'Line Width Distribution (StdDev: {width_std:.2f}px)', fontsize=14)
        plt.xlabel('Width (pixels)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 5. Decision Logic
    configs = []

    if width_std < variability_threshold:
        print(f">> DECISION: Uniform Widths (Std < {variability_threshold}). Running Single Pass.")
        configs.append({
            "line_width": round(width_median, 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c),
            "darkline": detect_dark_lines,
            "name": "Single_Pass"
        })
    else:
        print(f">> DECISION: Variable Widths (Std >= {variability_threshold}). Running Multi-Scale.")
        width_thin = np.percentile(raw_widths, 25)
        width_thick = np.percentile(raw_widths, 85)
        
        configs.append({
            "line_width": round(width_thin, 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c * 0.8), 
            "darkline": detect_dark_lines,
            "name": "Scale_Thin"
        })
        configs.append({
            "line_width": round(width_thick, 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c),
            "darkline": detect_dark_lines,
            "name": "Scale_Thick"
        })

    return configs


def main():
    ij, IJ = imagej_init(fiji_path=FIJI_PATH)
    if ij is None or IJ is None: return

    l_cv, a_cv, b_cv = getLab(image_path=INPUT_IMAGE)
    
    l_processed = imagej_lcn(l_cv, ij, IJ)

    l_clahe = fiji_opencv_clahe(l_processed, block_size_px=127, max_slope=3.0)
    l_autobrightness = fiji_auto_brightness_contrast(l_processed)
    l_processed = l_autobrightness

    print("Merging channels and saving...")
    merged_lab = cv2.merge([l_processed, a_cv, b_cv])
    result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)

    p = Path(INPUT_IMAGE)
    output_path = p.parent / f"{p.stem}_fixed_lcn{p.suffix}"
    
    cv2.imwrite(str(output_path), result_bgr)
    print(f"Done. Saved to: {output_path}")

    line_width, high_contrast, low_contrast = get_ridge_params_from_L(l_channel=l_processed, detect_dark_lines=True)
    print(
        f"Parameters for Ridge Detection:\n"
        f"line_width: {line_width}, high_contrast: {high_contrast}, low_contrast: {low_contrast}."
    )
    ridge_config = get_multiscale_ridge_params(l_channel=l_processed, detect_dark_lines=True, show_histogram=True)
    print(ridge_config)


if __name__ == "__main__":
    main()
