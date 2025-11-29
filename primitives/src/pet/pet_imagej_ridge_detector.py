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


def imagej_init(fiji_path: str = None, mode="headless"):
    """Initializes pyImageJ"""
    # --- STEP 0: INIT IMAGEJ ---
    print(f"Initializing ImageJ (Headless) from: {fiji_path}...")
    try:
        ij = imagej.init(fiji_path, mode)
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


def get_multiscale_ridge_params(l_channel, detect_dark_lines=True, spread_threshold=0.5, show_histogram=False):
    """
    Decides between Single vs Multi-Scale based on P90 and Relative Spread.
    
    Args:
        spread_threshold (0.5): The ratio (P90-P10)/Median. 
                                If > 0.5, the width varies by more than 50% of the median size -> MultiScale.
    """
    # 1. Pre-processing
    img_float = l_channel.astype(float)
    work_img = (255.0 - img_float) if detect_dark_lines else img_float
    img_smooth = gaussian(work_img, sigma=1.0)

    # 2. Binary Mask & Skeleton
    try:
        thresh = threshold_otsu(img_smooth)
        binary_mask = img_smooth > thresh
    except ValueError:
        return [{"line_width": 3.5, "high_contrast": 200, "low_contrast": 80, "darkline": detect_dark_lines, "name": "Default"}]

    # 3. Measure Widths
    dist_map = distance_transform_edt(binary_mask)
    skel = skeletonize(binary_mask)
    
    # Filter valid widths (> 0.5px)
    raw_widths = dist_map[skel] * 2.0
    raw_widths = raw_widths[raw_widths > 0.5]

    if len(raw_widths) == 0:
        return [{"line_width": 3.5, "high_contrast": 200, "low_contrast": 80, "darkline": detect_dark_lines}]

    # 4. ROBUST STATISTICS
    width_median = np.median(raw_widths)
    width_mean = np.mean(raw_widths)
    width_std = np.std(raw_widths)
    median = np.median(raw_widths)
    p10 = np.percentile(raw_widths, 10)
    p25 = np.percentile(raw_widths, 25)
    p90 = np.percentile(raw_widths, 90)
    
    # "Adjusted Variance" (Relative Inter-Percentile Spread)
    # How wide is the valid data range compared to the object size?
    spread_abs = p90 - p10
    spread_rel = spread_abs / width_median if width_median > 0 else 0

    # Calculate Contrast
    foreground_vals = work_img[binary_mask]
    background_vals = work_img[~binary_mask]
    bg_level = np.median(background_vals) if len(background_vals) > 0 else 0
    p90_contrast = np.percentile(foreground_vals, 90) if len(foreground_vals) > 0 else 200
    high_c = max(p90_contrast - bg_level, 10)
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

    # 5. DECISION LOGIC
    configs = []
    is_multiscale = spread_rel > spread_threshold

    if is_multiscale:
        decision_str = f"Multi-Scale (Spread Ratio {spread_rel:.2f} > {spread_threshold})"
        # Run 1: Thin (P25)
        configs.append({
            "line_width": round(float(p25), 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c * 0.8),
            "darkline": detect_dark_lines,
            "name": "Scale_Thin"
        })
        # Run 2: Thick (Targeting the P90 tail)
        configs.append({
            "line_width": round(float(p90), 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c),
            "darkline": detect_dark_lines,
            "name": "Scale_Thick_P90"
        })
    else:
        decision_str = f"Single-Scale (Spread Ratio {spread_rel:.2f} <= {spread_threshold})"
        configs.append({
            "line_width": round(float(width_median), 2),
            "high_contrast": int(high_c),
            "low_contrast": int(low_c),
            "darkline": detect_dark_lines,
            "name": "Single_Pass"
        })

    print(f">> STATS: Median={width_median:.2f}, P10={p10:.2f}, P90={p90:.2f}")
    print(f">> DECISION: {decision_str}")

    # 6. VISUALIZATION
    if show_histogram:
        max_val = np.max(raw_widths)
        bins_list = np.arange(0, math.ceil(max_val) + 1, 1)
        
        plt.figure(figsize=(12, 5))
        plt.hist(raw_widths, bins=bins_list, color='gray', alpha=0.4, label='All Widths')
        
        # Highlight the "Core" range (P10 to P90)
        plt.axvspan(p10, p90, color='yellow', alpha=0.2, label='Robust Range (P10-P90)')
        plt.axvline(width_median, color='black', linewidth=2, label=f'Median ({width_median:.1f})')
        plt.axvline(p90, color='red', linestyle='--', linewidth=2, label=f'P90 ({p90:.1f})')

        if is_multiscale:
            plt.scatter([p25], [0], color='green', s=100, zorder=10, label='Run 1: Thin')
            plt.scatter([p90], [0], color='blue', s=100, zorder=10, label='Run 2: Thick')
        else:
            plt.scatter([width_median], [0], color='green', s=100, zorder=10, label='Run: Single')

        plt.title(f'Width Distribution - Decision: {decision_str}')
        plt.legend()
        plt.show()

    return configs


def fiji_ridge_detector(ij,
                        l_channel, 
                        line_width, 
                        high_contrast, 
                        low_contrast, 
                        darkline=True, 
                        correct_position=True, 
                        estimate_width=True, 
                        add_to_manager=True, 
                        overlap_resolution="SLOPE"):
    """
    Runs Ridge Detection and returns a structured list of segments with their metadata.
    
    Args:
        ij: The initialized ImageJ/Fiji gateway.
        l_channel (numpy.ndarray): The 2D L-channel array.
        line_width, high_contrast, low_contrast: Detection parameters.
        darkline, correct_position, estimate_width, add_to_manager: Boolean flags.
        overlap_resolution: "NONE" or "SLOPE".
        cleanup (bool): If True, closes the temp image after processing.

    Returns:
        List[dict]: A list of segment dictionaries in the format:
        [
            {
                "nodes": [(x1, y1), (x2, y2), ...], 
                "meta":  {"Line Width": 3.5, "Length": 10.2, "Mean": 150.0, ...}
            },
            ...
        ]
    """
    # --- 1. JAVA IMPORTS & SETUP ---
    # We import these dynamically to ensure the gateway 'ij' is active
    RoiManager = imagej.sj.jimport('ij.plugin.frame.RoiManager')
    ResultsTable = imagej.sj.jimport('ij.measure.ResultsTable')
    WindowManager = imagej.sj.jimport('ij.WindowManager')

    # Get Instances
    rm = RoiManager.getRoiManager()
    rt = ResultsTable.getResultsTable()

    # CRITICAL: Reset ROI Manager and Results Table before running.
    # If we don't, we mix data from previous runs.
    if rm: rm.reset()
    if rt: rt.reset()

    # --- 2. PREPARE IMAGE ---
    # Convert Numpy to ImageJ (ImagePlus)
    ij_image = ij.py.to_java(l_channel)
    
    # Set this as the "Active" image without opening a window
    WindowManager.setTempCurrentImage(ij_image)

    # --- 3. CONSTRUCT PARAMETERS ---
    valid_methods = ["NONE", "SLOPE"]
    if overlap_resolution not in valid_methods:
        raise ValueError(f"overlap_resolution must be one of {valid_methods}")

    # Note: 'estimate_width' must be True to get width data in "meta"
    parameters = {
        "line_width": float(line_width),
        "high_contrast": int(high_contrast),
        "low_contrast": int(low_contrast),
        "correct_position": correct_position,
        "estimate_width": estimate_width,
        "add_to_manager": add_to_manager,
        "method_for_overlap_resolution": overlap_resolution,
        "darkline": darkline,
        "extend_line": True,
        "displayresults": True, # Required to populate 'rt'
        "show_junction_points": False,
        "show_ids": False,
        "verbose_mode": False,
        "make_binary": False
    }

    # Format for ImageJ Macro string
    param_list = []
    for k, v in parameters.items():
        if isinstance(v, bool):
            param_list.append(f"{k}={'true' if v else 'false'}")
        else:
            param_list.append(f"{k}={v}")
    param_str = " ".join(param_list)

    # --- 4. RUN PLUGIN ---
    try:
        ij.py.run_plugin("Ridge Detection", param_str)
    except Exception as e:
        print(f"Error running Ridge Detection: {e}")
        return []

    # --- 5. EXTRACT & SYNC DATA ---
    output_segments = []
    
    # Get updated references
    rois = rm.getRoisAsArray()
    rt = ResultsTable.getResultsTable()
    
    if rois is None:
        if cleanup: ij_image.close()
        return []

    # Check for consistency
    count_rois = len(rois)
    count_rt = rt.getCounter() if rt else 0
    
    if count_rois != count_rt:
        print(f"Warning: ROI Manager ({count_rois}) and Results Table ({count_rt}) sync mismatch.")

    # Get available column headers from Results Table (e.g. "Length", "Line Width", "Mean")
    # rt.getHeadings() returns a Java array of Strings
    headings = list(rt.getHeadings()) if rt else []

    for i, roi in enumerate(rois):
        # A. Extract Geometry ("nodes")
        # Use getFloatPolygon for sub-pixel precision
        poly = roi.getFloatPolygon()
        
        # poly.xpoints is a Java buffer that may be larger than npoints.
        # We strictly slice by npoints.
        nodes = [(poly.xpoints[k], poly.ypoints[k]) for k in range(poly.npoints)]

        # B. Extract Statistics ("meta")
        meta = {}
        if i < count_rt:
            for col_name in headings:
                # getValue(column, row)
                try:
                    val = rt.getValue(col_name, i)
                    meta[col_name] = val
                except:
                    pass
        
        # C. Build Structure
        output_segments.append({
            "nodes": nodes,
            "meta": meta
        })

    # --- 6. CLEANUP ---
    ij_image.close()
    # Optional: Hide the ROI Manager and Results Table again if they popped up
    # rm.close() 
    # rt.close()

    return output_segments


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
    ridge_config = get_multiscale_ridge_params(l_channel=l_processed, detect_dark_lines=True, spread_threshold=3, show_histogram=True)
    print(ridge_config)

    line_width = ridge_config[0]["line_width"]
    high_contrast = ridge_config[0]["high_contrast"]
    low_contrast = ridge_config[0]["low_contrast"]
    darkline = ridge_config[0]["darkline"]

    results = fiji_ridge_detector(ij, l_processed, line_width, high_contrast, low_contrast, darkline)



if __name__ == "__main__":
    main()
