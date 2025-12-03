import cv2
import numpy as np
import os
import sys

# --- UTILITY: Enhanced Debug Printing ---
def print_stat(label, value):
    print(f"    |-- {label:<25}: {value}")

def print_header(name):
    print(f"\n{'='*60}")
    print(f" PIPELINE BRANCH: {name}")
    print(f"{'='*60}")

# --- UTILITY: Smart Thresholding ---
def calculate_dynamic_threshold(channel_data, percentile=95, min_limit=200):
    """
    Calculates threshold and prints distribution statistics.
    """
    flat = channel_data.flatten()
    
    # Calculate basic stats
    img_min = np.min(flat)
    img_max = np.max(flat)
    img_mean = np.mean(flat)
    
    print_stat("Channel Stats", f"Min={img_min}, Max={img_max}, Mean={img_mean:.2f}")
    
    # Calculate Percentile
    calc_thresh = np.percentile(flat, percentile)
    print_stat(f"Percentile ({percentile}%)", f"{calc_thresh:.2f}")
    
    # Apply Safety Clamp
    final_thresh = int(max(calc_thresh, min_limit))
    
    if final_thresh > calc_thresh:
        print_stat("Decision", f"Clamped to Safety Limit ({min_limit})")
    else:
        print_stat("Decision", f"Using Calculated Percentile ({int(calc_thresh)})")
        
    return final_thresh

# ==========================================
# BRANCH 1: Statistical (Intensity Only)
# ==========================================
def remove_glare_statistical(image_path):
    print_header("STATISTICAL (Intensity Only)")
    
    # 1. Load
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    total_pixels = h * w
    print_stat("Input Image", f"{image_path} ({w}x{h} px)")

    # 2. Analyze Intensity
    print("  [Step 1] Analyzing Grayscale Intensity...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val = calculate_dynamic_threshold(gray)

    # 3. Create Mask
    print("  [Step 2] Generating Glare Mask...")
    # Logic: Pixel is glare if Brightness > Threshold
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Debug: Count pixels
    glare_pixels = cv2.countNonZero(mask)
    glare_pct = (glare_pixels / total_pixels) * 100
    print_stat("Glare Pixels Found", f"{glare_pixels} ({glare_pct:.2f}%)")

    # 4. Refine Mask (Dilation)
    print("  [Step 3] Dilating Mask to cover glare halo...")
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 5. Inpaint
    print("  [Step 4] Inpainting (Telea Method)...")
    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    # 6. Save
    file_root, file_ext = os.path.splitext(image_path)
    output_path = f"{file_root}_stat_clean{file_ext}"
    cv2.imwrite(output_path, result)
    print_stat("OUTPUT SAVED", output_path)

# ==========================================
# BRANCH 2: Chromatic (Intensity + Saturation)
# ==========================================
def remove_glare_chromatic(image_path):
    print_header("CHROMATIC (Intensity + Saturation)")
    
    # 1. Load
    if not os.path.exists(image_path): return
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    total_pixels = h * w
    print_stat("Input Image", f"{image_path} ({w}x{h} px)")

    # 2. Convert to HSV
    print("  [Step 1] Converting to HSV and splitting channels...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv) # We need Saturation (s) and Value (v)

    # 3. Analyze Intensity (Value Channel)
    print("  [Step 2] Analyzing Intensity (V-Channel)...")
    thresh_val = calculate_dynamic_threshold(v)

    # 4. Analyze Saturation
    # We define a "Low Saturation" cap. Glare is White (Sat ~ 0).
    # Liver is Red (Sat > 50). We want to avoid killing bright red pixels.
    sat_cap = total_pixels / 200
    print("  [Step 3] Applying Color Logic...")
    print_stat("Saturation Cutoff", f"< {sat_cap} (Pixels must be whiter than this)")
    
    # 5. Create Logic Mask
    # Logic: Pixel is glare if (Value > Threshold) AND (Saturation < Cap)
    print("  [Step 4] Computing Bitwise Logic Mask...")
    glare_mask = (v > thresh_val) & (s < sat_cap)
    mask_uint8 = glare_mask.astype(np.uint8) * 255
    
    # Debug: Count pixels
    glare_pixels = cv2.countNonZero(mask_uint8)
    glare_pct = (glare_pixels / total_pixels) * 100
    print_stat("Glare Pixels Found", f"{glare_pixels} ({glare_pct:.2f}%)")
    
    if glare_pixels == 0:
        print("    [WARNING] No glare found. Threshold might be too high or Saturation cap too low.")

    # 6. Refine Mask
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)

    # 7. Inpaint
    print("  [Step 5] Inpainting...")
    result = cv2.inpaint(img, mask_dilated, 5, cv2.INPAINT_TELEA)

    # 8. Save
    file_root, file_ext = os.path.splitext(image_path)
    output_path = f"{file_root}_chroma_clean{file_ext}"
    cv2.imwrite(output_path, result)
    print_stat("OUTPUT SAVED", output_path)

# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
    # Get image from command line or use default
    target_image = "photo_2025-11-17_23-50-05.jpg"
    if len(sys.argv) > 1:
        target_image = sys.argv[1]

    print(f"Starting Processing Pipeline for: {target_image}")

    # --- Run Branch A ---
    remove_glare_statistical(target_image)

    # --- Run Branch B ---
    remove_glare_chromatic(target_image)
    
    print("\nProcessing Complete.")
