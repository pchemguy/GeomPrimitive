import os
import sys
import cv2
import numpy as np
from pathlib import Path

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

def run_nlc():
    # --- STEP 0: INIT IMAGEJ ---
    print(f"Initializing ImageJ (Headless) from: {FIJI_PATH}...")
    try:
        ij = imagej.init(FIJI_PATH, mode='headless')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not start ImageJ.\n{e}")
        return

    # Import IJ class
    IJ = scyjava.jimport('ij.IJ')

    if not os.path.exists(INPUT_IMAGE):
        print(f"File not found: {INPUT_IMAGE}")
        return

    # --- STEP 1: OPENCV PRE-PROCESSING ---
    print(f"Loading image with OpenCV: {INPUT_IMAGE}")
    img_bgr = cv2.imread(INPUT_IMAGE)
    
    if img_bgr is None:
        print("Error: OpenCV could not read the image.")
        return

    # Convert to Lab and extract L (Luminance)
    # L channel is 0-255 (uint8)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_cv, a_cv, b_cv = cv2.split(img_lab)
    
    # Store original shape for reshaping later
    h, w = l_cv.shape

    # --- STEP 2: IMAGEJ PROCESSING ---
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

    print("Merging channels and saving...")
    merged_lab = cv2.merge([l_processed, a_cv, b_cv])
    result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)

    p = Path(INPUT_IMAGE)
    output_path = p.parent / f"{p.stem}_fixed_lcn{p.suffix}"
    
    cv2.imwrite(str(output_path), result_bgr)
    
    imp_l.close()
    print(f"Done. Saved to: {output_path}")

if __name__ == "__main__":
    run_nlc()
