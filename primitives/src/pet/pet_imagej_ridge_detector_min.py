import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import scyjava

# ================= CONFIGURATION =================
# 1. SETUP MEMORY & HEADLESS MODE
scyjava.config.add_option('-Xmx4g')
scyjava.config.add_option('-Djava.awt.headless=true')

import imagej

FIJI_PATH = r"G:\ProgramsMisc\Fiji"
INPUT_IMAGE = r"photo_2025-11-17_23-50-05.jpg"

# Ridge Parameters
LINE_WIDTH = 3.5
HIGH_CONTRAST = 200
LOW_CONTRAST = 80
# =================================================

def run_ridge_headless():
    # --- STEP 0: INIT ---
    print(f"Initializing ImageJ (Headless)...")
    ij = imagej.init(FIJI_PATH, mode='headless')
    
    # Import necessary classes
    IJ = scyjava.jimport('ij.IJ')
    WindowManager = scyjava.jimport('ij.WindowManager')
    ResultsTable = scyjava.jimport('ij.measure.ResultsTable')
    # Note: We do NOT import RoiManager anymore

    # --- STEP 1: LOAD IMAGE ---
    print(f"Loading image: {INPUT_IMAGE}")
    img_bgr = cv2.imread(INPUT_IMAGE)
    if img_bgr is None:
        print("Error reading image.")
        return

    # Extract L channel
    l_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)[:, :, 0]

    # --- STEP 2: PREPARE ENVIRONMENT ---
    # Convert to ImagePlus
    imp = ij.py.to_imageplus(l_cv)
    
    # Set as Active Image
    WindowManager.setTempCurrentImage(imp)

    # Clean ResultsTable if it exists
    rt = ResultsTable.getResultsTable()
    if rt: rt.reset()

    # --- STEP 3: RUN PLUGIN (Safe Mode) ---
    # CHANGES:
    # 1. add_to_manager=false (Prevents HeadlessException)
    # 2. We keep displayresults=true (ResultsTable is usually headless-safe)
    options = (
        f"line_width={LINE_WIDTH} "
        f"high_contrast={HIGH_CONTRAST} "
        f"low_contrast={LOW_CONTRAST} "
        "correct_position=true "
        "estimate_width=true "
        "add_to_manager=false "  # <--- CRITICAL FIX
        "displayresults=true " 
        "method_for_overlap_resolution=NONE "
        "darkline=true" 
    )

    print(f"Running Ridge Detection... Options: [{options}]")
    
    try:
        IJ.run("Ridge Detection", options)
    except Exception as e:
        print(f"Plugin Error: {e}")
        # Note: Even if it throws an error about 'showing' the table, 
        # the calculations often complete successfully.

    # --- STEP 4: RETRIEVE DATA FROM OVERLAY ---
    print("Extracting Data from Overlay...")

    # Get the Overlay (This replaces RoiManager)
    overlay = imp.getOverlay()
    
    if not overlay:
        print("No lines detected (Overlay is empty).")
        WindowManager.setTempCurrentImage(None)
        return

    # Convert Overlay to Array of ROIs
    rois = overlay.toArray()

    # --- STEP 5: SYNC WITH RESULTS TABLE ---
    rt = ResultsTable.getResultsTable()
    
    # Robust Width Retrieval
    widths = []
    if rt and rt.getCounter() > 0:
        try:
            # Try to get the width column
            width_col_idx = rt.getColumnIndex("Line Width")
            if width_col_idx != -1:
                widths = rt.getColumn(width_col_idx)
        except:
            print("Warning: Could not retrieve width stats.")
    
    # --- STEP 6: PROCESS RESULTS IN PYTHON ---
    results = []
    for i, roi in enumerate(rois):
        # Extract Geometry
        poly = roi.getFloatPolygon()
        points = [(round(poly.xpoints[j], 2), round(poly.ypoints[j]), 2) for j in range(poly.npoints)]
        
        # Extract Width (Safety check for index bounds)
        est_width = widths[i] if i < len(widths) else LINE_WIDTH

        results.append({
            "id": i,
            "points": points,
            "est_width": est_width
        })

    print(f"Successfully extracted {len(results)} segments.")
    if len(results) > 0:
        print(f"Sample Segment 0: Width={results[0]['est_width']:.2f}, Points={len(results[0]['points'])}")

    # --- CLEANUP ---
    WindowManager.setTempCurrentImage(None)
    imp.close()

    print(results)


if __name__ == "__main__":
    run_ridge_headless()
