import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import scyjava

# ================= CONFIGURATION =================
scyjava.config.add_option('-Xmx4g')

import imagej


FIJI_PATH = r"G:\ProgramsMisc\Fiji"
INPUT_IMAGE = r"photo_2025-11-17_23-50-05.jpg"

# Ridge Parameters
LINE_WIDTH = 3.5
HIGH_CONTRAST = 200
LOW_CONTRAST = 80
# =================================================

def use_fiji_jvm(fiji_dir):
    """
    Finds the JVM bundled within Fiji and forces scyjava to use it.
    Must be called BEFORE the JVM is initialized.
    """
    if scyjava.jvm_started():
        print("CRITICAL WARNING: JVM is already running! Cannot switch versions now.")
        return

    fiji_path = Path(fiji_dir)
    java_dir = Path(os.path.join(fiji_dir, "java"))

    # 1. Hunt for the bundled JRE/JDK
    # Fiji (Windows) usually has structure: Fiji/java/win64/jdk1.8.0_172/jre...
    target_java_home = None
    
    if java_dir.exists():
        # Look recursively for a 'bin' folder containing 'server/jvm.dll' (Windows)
        # or 'lib/server/libjvm.so' (Linux)
        for root, dirs, files in os.walk(java_dir):
            if "jvm.dll" in files:
                # Found the DLL. Now we need the 'Home' (usually 2 levels up from bin/server)
                # Standard layout: HOME/bin/server/jvm.dll
                # We want HOME.
                potential_bin = Path(root).parent
                if potential_bin.name == "bin":
                    target_java_home = potential_bin.parent
                    if target_java_home.name == "jre":
                        target_java_home = target_java_home.parent
                    break
                # Handle simplified JRE layouts
                elif Path(root).name == "server": 
                     target_java_home = Path(root).parent.parent
                     break
    
    # 2. Apply the Switch
    if target_java_home:
        print(f"--> Found Fiji Bundled Java: {target_java_home}")
        # Force the environment variable for this process ONLY
        os.environ["JAVA_HOME"] = str(target_java_home)
    else:
        print(f"--> WARNING: Could not find bundled Java in {fiji_dir}")


def java_env():
    print("-" * 40)
    # 1. Check what Windows/Linux thinks JAVA_HOME is
    print(f"OS Environment JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    
    # 2. Check what ScyJava/ImageJ is ACTUALLY using
    # (This is often different if scyjava found its own bundled Java)
    try:
        System = scyjava.jimport('java.lang.System')
        print(f"Active JVM java.home:     {System.getProperty('java.home')}")
        print(f"Active JVM Version:       {System.getProperty('java.version')}")
    except Exception as e:
        print(f"Could not get JVM info: {e}")
    print("-" * 40)

    
def imagej_init(fiji_path: str = None, mode="interactive"):
    """Initializes pyImageJ"""
    # --- STEP 0: INIT IMAGEJ ---
    print(f"Initializing ImageJ from: {fiji_path}...")
    try:
        ij = imagej.init(fiji_path, mode)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not start ImageJ.\n{e}")
        return None, None

    print(f"ImageJ version: {ij.getVersion()}")

    # Import IJ class
    IJ = scyjava.jimport('ij.IJ')

    return ij


def run_ridge():
    # 1. SWAP JVM (Must be first)
    # use_fiji_jvm(FIJI_PATH)
    
    # 2. CHECK ENV (Triggers JVM Start)
    print("Starting JVM...")
    java_env()
    
    # Import necessary classes
    ij = imagej_init(FIJI_PATH)
    return
    IJ = ij.IJ
    WindowManager = ij.WindowManager
    ResultsTable = ij.ResultsTable

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
    rt = ResultsTable.getResultsTable("Results")
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


    print(rt)

    # --- CLEANUP ---
    WindowManager.setTempCurrentImage(None)
    imp.close()

    #print(results)


if __name__ == "__main__":
    run_ridge()
