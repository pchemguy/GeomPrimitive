"""
LSD HEALTH CHECK (FIXED)
========================

Handles missing NFA (None), missing precision, and crippled LSD builds.
"""


import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_test_image():
    img = np.zeros((400, 400), np.uint8)
    cv2.line(img, (50, 100), (350, 100), 255, 3)
    cv2.line(img, (200, 50), (200, 350), 255, 2)
    cv2.line(img, (50, 300), (300, 150), 255, 1)
    return img


def create_lsd():
    """Try all valid cv2 signatures."""
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        print("LSD created with signature: cv2.createLineSegmentDetector(flag)")
        return lsd
    except TypeError:
        try:
            lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
            print("LSD created with signature: cv2.createLineSegmentDetector(_refine=flag)")
            return lsd
        except TypeError:
            print("WARNING: Falling back to no-refine LSD.")
            return cv2.createLineSegmentDetector()


def safe_array(x, length):
    """Convert LSD output to array safely."""
    if x is None:
        return np.zeros((length,), np.float32)
    return np.asarray(x, np.float32).reshape(-1)


def run_health_check():

    print("cv2 version:", cv2.__version__)
    print("cv2 file:", cv2.__file__)
    
    bi = cv2.getBuildInformation()
    print("\n=== Has Contrib? ====================")
    print("xfeatures2d" in bi)
    print("Nonfree" in bi)
    print("LineSegmentDetector" in bi)
    
    print("\n=== Build Information (first 80 lines) ===")
    print("\n".join(bi.splitlines()[:80]))    
        
    img = make_test_image()
    lsd = create_lsd()

    lines, widths, precisions, nfa = lsd.detect(img)

    if lines is None:
        print("\nx LSD returned no lines.")
        return

    lines = lines.reshape(-1, 4)
    N = len(lines)

    widths = safe_array(widths, N)
    precisions = safe_array(precisions, N)
    nfa = safe_array(nfa, N)

    print("\n=== LSD DETECTION RESULTS ===")
    print(f"Detected segments: {N}")

    # ---------------------------------------------------------
    # WIDTH
    # ---------------------------------------------------------
    print("\nWidth statistics:")
    print(f"  count   = {N}")
    print(f"  min     = {widths.min():.5g}")
    print(f"  max     = {widths.max():.5g}")
    print(f"  mean    = {widths.mean():.5g}")
    print(f"  median  = {np.median(widths):.5g}")
    print(f"  std     = {widths.std():.5g}")

    # ---------------------------------------------------------
    # PRECISION
    # ---------------------------------------------------------
    print("\nPrecision statistics:")
    print(f"  count   = {N}")
    print(f"  min     = {precisions.min():.5g}")
    print(f"  max     = {precisions.max():.5g}")
    print(f"  mean    = {precisions.mean():.5g}")
    print(f"  median  = {np.median(precisions):.5g}")
    print(f"  std     = {precisions.std():.5g}")

    if (precisions.max() - precisions.min()) < 1e-6:
        print("x PRECISION CONSTANT - LSD refinement missing.")

    # ---------------------------------------------------------
    # NFA
    # ---------------------------------------------------------
    print("\nNFA statistics:")
    print(f"  count   = {N}")
    print(f"  min     = {nfa.min():.5g}")
    print(f"  max     = {nfa.max():.5g}")
    print(f"  mean    = {nfa.mean():.5g}")
    print(f"  median  = {np.median(nfa):.5g}")
    print(f"  std     = {nfa.std():.5g}")

    if np.all(nfa == 0):
        print("x NFA missing - this OpenCV build does NOT compute statistical validation.")

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    plt.figure(figsize=(5, 5))
    plt.title("LSD Detected Segments")
    plt.imshow(vis, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    run_health_check()
