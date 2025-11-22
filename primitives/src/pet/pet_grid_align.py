"""
pet_grid_align.py
-----------------

https://chatgpt.com/c/6915c9bb-ec70-832a-94a1-560ec524b942
"""

import numpy as np
from scipy.optimize import minimize_scalar


def get_optimal_rotation(centers, reference_angle=0.0, search_range_deg=5.0, optimize_axis='x', bin_method='pixel'):
    """
    Finds the optimal rotation angle that maximizes grid alignment structure.
    
    Args:
        centers: (N, 2) array of x,y coordinates.
        reference_angle: The expected rotation angle (in degrees). The search will happen 
                         in the range [reference_angle - search_range, reference_angle + search_range].
                         Example: If your grid is tilted ~10 deg, set this to -10 to straighten it.
        search_range_deg: The +/- search window size (degrees).
        optimize_axis: 'x' -> Rotates to make lines Vertical (clusters X-coordinates).
                       'y' -> Rotates to make lines Horizontal (clusters Y-coordinates).
        bin_method: 'pixel' (default) -> Sets bin width to ~0.5 pixels. Best for images.
                    'fd' -> Freedman-Diaconis rule (robust stats).
                    int -> Forces a specific number of bins.
                       
    Returns:
        optimal_angle: The angle (in degrees) to rotate the points by.
    """
    # Center the data
    mean_centered = centers - np.mean(centers, axis=0)
    x = mean_centered[:, 0]
    y = mean_centered[:, 1]

    # --- 1. Pre-calculate Bin Config ---
    # Calculate bounds based on the max extent of the point cloud
    max_coord = np.max(np.abs(mean_centered))
    data_range = max_coord * 2.5 # Buffer for diagonal rotation
    
    if isinstance(bin_method, int):
        num_bins = bin_method
    elif bin_method == 'pixel':
        target_width = 0.5 # 0.5 pixel precision
        num_bins = int(data_range / target_width)
    elif bin_method == 'fd':
        # Use simple IQR estimation on the primary axis
        data = x if optimize_axis == 'x' else y
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        if iqr == 0: iqr = np.std(data) # Fallback for perfect grids
        bin_width = 2 * iqr / (len(data) ** (1/3)) if len(data) > 0 else 1.0
        if bin_width == 0: bin_width = 1.0
        num_bins = int(data_range / bin_width)
    
    # Safety clips for bin count
    num_bins = max(100, min(num_bins, 5000))

    # --- 2. Optimization ---
    def objective(angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        
        # Projection Logic
        if optimize_axis.lower() == 'x':
            # Rotate to align Verticals (constant X)
            proj = x * c - y * s
        else:
            # Rotate to align Horizontals (constant Y)
            proj = x * s + y * c
        
        # Calculate histogram entropy (Spikiness)
        # Using fixed range prevents "bin jitter" during optimization
        counts, _ = np.histogram(proj, bins=num_bins, range=(-data_range/2, data_range/2))
        
        # We want to MAXIMIZE sum of squares (peaks), so we return negative
        return -np.sum(counts**2)

    # Search bounds centered around the reference_angle
    bounds = (reference_angle - search_range_deg, reference_angle + search_range_deg)

    result = minimize_scalar(
        objective, 
        bounds=bounds, 
        method='bounded'
    )
    
    return result.x
