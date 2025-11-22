import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def calculate_multi_slice_spacing(centers, plot_distribution=True):
    """
    Calculates grid spacing by iterating through 10 down to 5 vertical splits,
    computing spacing for each slice, and aggregating the results.
    """
    # 1. Define Robust Y-Span (ignore top/bottom 1% outliers)
    ys = centers[:, 1]
    y_min, y_max = np.percentile(ys, [1, 99])
    total_span = y_max - y_min
    
    collected_spacings = []

    # 2. Iterate from 10 slices down to 5 slices
    split_counts = range(10, 4, -1) # [10, 9, 8, 7, 6, 5]
    
    print(f"{'Splits':<10} | {'Slice #':<10} | {'Points':<10} | {'Spacing (px)':<15}")
    print("-" * 55)

    for n_splits in split_counts:
        slice_height = total_span / n_splits
        
        for i in range(n_splits):
            # Define slice boundaries
            y_start = y_min + (i * slice_height)
            y_end = y_start + slice_height
            
            # Filter points in this slice
            mask = (ys >= y_start) & (ys < y_end)
            subset = centers[mask]
            
            # Skip if slice is too empty (e.g., inside the Petri dish gap)
            if len(subset) < 10: 
                continue
                
            # --- Core Spacing Calculation for this Slice ---
            x_coords = subset[:, 0]
            
            # Histogram
            # Dynamic range for bins ensures we don't create massive arrays for small slices
            if len(x_coords) == 0: continue
            x_local_min, x_local_max = x_coords.min(), x_coords.max()
            bins = np.arange(x_local_min, x_local_max + 1, 1)
            counts, bin_edges = np.histogram(x_coords, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find Peaks
            peaks, _ = find_peaks(counts, height=1, distance=10, prominence=2)
            peak_locs = bin_centers[peaks]
            
            if len(peak_locs) < 2:
                continue
            
            # Calculate diffs (spacings)
            diffs = np.diff(peak_locs)
            
            # Robust filtering for THIS slice:
            # We use median to find the "base" spacing and reject gaps (missing lines)
            median_diff = np.median(diffs)
            
            # Only accept spacings that are within 20% of the median
            # This filters out the "double spacing" caused by gaps
            valid_diffs = diffs[np.abs(diffs - median_diff) < (0.2 * median_diff)]
            
            if len(valid_diffs) > 0:
                # Average the valid gaps for this specific slice
                slice_spacing = np.mean(valid_diffs)
                collected_spacings.append(slice_spacing)
                # print(f"{n_splits:<10} | {i:<10} | {len(subset):<10} | {slice_spacing:.2f}")

    # 3. Final Aggregation
    all_values = np.array(collected_spacings)
    
    # Remove global outliers before final averaging 
    # (e.g., if one slice was extremely noisy and gave a wrong result)
    q25, q75 = np.percentile(all_values, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    clean_values = all_values[(all_values >= lower_bound) & (all_values <= upper_bound)]
    
    final_average = np.mean(clean_values)
    final_std = np.std(clean_values)

    print("-" * 55)
    print(f"Raw Estimates Count: {len(all_values)}")
    print(f"Clean Estimates Count: {len(clean_values)}")
    print(f"FINAL ROBUST SPACING: {final_average:.4f} px (+/- {final_std:.4f})")

    # 4. Visualization
    if plot_distribution:
        plt.figure(figsize=(10, 5))
        plt.hist(all_values, bins=30, alpha=0.5, color='gray', label='All Slice Estimates')
        plt.hist(clean_values, bins=30, alpha=0.7, color='blue', label='Clean Estimates')
        plt.axvline(final_average, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_average:.2f}')
        plt.xlabel("Calculated Spacing (px)")
        plt.ylabel("Frequency (Count of Slices)")
        plt.title(f"Distribution of Spacing Estimates\n(Aggregated from splits 10 down to 5)")
        plt.legend()
        plt.show()

    return final_average


def validate_period(centers, estimated_period):
    """
    Performs visual and statistical validation of a grid period.
    """
    x = centers[:, 0]
    
    # --- 1. The Phase Fold (Modulo Plot) ---
    # If period is correct, x % period should be constant (roughly).
    # If period is off by 0.1px, the error at x=1000 will be off by several pixels.
    
    phase = x % estimated_period
    
    # Handle "wrapping" (points near 0 and points near 'period' are actually the same)
    # We shift points near the top boundary down to group them visually
    phase[phase > estimated_period * 0.9] -= estimated_period
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Phase Consistency (The "Drift" Check)
    plt.subplot(2, 1, 1)
    plt.scatter(x, phase, s=3, alpha=0.6, c='blue')
    plt.axhline(np.median(phase), color='r', linestyle='--', alpha=0.5)
    plt.title(f"Method 1: Phase Consistency Check (Period = {estimated_period:.3f}px)")
    plt.xlabel("X Coordinate (px)")
    plt.ylabel("Remainder (x % Period)")
    plt.grid(True, alpha=0.3)
    
    # Interpretation Text
    plt.text(x.min(), phase.max(), 
             "GOOD: Flat horizontal band\nBAD: Tilted line (Drift)", 
             bbox=dict(facecolor='white', alpha=0.8))

    # --- 2. Sensitivity Analysis (Slice Sweep) ---
    # We recalculate the period using different slice counts (3 to 30)
    # To see if the result depends on the specific slicing choice.
    
    slice_counts = range(3, 101)
    results = []
    
    # Re-using a simplified version of the previous logic for the sweep
    y = centers[:, 1]
    y_min, y_max = np.percentile(y, [1, 99])
    
    for n in slice_counts:
        # Simple average of median diffs for this N
        slice_h = (y_max - y_min) / n
        local_estimates = []
        for i in range(n):
            ys, ye = y_min + i*slice_h, y_min + (i+1)*slice_h
            mask = (y >= ys) & (y < ye)
            sub_x = np.sort(centers[mask, 0])
            if len(sub_x) > 5:
                diffs = np.diff(sub_x)
                med = np.median(diffs)
                # Filter extreme outliers
                valid = diffs[np.abs(diffs-med) < med*0.2]
                if len(valid) > 0: local_estimates.append(np.mean(valid))
        
        if local_estimates:
            results.append(np.mean(local_estimates))
        else:
            results.append(np.nan)

    plt.subplot(2, 1, 2)
    plt.plot(slice_counts, results, 'o-', color='green')
    plt.axhline(estimated_period, color='r', linestyle='--', label='Current Estimate')
    plt.title("Method 2: Sensitivity Analysis (Stability vs Slicing)")
    plt.xlabel("Number of Slices Used")
    plt.ylabel("Calculated Period (px)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    # --- 3. Residual Standard Deviation ---
    # How strictly do points adhere to this grid?
    residuals = np.std(phase)
    print(f"Residual RMS Error: {residuals:.4f} pixels")
    if residuals < 1.0:
        print("Verdict: HIGH ACCURACY (Sub-pixel consistency)")
    elif residuals < 2.0:
        print("Verdict: MODERATE ACCURACY (Likely noise or slight distortion)")
    else:
        print("Verdict: LOW ACCURACY (Grid drift detected)")


def monte_carlo_grid_spacing(centers, num_runs=50, max_slices=10, plot_convergence=True):
    """
    Estimates grid spacing using a Monte Carlo approach with random dropout 
    and random slicing.
    
    Args:
        centers: (N, 2) array of points.
        num_runs: How many iterations to perform.
        max_slices: Max number of random slices to take per run.
        
    Returns:
        final_estimate: The converged average.
        history: List of the running average at each step.
    """
    # 1. Pre-calculate data bounds
    ys = centers[:, 1]
    y_min_global, y_max_global = np.percentile(ys, [1, 99])
    y_span_global = y_max_global - y_min_global
    
    run_averages = []      # The average period found in each specific run
    evolution_history = [] # The cumulative average over time
    
    print(f"Starting Monte Carlo Estimation ({num_runs} runs)...")

    for i in range(num_runs):
        # --- A. Random Dropout (0% to 50%) ---
        # Determine dropout rate for this specific run
        dropout_rate = np.random.uniform(0, 0.50)
        
        # Create mask to keep points
        n_points = len(centers)
        # Calculate how many to keep
        n_keep = int(n_points * (1 - dropout_rate))
        # Randomly choose indices without replacement
        keep_indices = np.random.choice(n_points, n_keep, replace=False)
        
        current_points = centers[keep_indices]
        current_ys = current_points[:, 1]
        
        # --- B. Random Slicing ---
        # Determine number of slices for this run (1 to 10)
        n_slices = np.random.randint(1, max_slices + 1)
        
        slice_estimates = []
        
        for _ in range(n_slices):
            # Random Slice Generation
            # Minimum height 5% of span, Max height 25% of span (to ensure local linearity)
            slice_height = np.random.uniform(y_span_global * 0.05, y_span_global * 0.25)
            
            # Random start position (ensure slice stays within bounds)
            max_start = y_max_global - slice_height
            if max_start <= y_min_global: continue # Safety check
            
            y_start = np.random.uniform(y_min_global, max_start)
            y_end = y_start + slice_height
            
            # Slice the dropped-out data
            mask = (current_ys >= y_start) & (current_ys < y_end)
            slice_subset = current_points[mask]
            
            # --- C. Calculate Period for this Slice ---
            period = _get_slice_period(slice_subset)
            if period is not None:
                slice_estimates.append(period)
        
        # --- D. Aggregate Run ---
        if len(slice_estimates) > 0:
            # Average of all random slices in this run
            run_avg = np.mean(slice_estimates)
            run_averages.append(run_avg)
        
        # Update Evolution History (Cumulative Mean)
        if len(run_averages) > 0:
            current_cumulative_avg = np.mean(run_averages)
            evolution_history.append(current_cumulative_avg)
        else:
            # If run failed (no valid slices), replicate previous state
            if len(evolution_history) > 0:
                evolution_history.append(evolution_history[-1])
            else:
                evolution_history.append(0) # Should not happen given valid data

    # Final Calculation
    final_estimate = evolution_history[-1]
    std_dev = np.std(run_averages)
    
    print(f"Converged Estimate: {final_estimate:.4f} px")
    print(f"Uncertainty (StdDev across runs): {std_dev:.4f}")

    # --- Visualization ---
    if plot_convergence:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: The individual runs (scatter) and the Running Average (line)
        plt.subplot(1, 2, 1)
        plt.plot(run_averages, 'o', color='gray', alpha=0.3, markersize=3, label='Individual Run Avg')
        plt.plot(evolution_history, '-', color='blue', linewidth=2, label='Running Average')
        plt.xlabel("Run Number")
        plt.ylabel("Estimated Period (px)")
        plt.title("Convergence History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of all runs
        plt.subplot(1, 2, 2)
        plt.hist(run_averages, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(final_estimate, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {final_estimate:.2f}')
        plt.xlabel("Estimated Period (px)")
        plt.title("Distribution of Run Estimates")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    return final_estimate, evolution_history


def _get_slice_period(subset):
    """Helper function to calculate spacing for a single slice"""
    if len(subset) < 5: return None
    
    x_coords = subset[:, 0]
    
    # Histogram
    # Use 1px bins for precision
    x_min, x_max = x_coords.min(), x_coords.max()
    if x_max - x_min < 10: return None
    
    bins = np.arange(x_min, x_max + 1, 1)
    counts, bin_edges = np.histogram(x_coords, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Peak Finding
    # Robust params: dist=10 (assuming period > 10), prominence=1
    peaks, _ = find_peaks(counts, distance=10, prominence=1, height=1)
    
    if len(peaks) < 2: return None
    
    peak_locs = bin_centers[peaks]
    diffs = np.diff(peak_locs)
    
    # Filter gaps
    median_diff = np.median(diffs)
    valid_diffs = diffs[np.abs(diffs - median_diff) < (0.3 * median_diff)]
    
    if len(valid_diffs) == 0: return None
    
    return np.mean(valid_diffs)

