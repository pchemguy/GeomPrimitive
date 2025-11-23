"""
```
pet_grid_nodes_bbox.py
----------------------
"""

__all__ = [
    "get_grid_pitch", "get_grid_bbox", "plot_grid_bbox", "diagnose_and_fix_eps",
]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.signal import find_peaks, savgol_filter


# Usage
# pitch, info = analyze_gated_topology(points)
def analyze_gated_topology(points):
    """
    1. SCOUT: Calculates Ref = Average(90th% of 2nd Nbr, 90th% of 3rd Nbr).
    2. GATE: Defines search window [0.75 * Ref, 1.75 * Ref].
    3. TOPOLOGY: Scores peaks inside the window based on prominence + harmonics.
    """
    
    # --- STEP 1: THE SCOUT (Calculate Reference) ---
    # We need indices 2 (2nd neighbor) and 3 (3rd neighbor)
    nbrs_scout = NearestNeighbors(n_neighbors=4).fit(points)
    distances_scout, _ = nbrs_scout.kneighbors(points)

    dist_2nd = distances_scout[:, 2]
    dist_3rd = distances_scout[:, 3]

    p90_2nd = np.percentile(dist_2nd, 90)
    p90_3rd = np.percentile(dist_3rd, 90)

    # The robust reference
    ref_pitch = (p90_2nd + p90_3rd) / 2.0
    
    # The User-Defined Window
    search_min = ref_pitch * 0.75
    search_max = ref_pitch * 1.75

    print(f"[Scout] 2nd/3rd Nbr 90%: {p90_2nd:.2f} / {p90_3rd:.2f}")
    print(f"[Gate] Reference: {ref_pitch:.2f} px. Window: {search_min:.1f} - {search_max:.1f} px")

    # --- STEP 2: FULL DATA COLLECTION ---
    # We scan deep (K=8) to ensure we see the harmonics for scoring
    nbrs = NearestNeighbors(n_neighbors=8).fit(points)
    distances, _ = nbrs.kneighbors(points)
    all_dists = distances.flatten()
    
    # Histogram range: Go up to 3.0x to visualize the 2P harmonic clearly
    max_range = ref_pitch * 3.0
    bins = np.arange(0, max_range, 0.5)
    counts, bin_edges = np.histogram(all_dists, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the signal
    smoothed = savgol_filter(counts, window_length=15, polyorder=3)
    
    # --- STEP 3: FIND ALL PEAKS ---
    # We find peaks everywhere first, because a valid peak inside the window
    # might rely on a "2P" harmonic peak that is OUTSIDE the window.
    peaks, properties = find_peaks(smoothed, prominence=np.max(smoothed)*0.05, width=1)
    
    if len(peaks) == 0:
        return None, {}
        
    peak_locs = centers[peaks]
    peak_heights = smoothed[peaks]
    
    # --- STEP 4: GATED SCORING ENGINE ---
    candidates = []
    
    for i, p_loc in enumerate(peak_locs):
        # *** THE GATE ***
        # Strictly ignore candidates outside the user's window
        if p_loc < search_min or p_loc > search_max:
            continue
            
        # Base Score: Log Prominence
        score = np.log(peak_heights[i] + 1) * 10 
        
        # --- TOPOLOGY CHECK ---
        # Look for harmonics in the FULL peak list (all_peak_locs)
        target_diagonal = p_loc * 1.414
        target_second = p_loc * 2.0
        
        # Tolerance +/- 10%
        has_diagonal = np.any(np.abs(peak_locs - target_diagonal) < target_diagonal * 0.1)
        has_second = np.any(np.abs(peak_locs - target_second) < target_second * 0.1)
        
        confidence_label = "Weak"
        if has_diagonal:
            score += 50 
            confidence_label = "Confirmed (Diag)"
        if has_second:
            score += 20 
            if has_diagonal: confidence_label = "Strong Lock"
            
        candidates.append({
            'pitch': p_loc,
            'score': score,
            'height': peak_heights[i],
            'label': confidence_label,
            'has_diag': has_diagonal
        })
    
    if not candidates:
        print("Scout failed: No valid peaks found inside the gate.")
        return p90_2nd, p90_3rd, None, {}

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    winner = candidates[0]
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    plt.plot(centers, smoothed, 'k-', alpha=0.6, label='Signal')
    plt.fill_between(centers, counts, color='gray', alpha=0.1)
    
    # Plot The Gate
    plt.axvspan(search_min, search_max, color='green', alpha=0.1, label='Gated Region')
    
    # Plot Candidates
    for c in candidates:
        alpha = 1.0 if c == winner else 0.4
        color = 'green' if c == winner else 'orange'
        plt.plot(c['pitch'], c['height'], 'o', color=color, alpha=alpha)
        if c == winner:
             plt.text(c['pitch'], c['height'] + 5, f"WINNER\n{c['score']:.1f}", 
                      color='green', ha='center', fontsize=9, fontweight='bold')

    # Plot Harmonic Expectations for the Winner
    w_p = winner['pitch']
    plt.axvline(w_p, color='lime', linestyle='--', linewidth=2)
    plt.axvline(w_p * 1.414, color='blue', linestyle=':', label='Exp. Diag (1.41)')
    plt.axvline(w_p * 2.0, color='purple', linestyle=':', label='Exp. 2nd (2.0)')

    plt.title(f"Gated Topology: Winner = {w_p:.2f} px")
    plt.xlabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return p90_2nd, p90_3rd, winner['pitch'], winner


# Usage
# pitch, details = analyze_grid_topology(points)
# print(f"Detected Pitch: {pitch:.2f} with confidence: {details['label']}")
def analyze_grid_topology(points):
    """
    Robustly identifies grid pitch by scoring peaks based on:
    1. Prominence (Shape)
    2. Harmonic Support (Geometry: x, 1.41x, 2x)
    """
    # 1. Collect Data (Deep search up to K=8 to find harmonics)
    nbrs = NearestNeighbors(n_neighbors=8).fit(points)
    distances, _ = nbrs.kneighbors(points)
    all_dists = distances.flatten()
    
    # 2. Adaptive Histogram
    # We use a percentile to find a reasonable max range, avoiding outliers
    max_dist = np.percentile(all_dists, 90) * 2.5
    bins = np.arange(0, max_dist, 0.5) # 0.5px resolution
    counts, bin_edges = np.histogram(all_dists, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 3. Smooth & Find All Candidates
    # Savgol filter preserves peak height better than Gaussian
    smoothed = savgol_filter(counts, window_length=15, polyorder=3)
    
    # Find peaks with very loose criteria (we filter later based on score)
    peaks, properties = find_peaks(smoothed, prominence=np.max(smoothed)*0.05, width=1)
    
    if len(peaks) == 0:
        return None, {}
        
    peak_locs = centers[peaks]
    peak_heights = smoothed[peaks]
    
    # 4. SCORING ENGINE
    candidates = []
    
    for i, p_loc in enumerate(peak_locs):
        if p_loc < 5: continue # Hard physics limit: Grid can't be 0-5px
        
        # Base Score: Log of height (Prominence)
        # We use log because signal strength can vary wildly
        score = np.log(peak_heights[i] + 1) * 10 
        
        # --- TOPOLOGY CHECK: HARMONICS ---
        # A real grid pitch P MUST have a diagonal neighbor at 1.41 * P
        
        target_diagonal = p_loc * 1.414
        target_second = p_loc * 2.0
        
        # Search for these harmonic peaks in our existing peak list
        # Tolerance: +/- 10%
        has_diagonal = np.any(np.abs(peak_locs - target_diagonal) < target_diagonal * 0.1)
        has_second = np.any(np.abs(peak_locs - target_second) < target_second * 0.1)
        
        confidence_label = "Weak"
        
        if has_diagonal:
            score += 50 # HUGE Bonus: Geometry confirmed
            confidence_label = "Confirmed (Diag)"
        
        if has_second:
            score += 20 # Bonus: Lattice confirmed
            if has_diagonal: confidence_label = "Strong Lock"
            
        candidates.append({
            'pitch': p_loc,
            'score': score,
            'height': peak_heights[i],
            'label': confidence_label,
            'has_diag': has_diagonal
        })
    
    if not candidates:
        print("No valid grid structure found.")
        return None, {}

    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    winner = candidates[0]
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    plt.plot(centers, smoothed, 'k-', alpha=0.5, label='Signal')
    plt.fill_between(centers, counts, color='gray', alpha=0.1)
    
    # Plot all candidates
    for c in candidates:
        color = 'green' if c['has_diag'] else 'orange'
        alpha = 1.0 if c == winner else 0.3
        plt.plot(c['pitch'], c['height'], 'o', color=color, alpha=alpha)
        if c == winner or c['has_diag']:
            plt.text(c['pitch'], c['height'], f"{c['label']}\nScore: {c['score']:.1f}", 
                     fontsize=9, rotation=45)

    # Highlight Winner
    plt.axvline(winner['pitch'], color='green', linestyle='--', linewidth=2, label=f"Winner: {winner['pitch']:.2f}")
    
    # Show the harmonic locations for the winner
    if winner['pitch']:
        w_p = winner['pitch']
        plt.axvline(w_p * 1.414, color='blue', linestyle=':', alpha=0.5, label='Exp. Diagonal')
        plt.axvline(w_p * 2.0, color='purple', linestyle=':', alpha=0.5, label='Exp. 2nd Nbr')

    plt.title(f"Topological Analysis: Winner = {winner['pitch']:.2f} px ({winner['label']})")
    plt.xlabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return winner['pitch'], winner


# Usage
# best_pitch = get_auto_tuned_pitch(extracted_points)
def get_auto_tuned_pitch(points):
    """
    Robust Grid Pitch Detector.
    Uses a 'Scout' (Percentile) to set dynamic limits,
    then uses a 'Sniper' (Peak Analysis) to find the exact pitch.
    """
    # --- PHASE 1: THE SCOUT (Coarse Estimation) ---
    # Look at K=3 (3rd neighbor) to jump over potential doublets/clumps
    nbrs = NearestNeighbors(n_neighbors=4).fit(points) # 4 includes self
    distances, _ = nbrs.kneighbors(points)
    
    # We grab the distance to the 3rd neighbor (index 3)
    # We use a high percentile (90th) to ensure we are looking at the 'grid structure'
    # not the 'clump structure'.
    scout_dist = distances[:, 3]
    rough_estimate = np.percentile(scout_dist, 90)
    
    # Define Dynamic Safety Floor
    # We assume the true pitch is roughly equal to this estimate (or slightly lower).
    # Setting the floor at 33% ensures we cut out noise (0-10%) but never cut out
    # the fundamental pitch even if our estimate locked onto a diagonal. (Trying 50%).
    safe_cutoff = rough_estimate * 0.5
    
    print(f"[Scout] Rough Estimate (the 3rd neighbor, 90th percentile): {rough_estimate:.2f} px")
    print(f"[Scout] Dynamic Noise Floor set to: {safe_cutoff:.2f} px")
    
    # --- PHASE 2: THE SNIPER (Precision Peak Finding) ---
    # 1. Histogram of ALL distances (flattened)
    all_dists = distances.flatten()
    
    # Limit histogram range to 2.5x the scout estimate (we don't need to look at infinity)
    max_range = rough_estimate * 2.5
    bins = np.arange(0, max_range, 0.5) # 0.5px resolution
    counts, bin_edges = np.histogram(all_dists, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Smooth the data (Savitzky-Golay) for shape analysis
    smoothed = savgol_filter(counts, window_length=15, polyorder=3)
    
    # 3. Find Peaks
    # prominence=5% of max height (very sensitive, because we will filter later)
    max_h = np.max(smoothed)
    peaks, properties = find_peaks(smoothed, prominence=max_h * 0.05)
    
    peak_locs = centers[peaks]
    peak_prominences = properties['prominences']
    
    # 4. Filter Peaks using the Scout's Cutoff
    valid_indices = np.where(peak_locs > safe_cutoff)[0]
    
    if len(valid_indices) == 0:
        print("Scout failed to find safe peaks. Defaulting to Scout estimate.")
        return rough_estimate

    valid_locs = peak_locs[valid_indices]
    valid_proms = peak_prominences[valid_indices]
    
    # 5. Select the "Left-Most Strong Peak"
    # We want the smallest distance (fundamental pitch), 
    # but it must have significant prominence compared to its neighbors.
    
    # Find the max prominence among valid peaks
    max_prom = np.max(valid_proms)
    
    # We consider a peak "strong" if it has at least 30% of the max prominence
    # This rejects small "echoes" that might survive the cutoff
    strong_indices = np.where(valid_proms > max_prom * 0.3)[0]
    strong_locs = valid_locs[strong_indices]
    
    # Pick the smallest distance among the strong peaks
    final_pitch = np.min(strong_locs)
    
    # --- PLOT DIAGNOSTIC ---
    plt.figure(figsize=(10, 6))
    plt.plot(centers, smoothed, 'k-', alpha=0.8, label='Smoothed Data')
    plt.fill_between(centers, counts, color='gray', alpha=0.2)
    
    # Draw the Cutoff Zone
    plt.axvspan(0, safe_cutoff, color='red', alpha=0.1, label='Dynamic Noise Zone')
    plt.axvline(x=safe_cutoff, color='red', linestyle='--', label=f'Cutoff ({safe_cutoff:.1f}px)')
    
    # Mark the Winner
    winner_height = smoothed[np.searchsorted(centers, final_pitch)]
    plt.plot(final_pitch, winner_height, "o", color='lime', markersize=10, 
             label=f'DETECTED PITCH: {final_pitch:.2f} px')
    
    plt.title("Scout & Sniper Pitch Detection")
    plt.xlabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return final_pitch


# Usage
# p, std = get_histogram_pitch(extracted_points)
def get_histogram_pitch(points, bin_size=1.0):
    """
    Determines grid pitch by analyzing the histogram of all local distances.
    This method does not require an initial guess.
    """
    # 1. Collect ALL local distances
    # We look at K=1 to K=6 to capture cardinal, diagonal, and noise
    nbrs = NearestNeighbors(n_neighbors=6).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Flatten array to treat all connections equally
    all_distances = distances.flatten()
    
    # 2. Filter out "Self" (0) and extreme outliers
    # We cap at a reasonable upper limit (e.g., 200px) to keep the histogram readable
    valid_dists = all_distances[(all_distances > 0.5) & (all_distances < 200)]
    
    # 3. Create Histogram
    # Bins of 1 pixel width (or smaller for sub-pixel precision)
    bins = np.arange(0, 200, bin_size)
    counts, bin_edges = np.histogram(valid_dists, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 4. Find the Peaks
    # We define a "Peak" as a bin that is higher than its neighbors
    # (Simple local maxima detection)
    peaks_indices = []
    for i in range(1, len(counts)-1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
            # Simple threshold: Peak must represent meaningful data (e.g., > 5% of max)
            if counts[i] > np.max(counts) * 0.05: 
                peaks_indices.append(i)
                
    if not peaks_indices:
        print("No clear peaks found.")
        return None

    # 5. Select the Correct Peak (The Grid Pitch)
    # The first peak might be "clump noise" (e.g., at 2px or 5px)
    # We look for the first peak that is "significant" in distance
    # Let's assume grid pitch must be > 10 pixels to be valid
    valid_peaks = [i for i in peaks_indices if bin_centers[i] > 10]
    
    if not valid_peaks:
        print("Only small noise clusters found. Is the grid pitch < 10px?")
        return None
        
    # The first valid peak is the Cardinal Neighbor (Pitch)
    # The second valid peak is usually the Diagonal (Pitch * 1.41)
    best_peak_idx = valid_peaks[0]
    estimated_pitch = bin_centers[best_peak_idx]
    
    # 6. Refine: Windowed Mean around this specific peak
    # Now we can safely average because we know EXACTLY where the data lives
    window = 5.0 # +/- 5 pixels
    mask = (valid_dists > estimated_pitch - window) & (valid_dists < estimated_pitch + window)
    refined_pitch = np.mean(valid_dists[mask])
    refined_std = np.std(valid_dists[mask])
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    plt.bar(bin_centers, counts, width=bin_size, color='gray', label='All Distances')
    plt.axvline(x=refined_pitch, color='red', linestyle='-', linewidth=2, label=f'Detected Pitch: {refined_pitch:.2f}')
    
    # Annotate the peak
    plt.text(refined_pitch + 2, np.max(counts)*0.9, f"{refined_pitch:.2f} px\n(std {refined_std:.2f})", color='red')
    
    plt.title("Distance Histogram Analysis")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return refined_pitch, refined_std


# Usage
# pitch, std, count = get_super_precise_pitch(points)
# print(f"Pitch: {pitch:.4f} +/- {std:.4f} (based on {count} connections)")
def get_grid_pitch(points):
    # 1. ROBUST GUESS (The "Gatekeeper")
    # Use K=2 and 90th percentile to ignore clumps and find the general scale
    nbrs = NearestNeighbors(n_neighbors=6).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # K=2 (index 1) or K=3 (index 2) usually safe for rough estimate
    # We use 90th percentile of K=2 as our robust anchor
    rough_pitch = np.percentile(distances[:, 1], 90)
    
    # 2. DEFINE WINDOW
    # We only trust distances that are within 20% of our rough guess.
    # This rejects clumps (near 0) and diagonals (near 1.41 * pitch)
    lower_bound = rough_pitch * 0.80
    upper_bound = rough_pitch * 1.20
    
    # 3. COLLECT ALL CANDIDATES
    # Flatten the distance array. We don't care which neighbor is which anymore.
    # We just want "all distances that look like a grid connection."
    all_distances = distances.flatten()
    
    # 4. FILTER AND AVERAGE
    # Boolean mask: Keep only values in the window
    valid_mask = (all_distances > lower_bound) & (all_distances < upper_bound)
    valid_samples = all_distances[valid_mask]
    
    if len(valid_samples) == 0:
        return rough_pitch, None, 0 # Fallback if data is weird
        
    precise_pitch = np.mean(valid_samples)
    std_dev = np.std(valid_samples)

    print(f"Grid pitch: {precise_pitch:.1f}, standard deviation: {std_dev:.1f}")
    
    return precise_pitch, std_dev, len(valid_samples)


def get_grid_bbox(points, eps=None):
    """
    Automatically detects grid orientation and bounding box, 
    filtering outliers without manual parameter tuning.
    """
    N = len(points)
    if N < 4: return None # Not enough data
    
    # --- AUTO-TUNE PARAMETERS ---
    # 1. Min Samples: 0.5% of data, but at least 3 points to form a cluster
    min_samples = max(3, int(0.005 * N))

    if eps is None:
        # ROBUST FIX: Use 90th percentile instead of median
        nbrs = NearestNeighbors(n_neighbors=3).fit(points)
        distances, _ = nbrs.kneighbors(points)
        # Distance to the 2nd closest neighbor (index 2, since 0 is self)
        # We use 90th percentile to capture the "grid pitch" even if points are clumped
        dist_metric = distances[:, 2] 
        grid_pitch = np.percentile(dist_metric, 90)
        
        # Allow for diagonal jumps (x1.41) + buffer
        eps = grid_pitch * 1.5
        print(f"Auto-Tuning: Grid Pitch ~{grid_pitch:.2f}, calculated eps={eps:.2f}")    
    
    # --- STEP 1: CLEAN (DBSCAN) ---
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Filter noise
    # We only keep the largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0: 
        print("All points considered noise! Try increasing eps manually.")
        return None, None, None
            
    largest_cluster_label = unique_labels[np.argmax(counts)]
    clean_mask = (labels == largest_cluster_label)
    clean_points = points[clean_mask]

    # Report Stats
    noise_points = points[~clean_mask]    
    n_noise = N - len(clean_points)
    print(f"Outliers Removed: {n_noise} ({(n_noise/N)*100:.1f}%)")
    
    # --- STEP 2: ALIGN (Neighbor Vectors) ---
    # Use nearest neighbor vectors to find grid angle
    nbrs_clean = NearestNeighbors(n_neighbors=2).fit(clean_points)
    _, inds = nbrs_clean.kneighbors(clean_points)
    
    neighbors = clean_points[inds[:, 1]]
    vectors = neighbors - clean_points
    # Modulo 90 to align horizontal/vertical grid lines
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 90
    
    # Histogram to find peak angle
    hist, bin_edges = np.histogram(angles, bins=90, range=(0, 90))
    best_angle = bin_edges[np.argmax(hist)]
    
    # --- STEP 3: ENCLOSE (Rotate & Clip) ---
    theta = np.radians(best_angle)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array(((c, -s), (s, c)))
    
    # Rotate to axis-aligned
    rotated = clean_points @ R.T
    
    # Use percentiles to clip edges (robust min/max)
    x_min, x_max = np.percentile(rotated[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(rotated[:, 1], [0.5, 99.5])
    
    box_rot = np.array([
        [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    ])
    
    # Rotate back
    c, s = np.cos(theta), np.sin(theta)
    R_inv = np.array(((c, -s), (s, c)))
    box_final = box_rot @ R_inv.T
    
    return box_final, clean_points, noise_points, labels


# --- USE THIS IN YOUR PIPELINE ---

# 1. Run the diagnostic to see the true spacing
# new_eps = diagnose_and_fix_eps(extracted_points)
# print(f"New Robust EPS: {new_eps:.2f}")

# 2. Run the main function with the FIXED eps
# (Assuming you have the previous get_robust_grid_bbox_auto function)
# We override the internal auto-tune by modifying the function or just running DBSCAN manually here:

# ... Inside your main routine ...
# Instead of calling the auto-tuner blindly, pass the new_eps:
# Note: You might need to slightly modify your function to accept an `eps` override
# or just hardcode the new value found above.
def diagnose_and_fix_eps(points):
    """
    Plots a K-Distance graph to visually find the correct 'eps'.
    Also returns a robust epsilon estimate using the 90th percentile.
    """
    # 1. Calculate distance to the k-th nearest neighbor
    # k=3 is usually good for 2D data
    k = 3
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # We look at the distance to the k-th neighbor (column k-1)
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances)
    
    # 2. Heuristic: Use the 90th percentile to jump over local clumps
    # This finds the "grid spacing" rather than "clump spacing"
    percentile_90 = np.percentile(k_distances, 90)
    suggested_eps = percentile_90 * 1.5  # Add 50% buffer
    
    # --- PLOT THE DIAGNOSTIC GRAPH ---
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f'Suggested EPS: {suggested_eps:.2f}')
    plt.title("K-Distance Graph (The 'Elbow' Method)")
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.xlabel("Points (sorted by distance)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return suggested_eps


def plot_grid_bbox(points, bbox, labels=None, invert_y=False, title="Grid Detection"):
    """
    points: (N, 2) numpy array of all input points
    bbox: (4, 2) numpy array of the calculated bounding box corners
    labels: (Optional) Array of size N. 
            -1 indicates noise (plotted as red 'x'). 
            Any other value is considered a cluster (plotted as dots).
    invert_y: Set True if points are from image coordinates (y=0 at top)
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Plot Points (differentiated by noise vs clean if labels provided)
    if labels is not None:
        # Boolean mask for noise (label -1) vs valid (anything else)
        noise_mask = (labels == -1)
        clean_mask = ~noise_mask
        
        # Plot Outliers (Red X)
        if np.any(noise_mask):
            plt.scatter(points[noise_mask, 0], points[noise_mask, 1], 
                        c='red', marker='x', label=f'Outliers ({np.sum(noise_mask)})')
        
        # Plot Valid Grid Points (Purple circles)
        plt.scatter(points[clean_mask, 0], points[clean_mask, 1], 
                    c='purple', alpha=0.6, s=20, label='Grid Points')
    else:
        # If no labels provided, just plot everything as blue
        plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6, s=20)

    # 2. Plot Bounding Box
    if bbox is not None:
        # We must append the first point to the end to close the loop visually
        box_plot = np.vstack([bbox, bbox[0]])
        plt.plot(box_plot[:, 0], box_plot[:, 1], 'g-', linewidth=2.5, label='Robust BBox')
        
        # Optional: Plot corner points to see order
        plt.scatter(bbox[:, 0], bbox[:, 1], c='green', s=50, zorder=5)

    # 3. Formatting
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # CRITICAL: Set aspect ratio to equal. 
    # Without this, a square grid looks like a rectangle, and angles look wrong.
    plt.axis('equal')
    
    if invert_y:
        plt.gca().invert_yaxis()  # Matches image coordinate system (0,0 at top left)
        
    plt.show()


"""
```
"""
