"""
```
pet_grid_optimizer.py
---------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# Usage Example
# opt = GridOptimizer(clean_points) # Uses clean_points from DBSCAN
# angle_q1, score_q1 = opt.optimize_quartile(0) # Optimize Q1
# angle_q4, score_q4 = opt.optimize_quartile(3) # Optimize Q4
# print(f"Q1 Opt Angle: {angle_q1:.4f} | Q4 Opt Angle: {angle_q4:.4f}")
#
# If you expect the grid is rotated ~20 degrees CCW:
# opt = GridOptimizer(points)
# angle_q1, _ = opt.optimize_quartile(0, initial_angle=20.0, search_width=10.0)
# This will scan from 15.0 to 25.0 degrees.
# 
# ### How to Debug with this
# Run the optimization with `debug=True`.
#
class GridOptimizer:
    def __init__(self, points, bw=None):
        """
        points: (N, 2) array of x,y coordinates
        bw: Bandwidth for KDE. If None, auto-calculated based on pitch.
        """
        if len(points) == 0:
            raise ValueError("Optimizer received empty point set.")
            
        self.points = points
        
        # 1. Auto-calculate geometry
        self.center = np.mean(points, axis=0)
        self.x_range = (np.min(points[:, 0]), np.max(points[:, 0]))
        
        # 2. Auto-calculate pitch and default bandwidth
        self.pitch, self.bw = self._estimate_robust_pitch_bw(points)
        
        if bw is None:
            print(f"[GridOptimizer] Init. N={len(points)}, Pitch={self.pitch:.2f}, BW={self.bw:.2f}")
        else:
            self.bw = bw
            print(f"[GridOptimizer] Init. N={len(points)}, Pitch={self.pitch:.2f}, Manual BW={self.bw:.2f}")

        # 2. Compute Bounding Box & Orientation
        # We use the estimated pitch to set a robust eps for DBSCAN (1.5x pitch)
        bbox_eps = self.pitch * 1.5
        bbox, _, _, labels = self.get_grid_bbox(points, eps=bbox_eps, margin_ratio=0.25)
        
        if bbox is None:
            print("[GridOptimizer] Warning: BBox detection failed. Using raw points.")
            self.points = points
            self.bbox = None
            self.bbox_rotation_angle = 0.0
        else:
            self.bbox = bbox
            
            # 3. Reject Outliers (Noise points outside the determined BBox)
            self.points = self.reject_outliers(points, bbox, labels)
            
            # 4. Determine Rotation Angle from BBox (Angle of bottom edge)
            p0, p1 = bbox[0], bbox[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            self.bbox_rotation_angle = np.degrees(np.arctan2(dy, dx))
            
            print(f"[GridOptimizer] BBox Found. Rotation: {self.bbox_rotation_angle:.2f} deg")

        # 5. Update Geometry based on clean points
        self.center = np.mean(self.points, axis=0)
        self.x_range = (np.min(self.points[:, 0]), np.max(self.points[:, 0]))
                                
    def _estimate_robust_pitch_bw(self, points):
        """
        Estimates grid pitch using 2nd and 3rd nearest neighbors
        and sets bandwidth to 10% of that pitch.
        """
        # We need indices 0,1,2,3 (so k=4)
        k = min(len(points), 4)
        if k < 4:
            print("[GridOptimizer] Warning: Not enough points for robust BW. Defaulting to 1.0")
            return 1.0
            
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        dists, _ = nbrs.kneighbors(points)
        
        # dists columns: [0]=Self, [1]=1st NN, [2]=2nd NN, [3]=3rd NN
        dist_2nd = dists[:, 2]
        dist_3rd = dists[:, 3]
        
        p90_2nd = np.percentile(dist_2nd, 90)
        p90_3rd = np.percentile(dist_3rd, 90)
        
        # Check for anisotropy/rectangularity
        # Avoid division by zero
        denom = max(p90_2nd, p90_3rd) + 1e-9
        diff_ratio = abs(p90_2nd - p90_3rd) / denom
        
        if diff_ratio > 0.25:
            print(f"[GridOptimizer] Warning: Large Pitch Anisotropy detected ({diff_ratio:.1%}).")
            print(f"                2nd Neighbor: {p90_2nd:.2f} px | 3rd Neighbor: {p90_3rd:.2f} px")
            print(f"                Is the grid highly rectangular or 1D?")

        # The robust reference pitch
        ref_pitch = (p90_2nd + p90_3rd) / 2.0
        
        # Golden Rule: 10% of Pitch
        bw = ref_pitch * 0.10
        
        print(f"[GridOptimizer] Auto-BW: {bw:.2f} px (Pitch ~{ref_pitch:.2f} px)")
        return ref_pitch, bw

    def get_grid_bbox(self, points, eps=None, margin_ratio=0.25):
        """
        Automatically detects grid orientation and bounding box, 
        filtering outliers without manual parameter tuning.
        """
        N = len(points)
        if N < 4: return None, None, None, None
        
        # --- AUTO-TUNE PARAMETERS ---
        # 1. Min Samples: 0.5% of data, but at least 3 points to form a cluster
        min_samples = max(3, int(0.005 * N))

        if eps is None:
            eps = self.pitch * 1.5
        
        # --- STEP 1: CLEAN (DBSCAN) ---
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Filter noise
        # We only keep the largest cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0: 
            print("All points considered noise! Try increasing eps manually.")
            return None, None, None, None
                
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
        
        # A. Initial Tight Bounds (using robust percentiles)
        # We use 0.5/99.5 to ignore slight jitter at the very edges
        x_min, x_max = np.percentile(rotated[:, 0], [0.5, 99.5])
        y_min, y_max = np.percentile(rotated[:, 1], [0.5, 99.5])
        
        # B. Apply Dynamic Padding
        # Expand the box by margin_ratio * pitch
        padding = self.pitch * margin_ratio     
        
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding     
        
        box_rot = np.array([
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
        ])
        
        # Rotate back
        c, s = np.cos(theta), np.sin(theta)
        R_inv = np.array(((c, -s), (s, c)))
        box_final = box_rot @ R_inv.T
        
        return box_final, clean_points, noise_points, labels

    def reject_outliers(self, points, bbox, labels):
        """
        Filters points by keeping all cluster points and only checking geometry for noise points.
        
        Logic:
        1. Points with label != -1 (Cluster) -> KEPT automatically.
        2. Points with label == -1 (Noise)   -> CHECKED against bbox.
           - If inside bbox -> KEPT (Rescued).
           - If outside bbox -> DROPPED.

        Args:
            points: (N, 2) numpy array of x,y coordinates
            bbox: (4, 2) numpy array of the OBB corners
            labels: (N,) array from DBSCAN. REQUIRED.
            
        Returns:
            clean_points: (M, 2) numpy array of valid points.
        """
        if len(points) == 0:
            return np.array([])
        
        if labels is None:
            # Fallback if labels are missing: Check everything
            final_mask = np.zeros(len(points), dtype=bool)
            noise_indices = np.arange(len(points))
        else:
            # 1. Initialize mask with cluster points (Keep by default)
            final_mask = (labels != -1)
            # 2. Identify Noise Points to Check
            noise_indices = np.where(labels == -1)[0]
        
        if len(noise_indices) == 0:
            print("Outlier Rejection: 0 points dropped (0 labeled as noise).")
            return points

        # --- GEOMETRIC CHECK (ONLY ON NOISE POINTS) ---
        noise_points_to_check = points[noise_indices]

        # Corner 0 is usually Bottom-Left
        p0 = bbox[0]
        p1 = bbox[1]
        p3 = bbox[3]
        
        u = p1 - p0
        v = p3 - p0
        
        u_len_sq = np.dot(u, u)
        v_len_sq = np.dot(v, v)
        
        # Vector from p0 to noise points only
        w = noise_points_to_check - p0
        
        # Projection: Dot product
        proj_u = np.dot(w, u)
        proj_v = np.dot(w, v)
        
        # Check bounds
        in_u = (proj_u >= 0) & (proj_u <= u_len_sq)
        in_v = (proj_v >= 0) & (proj_v <= v_len_sq)
        
        is_inside_mask = in_u & in_v
        
        # Stats
        n_rescued = np.sum(is_inside_mask)
        n_dropped = len(noise_indices) - n_rescued
        print(f"Outlier Rejection: {n_dropped} points dropped (from {len(noise_indices)} noise candidates). {n_rescued} rescued.")
        
        # 3. Update the final mask
        final_mask[noise_indices] = is_inside_mask
        
        final_points = points[final_mask]
        
        return final_points

    def calculate_gini(self, density_array):
        """
        Calculates Gini Coefficient of a density profile.
        High Gini (>0.6) = Sharp peaks (Grid).
        Low Gini (<0.4) = Uniform/Noise.
        """
        # Ensure positive values and flatten
        y = np.abs(density_array.flatten()) + 1e-12
        
        # Sort values
        y = np.sort(y)
        n = len(y)
        
        # Standard Gini formula
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * y).sum() / (n * y.sum())

    def _get_projected_density(self, points_subset, angle):
        """Returns the normalized density profile for a specific rotation."""
        # Rotate
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        
        # Project onto X-axis (Vertical Line detection)
        # x' = x*cos - y*sin
        x_rot = (points_subset[:, 0] - self.center[0]) * c - \
                (points_subset[:, 1] - self.center[1]) * s
        
        # Handle degenerate case (all points at same X)
        if len(x_rot) == 0: return np.array([1.0])
        
        x_min, x_max = np.min(x_rot), np.max(x_rot)
        span = x_max - x_min
        
        # If span is tiny (single line), pad it to avoid division by zero
        if span < 1e-6: span = 1.0
            
        # KDE Grid
        grid = np.linspace(x_min - span*0.1, x_max + span*0.1, 500)
        
        # Vectorized KDE
        diffs = grid[:, None] - x_rot[None, :]
        pdfs = np.exp(-0.5 * (diffs / self.bw)**2)
        density = np.sum(pdfs, axis=1)
        
        # Normalize
        total_mass = np.sum(density)
        if total_mass == 0:
            return np.ones_like(density) / len(density)
            
        return density / total_mass

    def _objective_entropy(self, angle, points_subset):
        """Objective Function: Shannon Entropy (Minimize this)"""
        prob_dist = self._get_projected_density(points_subset, angle)
        return entropy(prob_dist)

    def optimize_quartile(self, q_index, initial_angle=0.0, search_width=10.0, debug=False):
        """
        Optimizes rotation for a specific quartile.
        
        Args:
            q_index: 0=Q1, 1=Q2, etc.
            initial_angle: Center of search (deg).
            search_width: Total scan range (deg).
            debug: If True, plots the entropy landscape.
            
        Returns: 
            (optimal_angle, entropy_score, gini_score)
        """
        # 1. Split Data
        x_curr = self.points[:, 0]
        sort_idx = np.argsort(x_curr)
        sorted_points = self.points[sort_idx]
        
        n = len(sorted_points)
        q_len = n // 4
        if q_len < 2:
            print(f"[Error] Quartile {q_index} has too few points ({q_len}).")
            return initial_angle, 0.0, 0.0
            
        start = q_index * q_len
        # Ensure the last quartile grabs any remainder points
        end = (q_index + 1) * q_len if q_index < 3 else n
        subset = sorted_points[start:end]
        
        # Define Bounds
        search_min = initial_angle - (search_width / 2.0)
        search_max = initial_angle + (search_width / 2.0)
        
        # 2. Coarse Sweep (Crucial to avoid local minima)
        # Step size 0.5 deg is usually safe for grids
        coarse_grid = np.arange(search_min, search_max, 0.5)
        if len(coarse_grid) == 0: coarse_grid = np.array([initial_angle])
            
        scores = [self._objective_entropy(a, subset) for a in coarse_grid]
        
        best_coarse_idx = np.argmin(scores)
        best_coarse_angle = coarse_grid[best_coarse_idx]
        best_coarse_score = scores[best_coarse_idx]
        
        # Debug Plotting
        if debug:
            self.plot_entropy_gini_landscape(subset, coarse_grid, scores, best_coarse_angle, q_index)

        # 3. Fine Optimization
        optimal_angle = best_coarse_angle
        final_entropy = best_coarse_score
        
        # Search +/- 1.0 deg around the coarse winner
        try:
            res = minimize_scalar(
                self._objective_entropy, 
                args=(subset,),
                bounds=(best_coarse_angle - 1.0, best_coarse_angle + 1.0),
                method='bounded'
            )
            
            if res.success:
                optimal_angle = res.x
                final_entropy = res.fun
            else:
                print(f"[Q{q_index+1}] Opt Failed. Using Coarse: {best_coarse_angle:.4f}")
                
        except Exception as e:
            print(f"[Q{q_index+1}] Exception in minimize_scalar: {e}")

        # 4. Post-Optimization Quality Check (Gini)
        final_density = self._get_projected_density(subset, optimal_angle)
        final_gini = self.calculate_gini(final_density)
        
        print(f"[Q{q_index+1}] Result: {optimal_angle:.4f} deg (Ent: {final_entropy:.2f}, Gini: {final_gini:.2f})")
        
        return optimal_angle, final_entropy, final_gini

    def _objective_gini_neg(self, angle, points_subset):
        """Objective: Maximize Gini (Minimize Negative Gini)"""
        dens = self._get_projected_density(points_subset, angle)
        return -self.calculate_gini(dens)

    def analyze_twist_profile(self, angle_start, angle_end, step=0.5, window_fraction=0.25):
        """
        Performs a 'Focal Plane Sweep' to analyze grid twist.
        For each angle, finds the X-location where the grid is sharpest (Max Gini).
        
        Args:
            angle_start, angle_end: Range of angles to sweep (e.g., 42 to 44).
            step: Angle step size.
            window_fraction: Size of the sliding window (0.0 to 1.0).
                             0.25 means the window covers 25% of the points.
        """
        print(f"[GridOptimizer] Analyzing Twist: {angle_start}deg -> {angle_end}deg (Window: {window_fraction:.0%})")
        
        # Sort data spatially once (approximation, assuming small rotation)
        # Ideally, we sort inside the loop, but sorting by raw X is usually stable enough for small angles
        x_raw = self.points[:, 0]
        sort_idx = np.argsort(x_raw)
        sorted_points = self.points[sort_idx]
        sorted_x_vals = x_raw[sort_idx]
        
        n = len(sorted_points)
        window_size = int(n * window_fraction)
        if window_size < 4:
            print("Window too small.")
            return

        angles = np.arange(angle_start, angle_end + step, step)
        
        # Results containers
        best_x_locs = []
        max_gini_vals = []
        
        for ang in angles:
            # Sliding Window Scan for THIS angle
            local_ginis = []
            window_centers = []
            
            # Slide with overlap (step size = 5% of N)
            slide_step = max(1, int(n * 0.05))
            
            for i in range(0, n - window_size, slide_step):
                subset = sorted_points[i : i + window_size]
                center_x = np.mean(sorted_x_vals[i : i + window_size])
                
                # Compute Sharpness (Gini) for this window
                dens = self._get_projected_density(subset, ang)
                gini = self.calculate_gini(dens)
                
                local_ginis.append(gini)
                window_centers.append(center_x)
            
            # Find where the sharpness was maximized for this angle
            if not local_ginis:
                best_x_locs.append(np.nan)
                max_gini_vals.append(np.nan)
                continue
                
            best_idx = np.argmax(local_ginis)
            best_x_locs.append(window_centers[best_idx])
            max_gini_vals.append(local_ginis[best_idx])

        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Trace 1: Location of Best Alignment (Left Axis)
        color_loc = 'tab:red'
        ax1.set_xlabel('Rotation Angle (deg)')
        ax1.set_ylabel('X-Location of Max Sharpness', color=color_loc, fontweight='bold')
        ax1.plot(angles, best_x_locs, 'o-', color=color_loc, linewidth=2, label='Focus Point (X)')
        ax1.tick_params(axis='y', labelcolor=color_loc)
        ax1.grid(True, alpha=0.3)
        
        # Indicate image bounds on Y-axis
        ax1.axhline(self.x_range[0], color='gray', linestyle=':', alpha=0.5, label='Left Edge')
        ax1.axhline(self.x_range[1], color='gray', linestyle=':', alpha=0.5, label='Right Edge')

        # Trace 2: Max Gini Value (Right Axis)
        ax2 = ax1.twinx()
        color_qual = 'tab:green'
        ax2.set_ylabel('Max Gini Score (Quality)', color=color_qual, fontweight='bold')
        ax2.plot(angles, max_gini_vals, 'x--', color=color_qual, alpha=0.7, label='Sharpness Score')
        ax2.tick_params(axis='y', labelcolor=color_qual)
        
        plt.title(f"Grid Twist Profile\nSweep: {angle_start}deg to {angle_end}deg")
        plt.tight_layout()
        plt.show()
        
        return angles, best_x_locs, max_gini_vals
    
    def analyze_spatial_twist(self, angle_center=0.0, search_width=10.0, num_slices=8):
        """
        Slices the grid into N vertical strips and finds the optimal angle for EACH strip.
        Returns the Twist Profile: Angle(x).
        
        Args:
            angle_center: Approximate correct angle (e.g., 43.0).
            search_width: Range to search (+/- 5.0).
            num_slices: How many vertical strips to divide the grid into.
        """
        print(f"[GridOptimizer] Analyzing Spatial Twist ({num_slices} slices)...")
        
        # 1. Sort Data Spatially
        x_raw = self.points[:, 0]
        sort_idx = np.argsort(x_raw)
        sorted_points = self.points[sort_idx]
        sorted_x = x_raw[sort_idx]
        
        n = len(sorted_points)
        slice_len = n // num_slices
        
        x_locations = []
        optimal_angles = []
        quality_scores = [] # Peak Gini
        
        search_min = angle_center - search_width/2
        search_max = angle_center + search_width/2
        
        # 2. Iterate through Slices
        for i in range(num_slices):
            # Define Slice Indices
            start = i * slice_len
            # Ensure last slice grabs remainder
            end = (i + 1) * slice_len if i < num_slices - 1 else n
            
            subset = sorted_points[start:end]
            
            # Skip empty/tiny slices
            if len(subset) < 4:
                continue
                
            # Calculate Slice Center X
            center_x = np.mean(sorted_x[start:end])
            
            # 3. Optimize Angle for THIS Slice
            # Use bounded minimization on Negative Gini (Maximize Sharpness)
            res = minimize_scalar(
                self._objective_gini_neg,
                args=(subset,),
                bounds=(search_min, search_max),
                method='bounded'
            )
            
            best_angle = res.x
            best_gini = -res.fun
            
            x_locations.append(center_x)
            optimal_angles.append(best_angle)
            quality_scores.append(best_gini)
            
            print(f"  Slice {i+1}: X={center_x:.1f} -> Angle={best_angle:.2f}deg (Gini: {best_gini:.2f})")

        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Trace 1: Twist Profile (Angle vs X)
        color_ang = 'tab:blue'
        ax1.set_xlabel('X-Location (pixels)')
        ax1.set_ylabel('Optimal Rotation Angle (deg)', color=color_ang, fontweight='bold')
        ax1.plot(x_locations, optimal_angles, 'o-', color=color_ang, linewidth=2, label='Twist Profile')
        ax1.tick_params(axis='y', labelcolor=color_ang)
        ax1.grid(True, alpha=0.3)
        
        # Fit a trendline
        if len(x_locations) > 1:
            z = np.polyfit(x_locations, optimal_angles, 1)
            p = np.poly1d(z)
            ax1.plot(x_locations, p(x_locations), 'b:', alpha=0.5, label=f'Trend: {z[0]*1000:.2f} mdeg/px')

        # Trace 2: Quality (Gini vs X)
        ax2 = ax1.twinx()
        color_qual = 'tab:green'
        ax2.set_ylabel('Grid Quality (Max Gini)', color=color_qual, fontweight='bold')
        ax2.plot(x_locations, quality_scores, 'x--', color=color_qual, alpha=0.7, label='Local Quality')
        ax2.tick_params(axis='y', labelcolor=color_qual)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.title(f"Spatially Resolved Grid Alignment\n(Twist Analysis)")
        plt.tight_layout()
        plt.show()
        
        return x_locations, optimal_angles, quality_scores

    def plot_entropy_landscape(self, subset, grid, scores, winner, q_idx):
        """Visualizes the optimization basin (Entropy only)."""
        plt.figure(figsize=(8, 4))
        plt.plot(grid, scores, 'b.-', label='Entropy Score')
        plt.axvline(winner, color='r', linestyle='--', label=f'Min: {winner:.2f}')
        plt.title(f"Entropy Landscape (Q{q_idx+1})")
        plt.xlabel("Rotation Angle (deg)")
        plt.ylabel("Shannon Entropy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_entropy_gini_landscape(self, subset, grid, entropy_scores, winner, q_idx):
        """
        Visualizes optimization basin with BOTH Entropy (Minimization) and Gini (Maximization).
        Uses secondary Y-axis for Gini.
        """
        gini_scores = []
        for angle in grid:
            dens = self._get_projected_density(subset, angle)
            gini_scores.append(self.calculate_gini(dens))
            
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 1. Entropy Trace (Left Axis)
        color_ent = 'tab:blue'
        ax1.set_xlabel('Rotation Angle (deg)')
        ax1.set_ylabel('Shannon Entropy (Minimize)', color=color_ent, fontweight='bold')
        line1 = ax1.plot(grid, entropy_scores, color=color_ent, marker='o', markersize=4, linestyle='-', label='Entropy')
        ax1.tick_params(axis='y', labelcolor=color_ent)
        
        # Mark the winner (Min Entropy)
        line3 = ax1.axvline(winner, color='r', linestyle='--', alpha=0.8, label=f'Coarse Opt: {winner:.2f}deg')
        
        # 2. Gini Trace (Right Axis)
        ax2 = ax1.twinx() 
        color_gini = 'tab:green'
        ax2.set_ylabel('Gini Coefficient (Maximize)', color=color_gini, fontweight='bold')
        line2 = ax2.plot(grid, gini_scores, color=color_gini, marker='x', markersize=4, linestyle=':', label='Gini')
        ax2.tick_params(axis='y', labelcolor=color_gini)
        
        # Combined Legend
        lines = line1 + line2 + [line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        plt.title(f"Optimization Landscape (Q{q_idx+1})", y=1.15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _calculate_landscape(self, subset, step):
        """Helper to compute 360 landscape metrics for a specific subset."""
        angles = np.arange(0, 360, step)
        ent_scores = []
        gini_scores = []
        
        for ang in angles:
            dens = self._get_projected_density(subset, ang)
            ent_scores.append(entropy(dens))
            gini_scores.append(self.calculate_gini(dens))
            
        return angles, np.array(ent_scores), np.array(gini_scores)

    def plot_360_landscape(self, q_index=None, step=1.0):
        """
        Scans 0-360 degrees and plots Entropy and Gini profiles.
        If q_index is None, generates a 2x2 plot for all quartiles.
        Includes dashed lines for BBox orientation and its orthogonals.
        """
        # 1. Sort Data to define Quartiles
        x_curr = self.points[:, 0]
        sort_idx = np.argsort(x_curr)
        sorted_points = self.points[sort_idx]
        n = len(sorted_points)
        q_len = n // 4

        def get_subset(idx):
            start = idx * q_len
            end = (idx + 1) * q_len if idx < 3 else n
            return sorted_points[start:end]

        # Helper plotting logic
        def plot_on_ax(ax, angles, ent, gini, title):
            # Entropy
            color_ent = 'tab:blue'
            ax.set_xlabel('Angle (deg)')
            ax.set_ylabel('Entropy', color=color_ent, fontweight='bold')
            ax.plot(angles, ent, color=color_ent, linewidth=1.5)
            ax.tick_params(axis='y', labelcolor=color_ent)
            
            # Gini
            ax2 = ax.twinx()
            color_gini = 'tab:green'
            ax2.set_ylabel('Gini', color=color_gini, fontweight='bold')
            ax2.plot(angles, gini, color=color_gini, linestyle='--', linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor=color_gini)
            
            # Mark Optima (Data-driven)
            min_ent = np.argmin(ent)
            max_gini = np.argmax(gini)
            ax.axvline(angles[min_ent], color=color_ent, linestyle=':', alpha=0.6)
            ax2.axvline(angles[max_gini], color=color_gini, linestyle=':', alpha=0.6)
            
            # Mark BBox Orientation (Geometric)
            if hasattr(self, 'bbox_rotation_angle') and self.bbox_rotation_angle is not None:
                # Base angle normalized to 0-360
                base_angle = self.bbox_rotation_angle % 360
                ortho_angles = [(base_angle + i * 90) % 360 for i in range(4)]
                
                for i, ang in enumerate(ortho_angles):
                    label = 'BBox' if i == 0 else None # Label only once
                    ax.axvline(ang, color='red', linestyle='-.', linewidth=2, alpha=0.75, label=label)
            
            ax.set_title(f"{title}\nBest: {angles[min_ent]:.1f}deg(E) / {angles[max_gini]:.1f}deg(G)", fontsize=10)
            ax.grid(True, alpha=0.3)

        # --- CASE A: Single Quartile ---
        if q_index is not None:
            if q_index < 0 or q_index > 3:
                print("Invalid Quartile Index. Use 0-3.")
                return
                
            print(f"[GridOptimizer] Scanning Q{q_index+1} (0-360)...")
            subset = get_subset(q_index)
            angles, ent, gini = self._calculate_landscape(subset, step)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_on_ax(ax, angles, ent, gini, f"Landscape Q{q_index+1}")
            plt.tight_layout()
            plt.show()
            
        # --- CASE B: 2x2 Grid (All Quartiles) ---
        else:
            print("[GridOptimizer] Scanning All Quartiles (0-360)...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            for i in range(4):
                subset = get_subset(i)
                angles, ent, gini = self._calculate_landscape(subset, step)
                plot_on_ax(axes[i], angles, ent, gini, f"Quartile Q{i+1}")
            
            plt.suptitle("Full 360deg Grid Alignment Landscape (All Quartiles)", y=1.02)
            plt.tight_layout()
            plt.show()

"""
```
"""