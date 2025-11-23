"""
```
pet_grid_optimizer.py
---------------------

IMPORTANT: KDE Std Dev optimization presently is broken (it yields results, which appear to be wrong).
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
        
        # 2. Auto-calculate pitch and default bandwidth (0.1*P)
        self.pitch, self.bw = self._estimate_robust_pitch_bw(points)
        # self.bw = self.pitch / 20.0
        
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
            self.center = np.mean(points, axis=0)
        else:
            self.bbox = bbox
            
            # 3. Reject Outliers (Noise points outside the determined BBox)
            self.points = self.reject_outliers(points, bbox, labels)
            
            # 4. Determine Rotation Angle from BBox (Angle of bottom edge)
            p0, p1 = bbox[0], bbox[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            self.bbox_rotation_angle = np.degrees(np.arctan2(dy, dx))

            # 5. Set Pivot to BBox Geometric Center (Crucial for stable projection)
            self.center = np.mean(self.bbox, axis=0)
            
            print(f"[GridOptimizer] BBox Found. Rotation: {self.bbox_rotation_angle:.2f} deg")

        # 5. Update Geometry based on clean points
        self.x_range = (np.min(self.points[:, 0]), np.max(self.points[:, 0]))
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
        
        # 2. Auto-calculate pitch and default bandwidth (0.1*P)
        self.pitch, self.bw = self._estimate_robust_pitch_bw(points)
        # self.bw = self.pitch / 20.0
        
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
            self.center = np.mean(points, axis=0)
        else:
            self.bbox = bbox
            
            # 3. Reject Outliers (Noise points outside the determined BBox)
            self.points = self.reject_outliers(points, bbox, labels)
            
            # 4. Determine Rotation Angle from BBox (Angle of bottom edge)
            p0, p1 = bbox[0], bbox[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            self.bbox_rotation_angle = np.degrees(np.arctan2(dy, dx))

            # 5. Set Pivot to BBox Geometric Center (Crucial for stable projection)
            self.center = np.mean(self.bbox, axis=0)
            
            print(f"[GridOptimizer] BBox Found. Rotation: {self.bbox_rotation_angle:.2f} deg")

        # 5. Update Geometry based on clean points
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
        bw = ref_pitch * 0.05
        
        print(f"[GridOptimizer] Auto-BW: {bw:.2f} px (Pitch ~{ref_pitch:.2f} px)")
        return ref_pitch, bw

    def _get_quartile_subset(self, q_index):
        # Sort along the BBox Main Axis to ensure consistent quartiles regardless of rotation
        theta = np.radians(self.bbox_rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        # Project to BBox Space (X-axis)
        x_proj = self.points[:, 0] * c + self.points[:, 1] * s
        
        sort_idx = np.argsort(x_proj)
        sorted_points = self.points[sort_idx]
        
        n = len(sorted_points)
        q_len = n // 4
        start = q_index * q_len
        end = (q_index + 1) * q_len if q_index < 3 else n
        return sorted_points[start:end]

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

    def _get_projected_density(self, points_subset, angle, grid=None, normalize='pdf'):
        """
        Returns density profile.
        Args:
            normalize: 
                True = Sums to 1 (Probability Mass for Entropy).
                'pdf' = Scaled by 1/(N*sigma*sqrt(2pi)) (Probability Density for StdDev).
                False = Raw Sum.
        """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        x_centered = (points_subset[:, 0] - self.center[0]) * c - \
                     (points_subset[:, 1] - self.center[1]) * s
        x_rot = x_centered + self.center[0]
        
        n_points = len(x_rot)
        if n_points == 0: return np.array([1.0])
        
        if grid is not None:
            diffs = grid[:, None] - x_rot[None, :]
            # Raw Gaussian sum
            pdfs = np.exp(-0.5 * (diffs / self.bw)**2)
            density = np.sum(pdfs, axis=1)
        else:
            x_min, x_max = np.min(x_rot), np.max(x_rot)
            span = x_max - x_min
            if span < 1e-6: span = 1.0
            # Match interactive tool padding (0.2)
            grid = np.linspace(x_min - span*0.2, x_max + span*0.2, 500)
            diffs = grid[:, None] - x_rot[None, :]
            pdfs = np.exp(-0.5 * (diffs / self.bw)**2)
            density = np.sum(pdfs, axis=1)
        
        # Apply Normalization
        if normalize is True:
            # Probability Mass (Sum = 1)
            total_mass = np.sum(density)
            if total_mass > 0: density /= total_mass
        elif normalize == 'pdf':
            # Probability Density (Integral ~= 1)
            # Formula: sum(exp) / (N * sigma * sqrt(2pi))
            factor = 1.0 / (n_points * self.bw * np.sqrt(2 * np.pi))
            density *= factor
            
        return density

    def _objective_entropy(self, angle, points_subset):
        """Objective Function: Shannon Entropy (Minimize this)"""
        prob_dist = self._get_projected_density(points_subset, angle)
        return entropy(prob_dist)

    def optimize_quartile(self, q_index, initial_angle=0.0, search_width=10.0, debug=False):
        """
        Optimizes rotation for a specific quartile.
        Uses _get_quartile_subset to correctly isolate the region relative to the BBox axis.
        """
        subset = self._get_quartile_subset(q_index)
        
        if len(subset) < 2:
            print(f"[Error] Quartile {q_index} has too few points.")
            return initial_angle, 0.0, 0.0
            
        search_min = initial_angle - search_width / 2.0
        search_max = initial_angle + search_width / 2.0
        coarse_grid = np.arange(search_min, search_max, 0.5)
        if len(coarse_grid) == 0: coarse_grid = np.array([initial_angle])
            
        scores = [self._objective_entropy(a, subset) for a in coarse_grid]
        best_coarse_idx = np.argmin(scores)
        best_coarse_angle = coarse_grid[best_coarse_idx]
        best_coarse_score = scores[best_coarse_idx]
        
        if debug:
            self.plot_entropy_gini_landscape(subset, coarse_grid, scores, best_coarse_angle, q_index)

        optimal_angle = best_coarse_angle
        final_entropy = best_coarse_score
        
        try:
            res = minimize_scalar(
                self._objective_entropy, args=(subset,),
                bounds=(best_coarse_angle - 1.0, best_coarse_angle + 1.0),
                method='bounded'
            )
            if res.success:
                optimal_angle = res.x
                final_entropy = res.fun
            else:
                print(f"[Q{q_index+1}] Opt Failed. Using Coarse: {best_coarse_angle:.4f}")
        except Exception as e:
            print(f"[Q{q_index+1}] Exception: {e}")

        final_density = self._get_projected_density(subset, optimal_angle)
        final_gini = self.calculate_gini(final_density)
        
        print(f"[Q{q_index+1}] Result: {optimal_angle:.4f} deg (Ent: {final_entropy:.2f}, Gini: {final_gini:.2f})")
        return optimal_angle, final_entropy, final_gini

    def optimize_quartile_std(self, q_index, initial_angle=0.0, search_width=10.0, debug=False):
        """
        Optimizes rotation for a specific quartile by MAXIMIZING Density Standard Deviation.
        Uses 'pdf' normalization to match interactive tool.
        """
        subset = self._get_quartile_subset(q_index)
        if len(subset) < 2: return initial_angle, 0.0, 0.0
            
        search_min = initial_angle - search_width / 2.0
        search_max = initial_angle + search_width / 2.0
        coarse_grid = np.arange(search_min, search_max, 0.5)
        if len(coarse_grid) == 0: coarse_grid = np.array([initial_angle])
            
        # Use PDF density for Std Dev optimization
        scores = [self._objective_std_neg(a, subset) for a in coarse_grid]
        best_coarse_idx = np.argmin(scores)
        best_coarse_angle = coarse_grid[best_coarse_idx]
        best_coarse_score = scores[best_coarse_idx]
        
        if debug:
            self.plot_entropy_std_landscape(subset, coarse_grid, scores, best_coarse_angle, q_index)

        optimal_angle = best_coarse_angle
        final_neg_std = best_coarse_score
        
        try:
            res = minimize_scalar(
                self._objective_std_neg, 
                args=(subset,),
                bounds=(best_coarse_angle - 1.0, best_coarse_angle + 1.0),
                method='bounded'
            )
            if res.success:
                optimal_angle = res.x
                final_neg_std = res.fun
        except Exception:
            pass

        # For reporting Gini, we can use normalized or pdf, Gini is scale invariant
        final_density = self._get_projected_density(subset, optimal_angle, normalize='pdf')
        final_gini = self.calculate_gini(final_density)
        final_std = -final_neg_std
        
        print(f"[Q{q_index+1}] Result: {optimal_angle:.4f} deg (Std: {final_std:.4f}, Gini: {final_gini:.2f})")
        return optimal_angle, final_std, final_gini
    
    def _objective_gini_neg(self, angle, points_subset):
        """Objective: Maximize Gini (Minimize Negative Gini)"""
        dens = self._get_projected_density(points_subset, angle)
        return -self.calculate_gini(dens)

    def _objective_std_neg(self, angle, points_subset):
        """Objective: Maximize Std Dev of PDF (Minimize Negative)"""
        # Matches interactive tool: std of the PDF values
        dens = self._get_projected_density(points_subset, angle, normalize='pdf')
        return -np.std(dens)
            
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

    def analyze_spatial_profile_std(self, angle_center=0.0, sweep_width=10.0, num_slices=10, angle_steps=20):
        """
        Performs a spatial focus sweep using Standard Deviation of Density as the sharpness metric.
        Normalizes each slice's response to 0-1 to compare relative sharpness across spatially varying densities.
        Plots the trajectory of the 'Focus Plane' across the grid.
        """
        print(f"[GridOptimizer] Analyzing Spatial Profile (Std Dev)...")

        # 1. Sort Data Spatially (along BBox axis)
        theta_bbox = np.radians(self.bbox_rotation_angle)
        c_b, s_b = np.cos(theta_bbox), np.sin(theta_bbox)
        x_proj = self.points[:, 0] * c_b + self.points[:, 1] * s_b
        
        sort_idx = np.argsort(x_proj)
        sorted_points = self.points[sort_idx]
        sorted_x_proj = x_proj[sort_idx] 

        n = len(sorted_points)
        slice_len = n // num_slices

        # Setup Angles
        angles = np.linspace(angle_center - sweep_width/2, angle_center + sweep_width/2, angle_steps)
        
        # Storage: [num_slices, num_angles]
        std_matrix = np.zeros((num_slices, angle_steps))
        slice_centers = []

        # 2. Compute Std Dev Matrix
        for i in range(num_slices):
            start = i * slice_len
            end = (i + 1) * slice_len if i < num_slices - 1 else n
            subset = sorted_points[start:end]
            
            # Calculate Slice Center (in projected coordinates)
            if len(subset) > 0:
                center_x = np.mean(sorted_x_proj[start:end])
            else:
                center_x = 0
            slice_centers.append(center_x)

            if len(subset) < 2:
                continue

            for j, ang in enumerate(angles):
                dens = self._get_projected_density(subset, ang)
                # Standard Deviation is a good proxy for contrast/sharpness
                std_val = np.std(dens)
                std_matrix[i, j] = std_val

        slice_centers = np.array(slice_centers)

        # 3. Normalize per Slice (Row-wise) to 0..1
        # This is crucial: it allows us to compare the "best angle" for a sparse slice
        # vs a dense slice on equal footing.
        row_mins = np.min(std_matrix, axis=1)[:, None]
        row_maxs = np.max(std_matrix, axis=1)[:, None]
        ranges = row_maxs - row_mins
        # Avoid division by zero for empty slices
        ranges[ranges == 0] = 1.0 
        
        norm_matrix = (std_matrix - row_mins) / ranges

        # 4. Find Focus Profile (Ridge Detection)
        # For each angle (column), finding the slice (row) with the highest normalized score
        # is one way, BUT for Twist Analysis, we usually want:
        # "For each Angle, where is the focus?" -> argmax over rows (Y-axis of plot)
        # This gives us X_location(Angle).
        best_slice_indices = np.argmax(norm_matrix, axis=0)
        focus_x_locs = slice_centers[best_slice_indices]

        # 5. Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Heatmap Background
        # We use pcolormesh to correctly align the grid cells to axes
        X_mesh, Y_mesh = np.meshgrid(angles, slice_centers)
        # Note: pcolormesh expects X/Y to define corners or centers. 
        # For simple alignment with data shape (N, M), we can pass the centers and use shading='auto' or 'nearest'
        c = ax.pcolormesh(X_mesh, Y_mesh, norm_matrix, shading='nearest', cmap='viridis', alpha=0.5)
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('Normalized Sharpness (0-1)')
        
        # Focus Trajectory Line
        ax.plot(angles, focus_x_locs, 'o-', color='red', linewidth=2, label='Max Sharpness Plane')
        
        # Trend Line
        if len(angles) > 1:
            z = np.polyfit(angles, focus_x_locs, 1)
            p = np.poly1d(z)
            ax.plot(angles, p(angles), 'b:', linewidth=2, label=f'Twist Rate: {z[0]:.1f} px/deg')

        ax.set_xlabel('Rotation Angle (deg)')
        ax.set_ylabel('Grid Position (Projected X)')
        ax.set_title(f"Spatial Focus Profile (Std Dev)\nSweep: {angle_center}deg +/- {sweep_width/2}deg")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        plt.show()

        return angles, focus_x_locs, norm_matrix

    def kde_table(self, angle_center=0.0, sweep_width=10.0, angle_steps=20, filename="grid_kde_sweep.csv"):
        """
        Generates a CSV with KDE profiles in Columns.
        Row 1: Angles
        Column 1: Position (X coordinates)
        Columns 2..N: Density values for each angle
        """
        print(f"[GridOptimizer] Generating KDE Table ({angle_steps} steps) -> {filename}")
        
        angles = np.linspace(angle_center - sweep_width/2, angle_center + sweep_width/2, angle_steps)
        
        # Determine Fixed Spatial Grid based on center angle
        theta = np.radians(angle_center)
        c, s = np.cos(theta), np.sin(theta)
        x_centered = (self.points[:, 0] - self.center[0]) * c - \
                     (self.points[:, 1] - self.center[1]) * s
        x_rot = x_centered + self.center[0]
        
        x_min, x_max = np.min(x_rot), np.max(x_rot)
        span = x_max - x_min
        if span < 1e-6: span = 1.0
        
        # Add 20% padding for rotation variance
        grid_min = x_min - span * 0.2
        grid_max = x_max + span * 0.2
        grid_x = np.linspace(grid_min, grid_max, 500)
        
        # Container for all density profiles
        # Shape: (num_grid_points, num_angles)
        all_profiles = np.zeros((len(grid_x), len(angles)))

        for i, ang in enumerate(angles):
            # Calculate Density on FIXED grid using FULL dataset
            dens = self._get_projected_density(self.points, ang, grid=grid_x)
            all_profiles[:, i] = dens
        
        try:
            with open(filename, "w") as f:
                # Header Row: "Position" followed by Angle values
                header = ["Position"] + [f"{a:.4f}" for a in angles]
                f.write(",".join(header) + "\n")
                
                # Data Rows: grid_x value followed by density for each angle
                for i in range(len(grid_x)):
                    row_vals = [f"{grid_x[i]:.2f}"] + [f"{val:.6f}" for val in all_profiles[i, :]]
                    f.write(",".join(row_vals) + "\n")
                    
            print(f"[GridOptimizer] Done.")
        except IOError as e:
            print(f"[Error] Could not write file: {e}")

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

    def plot_entropy_std_landscape(self, subset, grid, optimization_scores, winner, q_idx):
        """
        Plots KDE at optimal angle AND Optimization Landscape.
        
        Args:
            optimization_scores: The scores computed during coarse sweep (Negative Std Dev)
        """
        # 1. Re-calculate Metrics for Plotting
        # We recalculate to ensure we display Positive Std Dev and Normalized Entropy
        std_scores = []
        ent_scores = []
        for angle in grid:
            dens_pdf = self._get_projected_density(subset, angle, normalize='pdf')
            dens_prob = self._get_projected_density(subset, angle, normalize=True)
            std_scores.append(np.std(dens_pdf))
            ent_scores.append(entropy(dens_prob))
            
        # 2. Calculate KDE Profile for Winner
        best_dens = self._get_projected_density(subset, winner, normalize='pdf')
        
        # --- Create Plot with 2 Subplots ---
        fig, (ax_kde, ax_land) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: KDE Profile
        theta = np.radians(winner)
        c, s = np.cos(theta), np.sin(theta)
        x_rot = (subset[:, 0] - self.center[0]) * c - \
                (subset[:, 1] - self.center[1]) * s + self.center[0]
        x_min, x_max = np.min(x_rot), np.max(x_rot)
        span = x_max - x_min
        if span < 1e-6: span = 1.0
        grid_x = np.linspace(x_min - span*0.2, x_max + span*0.2, 500) # Match internal grid padding
        
        # Recalculate on plotting grid for visualization consistency
        diffs = grid_x[:, None] - x_rot[None, :]
        pdfs = np.exp(-0.5 * (diffs / self.bw)**2)
        n_points = len(subset)
        factor = 1.0 / (n_points * self.bw * np.sqrt(2 * np.pi))
        dens_plot = np.sum(pdfs, axis=1) * factor
        
        ax_kde.plot(grid_x, dens_plot, 'k-', linewidth=1.5)
        ax_kde.fill_between(grid_x, dens_plot, color='orange', alpha=0.3)
        ax_kde.set_title(f"Q{q_idx+1} KDE Profile @ {winner:.2f}deg\n(Std Dev: {np.std(best_dens):.4f})")
        ax_kde.set_xlabel("Projected Coordinate")
        ax_kde.set_ylabel("Density (PDF)")
        ax_kde.grid(True, alpha=0.3)
        
        # Plot 2: Landscape
        color_std = 'tab:orange'
        ax_land.set_xlabel('Rotation Angle (deg)')
        ax_land.set_ylabel('Std Dev (Maximize)', color=color_std, fontweight='bold')
        line1 = ax_land.plot(grid, std_scores, color=color_std, marker='o', markersize=4, label='Std Dev')
        ax_land.tick_params(axis='y', labelcolor=color_std)
        
        ax_land2 = ax_land.twinx()
        color_ent = 'tab:blue'
        ax_land2.set_ylabel('Entropy (Minimize)', color=color_ent, fontweight='bold')
        line2 = ax_land2.plot(grid, ent_scores, color=color_ent, marker='x', markersize=4, linestyle=':', label='Entropy')
        ax_land2.tick_params(axis='y', labelcolor=color_ent)
        
        line3 = ax_land.axvline(winner, color='r', linestyle='--', alpha=0.8, label=f'Opt: {winner:.2f}deg')
        
        lines = line1 + line2 + [line3]
        labels = [l.get_label() for l in lines]
        ax_land.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        ax_land.set_title(f"Optimization Landscape (Q{q_idx+1})", y=1.15)
        ax_land.grid(True, alpha=0.3)
        
        plt.tight_layout()
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

    def _calculate_landscape_std(self, subset, step):
        angles = np.arange(0, 360, step)
        ent_scores = []
        std_scores = []
        for ang in angles:
            dens = self._get_projected_density(subset, ang)
            ent_scores.append(entropy(dens))
            std_scores.append(np.std(dens))
        return angles, np.array(ent_scores), np.array(std_scores)

    def plot_360_landscape_std(self, q_index=None, step=1.0):
        """Scans 0-360 degrees and plots Entropy and Std Dev profiles."""
        def plot_on_ax(ax, angles, ent, std, title):
            color_ent = 'tab:blue'
            ax.set_xlabel('Angle (deg)')
            ax.set_ylabel('Entropy', color=color_ent, fontweight='bold')
            ax.plot(angles, ent, color=color_ent, linewidth=1.5)
            ax.tick_params(axis='y', labelcolor=color_ent)
            
            ax2 = ax.twinx()
            color_std = 'tab:orange'
            ax2.set_ylabel('Std Dev', color=color_std, fontweight='bold')
            ax2.plot(angles, std, color=color_std, linestyle='--', linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor=color_std)
            
            min_ent = np.argmin(ent)
            max_std = np.argmax(std)
            ax.axvline(angles[min_ent], color=color_ent, linestyle=':', alpha=0.6)
            ax2.axvline(angles[max_std], color=color_std, linestyle=':', alpha=0.6)
            
            if hasattr(self, 'bbox_rotation_angle') and self.bbox_rotation_angle is not None:
                base_angle = self.bbox_rotation_angle % 360
                ortho_angles = [(base_angle + i * 90) % 360 for i in range(4)]
                for i, ang in enumerate(ortho_angles):
                    label = 'BBox' if i == 0 else None
                    ax.axvline(ang, color='green', linestyle='-.', linewidth=2, alpha=0.75, label=label)
            
            ax.set_title(f"{title}\nBest: {angles[min_ent]:.1f}deg(E) / {angles[max_std]:.1f}deg(S)", fontsize=10)
            ax.grid(True, alpha=0.3)

        if q_index is not None:
            print(f"[GridOptimizer] Scanning Q{q_index+1} (0-360, Std)...")
            subset = self._get_quartile_subset(q_index)
            angles, ent, std = self._calculate_landscape_std(subset, step)
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_on_ax(ax, angles, ent, std, f"Landscape Q{q_index+1} (Std)")
            plt.tight_layout()
            plt.show()
        else:
            print("[GridOptimizer] Scanning All Quartiles (0-360, Std)...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            for i in range(4):
                subset = self._get_quartile_subset(i)
                angles, ent, std = self._calculate_landscape_std(subset, step)
                plot_on_ax(axes[i], angles, ent, std, f"Quartile Q{i+1}")
            plt.suptitle("Full 360deg Grid Alignment Landscape (Entropy vs Std Dev)", y=1.02)
            plt.tight_layout()
            plt.show()

    def plot_quartile_optimization_report(self, initial_angle=None, search_width=20.0, bbox_aux_angle=False, plot=True):
        """
        Runs optimization for each quartile.
        Plots: Column 1: Full KDE (Black). NO GREEN FILL.
               Column 2: Optimization Landscape.
        """
        if initial_angle is None:
            initial_angle = self.bbox_rotation_angle
            if bbox_aux_angle:
                # Correct Orthogonal Logic: Add 90, normalize
                initial_angle = (initial_angle + 90 + 90) % 180 - 90

        results = {}
        
        if plot:
            print(f"[GridOptimizer] Generating Report (Center: {initial_angle:.2f}deg, Width: {search_width}deg)")
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        search_min = initial_angle - search_width/2
        search_max = initial_angle + search_width/2
        
        for i in range(4):
            # 1. Get Subset (For optimization calculations ONLY)
            subset = self._get_quartile_subset(i)
            
            # 2. Optimize
            best_angle, final_ent, final_gini = self.optimize_quartile(i, initial_angle, search_width)
            
            key = f"Q{i+1}"
            results[key] = {"angle": float(best_angle), "entropy": float(final_ent), "gini": float(final_gini)}
            
            if plot:
                # --- Plot Column 1: Global KDE at Best Angle ---
                ax_kde = axes[i, 0]
                
                theta = np.radians(best_angle)
                c, s = np.cos(theta), np.sin(theta)
                
                # Project Full Set (Screen Space)
                x_centered_full = (self.points[:, 0] - self.center[0]) * c - \
                                  (self.points[:, 1] - self.center[1]) * s
                x_rot_full = x_centered_full + self.center[0]
                
                # Shared Grid based on FULL range
                x_min, x_max = np.min(x_rot_full), np.max(x_rot_full)
                span = x_max - x_min
                if span < 1e-6: span = 1.0
                grid_x = np.linspace(x_min - span*0.1, x_max + span*0.1, 500)
                
                # Calculate Full Density (Normalized Mass)
                diffs_full = grid_x[:, None] - x_rot_full[None, :]
                pdfs_full = np.exp(-0.5 * (diffs_full / self.bw)**2)
                raw_dens_full = np.sum(pdfs_full, axis=1)
                
                total_mass = np.sum(raw_dens_full)
                plot_dens_full = raw_dens_full / total_mass if total_mass > 0 else raw_dens_full
                
                # PLOT FULL GRID ONLY (NO SUBSET FILL)
                ax_kde.plot(grid_x, plot_dens_full, 'k-', linewidth=1.5, label='Full Grid')
                
                ax_kde.set_title(f"Q{i+1} Optimal: {best_angle:.2f}deg\n(Global KDE Profile)", fontsize=10, fontweight='bold')
                if i == 3: ax_kde.set_xlabel("Projected Spatial Coordinate (pixels)")
                ax_kde.set_ylabel("Density")
                ax_kde.grid(True, alpha=0.3)
                
                # --- Plot Column 2: Landscape ---
                ax_land = axes[i, 1]
                coarse_grid = np.arange(search_min, search_max, 0.5)
                
                ent_scores = []
                gini_scores = []
                for ang in coarse_grid:
                    dens = self._get_projected_density(subset, ang)
                    ent_scores.append(entropy(dens))
                    gini_scores.append(self.calculate_gini(dens))
                
                color_ent = 'tab:blue'
                if i == 3: ax_land.set_xlabel('Angle (deg)')
                ax_land.set_ylabel('Entropy', color=color_ent, fontweight='bold')
                ax_land.plot(coarse_grid, ent_scores, 'o-', color=color_ent, markersize=3, label='Entropy')
                ax_land.tick_params(axis='y', labelcolor=color_ent)
                
                ax_land2 = ax_land.twinx()
                color_gini = 'tab:green'
                ax_land2.set_ylabel('Gini', color=color_gini, fontweight='bold')
                ax_land2.plot(coarse_grid, gini_scores, 'x--', color=color_gini, markersize=4, label='Gini')
                ax_land2.tick_params(axis='y', labelcolor=color_gini)
                
                ax_land.axvline(best_angle, color='red', linestyle='--', alpha=0.8, label=f'Opt: {best_angle:.2f}deg')
                
                lines1, labels1 = ax_land.get_legend_handles_labels()
                lines2, labels2 = ax_land2.get_legend_handles_labels()
                ax_land.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=3, fontsize='small')
                
                ax_land.set_title(f"Optimization (Q{i+1})", fontsize=10)
                ax_land.grid(True, alpha=0.3)
            
        if plot:
            plt.suptitle(f"Quartile Optimization Report\nSearch: {initial_angle:.2f}deg +/- {search_width/2:.1f}deg", fontsize=14, y=0.92)
            plt.show()
            
        return results

"""
```
"""