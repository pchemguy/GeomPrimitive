"""
```
pet_grid_optimizer.py
---------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.optimize import minimize_scalar


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
        
        # 2. Auto-calculate bandwidth if not provided
        if bw is None:
            from sklearn.neighbors import NearestNeighbors
            # Safe check for small datasets
            k = min(len(points), 2)
            if k < 2:
                self.bw = 1.0
            else:
                nbrs = NearestNeighbors(n_neighbors=k).fit(points)
                dists, _ = nbrs.kneighbors(points)
                pitch = np.percentile(dists[:, 1], 50) # Median spacing
                self.bw = pitch * 0.15 # 15% of pitch is the "Golden Rule"
        else:
            self.bw = bw
            
        print(f"[GridOptimizer] Init. N={len(points)}, BW={self.bw:.2f}")

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
            optimal_angle, metric_value
        """
        # 1. Split Data
        x_curr = self.points[:, 0]
        sort_idx = np.argsort(x_curr)
        sorted_points = self.points[sort_idx]
        
        n = len(sorted_points)
        q_len = n // 4
        if q_len < 2:
            print(f"[Error] Quartile {q_index} has too few points ({q_len}).")
            return initial_angle, 0.0
            
        start = q_index * q_len
        end = (q_index + 1) * q_len
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
            self.plot_entropy_landscape(subset, coarse_grid, scores, best_coarse_angle, q_index)

        # 3. Fine Optimization
        # Search +/- 1.0 deg around the coarse winner
        try:
            res = minimize_scalar(
                self._objective_entropy, 
                args=(subset,),
                bounds=(best_coarse_angle - 1.0, best_coarse_angle + 1.0),
                method='bounded'
            )
            
            if res.success:
                print(f"[Q{q_index+1}] Success. Angle: {res.x:.4f} (Entropy: {res.fun:.4f})")
                return res.x, res.fun
            else:
                print(f"[Q{q_index+1}] Opt Failed. Using Coarse: {best_coarse_angle:.4f}")
                return best_coarse_angle, best_coarse_score
                
        except Exception as e:
            print(f"[Q{q_index+1}] Exception in minimize_scalar: {e}")
            return best_coarse_angle, best_coarse_score

    def plot_entropy_landscape(self, subset, grid, scores, winner, q_idx):
        """Visualizes the optimization basin."""
        plt.figure(figsize=(8, 4))
        plt.plot(grid, scores, 'b.-', label='Entropy Score')
        plt.axvline(winner, color='r', linestyle='--', label=f'Min: {winner:.2f}')
        plt.title(f"Entropy Landscape (Q{q_idx+1})")
        plt.xlabel("Rotation Angle (deg)")
        plt.ylabel("Shannon Entropy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


"""
```
"""