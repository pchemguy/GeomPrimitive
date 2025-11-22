"""
```
pet_grid_solver_extended.py
---------------------------
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects


# ==============================================================================
#  PART 1: CORE SOLVER (From previous iteration)
# ==============================================================================

def analyze_grid_centers(centers, optimize_axis='x', search_angle=0.0):
    """
    Robustly determines grid period/rotation from a set of points.
    Returns dict with 'period', 'angle', 'rms_error', 'confidence'.
    """
    if len(centers) < 10:
        return {'error': 'Insufficient points', 'confidence': 'FAIL', 'period': 0, 'rms_error': 999}

    # 1. OPTIMAL ROTATION
    best_angle = _get_optimal_angle(centers, optimize_axis, search_angle)
    
    # 2. PROJECTION
    theta = np.radians(best_angle)
    c, s = np.cos(theta), np.sin(theta)
    
    if optimize_axis == 'x':
        # Project onto X (Vertical lines) -> x' = x*c - y*s
        proj = centers[:, 0] * c - centers[:, 1] * s
    else:
        # Project onto Y (Horizontal lines) -> y' = x*s + y*c
        proj = centers[:, 0] * s + centers[:, 1] * c
        
    # 3. COARSE PERIOD (Global Autocorrelation)
    p_min, p_max = proj.min(), proj.max()
    bins = int(p_max - p_min)
    if bins < 10: 
        return {'error': 'Data span too small', 'confidence': 'FAIL', 'period': 0, 'rms_error': 999}
    
    hist, _ = np.histogram(proj, bins=bins)
    corr = correlate(hist, hist, mode='full')
    lags = np.arange(len(corr)) - (len(corr)//2)
    
    mask = (lags > 10) & (lags < len(lags)//2)
    valid_lags = lags[mask]
    valid_corr = corr[mask]
    
    if len(valid_lags) == 0:
        return {'period': 0, 'confidence': 'FAIL', 'rms_error': 999}

    peaks, _ = find_peaks(valid_corr, prominence=np.max(valid_corr)*0.1)
    
    if len(peaks) == 0:
        coarse_period = valid_lags[np.argmax(valid_corr)]
    else:
        sorted_peaks = peaks[np.argsort(valid_corr[peaks])[::-1]]
        best_peak_idx = sorted_peaks[0]
        coarse_period = valid_lags[best_peak_idx]

    # 4. FINE REFINEMENT
    def phase_variance(p):
        if p <= 0: return 1e9
        phase = proj % p
        phase_centered = (phase - np.median(phase))
        phase_centered[phase_centered > p/2] -= p
        phase_centered[phase_centered < -p/2] += p
        return np.std(phase_centered)

    # Bounds check to prevent crash if coarse period is garbage
    if coarse_period < 1: coarse_period = 10
    
    res = minimize_scalar(
        phase_variance, 
        bounds=(coarse_period * 0.9, coarse_period * 1.1), 
        method='bounded'
    )
    fine_period = res.x
    final_rms = res.fun
    
    confidence = "LOW"
    if final_rms < 1.0: confidence = "HIGH"
    elif final_rms < 2.0: confidence = "MODERATE"

    return {
        'period': float(round(fine_period, 4)),
        'angle': float(round(best_angle, 4)),
        'rms_error': float(round(final_rms, 4)),
        'confidence': confidence,
        'coarse_guess': float(coarse_period),
        'point_count': len(centers)
    }

def _get_optimal_angle(centers, axis, ref_angle):
    pts = centers - centers.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    
    def objective(deg):
        theta = np.radians(deg)
        c, s = np.cos(theta), np.sin(theta)
        if axis == 'x': p = x*c - y*s
        else: p = x*s + y*c
        rng = p.max() - p.min()
        if rng == 0: return 0
        bins = int(rng * 2)
        counts, _ = np.histogram(p, bins=max(10, bins))
        return -np.sum(counts**2)

    res = minimize_scalar(objective, bounds=(ref_angle - 10, ref_angle + 10), method='bounded')
    return res.x


# ==============================================================================
#  PART 2: HIERARCHICAL MANAGER
# ==============================================================================

class GridHierarchicalSolver:
    def __init__(self, centers):
        """
        centers: (N, 2) numpy array
        """
        self.centers = centers
        self.x_bounds = (centers[:, 0].min(), centers[:, 0].max())
        self.y_bounds = (centers[:, 1].min(), centers[:, 1].max())
        
    def determine_robust_limits(self, optimize_axis, min_points_per_cell=25, min_periods_per_cell=2.5):
        """
        Determines the maximum safe split count for parallel and perpendicular axes.
        
        Args:
            optimize_axis: 'x' or 'y'. The direction we are measuring period in.
            min_points_per_cell: Statistical floor (autocorrelation needs data).
            min_periods_per_cell: Geometric floor (we need roughly 2-3 grid lines to see a period).
        
        Returns:
            (max_sol_splits, max_ortho_splits)
        """
        # 1. Estimate Global Period first to know geometric limits
        global_res = analyze_grid_centers(self.centers, optimize_axis=optimize_axis)
        est_period = global_res.get('period', 50) # Default to 50 if fail
        if est_period == 0: est_period = 50
        
        total_points = len(self.centers)
        
        # --- DIMENSION MAPPING ---
        # If optim_axis is 'x', we measure along X.
        # 'Solution Direction' is X. 'Ortho Direction' is Y.
        if optimize_axis == 'x':
            span_sol = self.x_bounds[1] - self.x_bounds[0]
            span_ortho = self.y_bounds[1] - self.y_bounds[0]
        else:
            span_sol = self.y_bounds[1] - self.y_bounds[0]
            span_ortho = self.x_bounds[1] - self.x_bounds[0]

        # --- CONSTRAINT A: GEOMETRY ---
        # How many times can we split the Solution Axis before the cell is too narrow 
        # to contain 'min_periods'?
        # width / N > min_periods * period
        # N < width / (min_periods * period)
        geom_limit_sol = int(span_sol / (min_periods_per_cell * est_period))
        
        # Ortho axis has no geometric limit (a 1px tall slice is theoretically fine if it has points)
        geom_limit_ortho = 50 # Arbitrary high cap
        
        # --- CONSTRAINT B: DENSITY ---
        # Average points per cell = Total / (Nx * Ny).
        # This is coupled. We want to find independent maxima.
        
        # Let's simulate splitting along Sol axis ONLY, finding where min(cell_points) < limit
        density_limit_sol = self._scan_split_limit(optimize_axis, True, min_points_per_cell)
        
        # Simulate splitting along Ortho axis ONLY
        density_limit_ortho = self._scan_split_limit(optimize_axis, False, min_points_per_cell)
        
        # Final Maxima
        max_sol = min(geom_limit_sol, density_limit_sol)
        max_ortho = min(geom_limit_ortho, density_limit_ortho)
        
        return max(1, max_sol), max(1, max_ortho)

    def _scan_split_limit(self, optimize_axis, is_solution_axis, min_points):
        """Helper to test split limits in one direction while keeping the other at 1."""
        limit = 1
        # Map axis logic
        # If optimize 'x': Sol axis is splits in X (cols), Ortho is splits in Y (rows)
        # If optimize 'y': Sol axis is splits in Y (rows), Ortho is splits in X (cols)
        
        for n in range(1, 20): # Test up to 20 splits
            if optimize_axis == 'x':
                nx = n if is_solution_axis else 1
                ny = 1 if is_solution_axis else n
            else:
                nx = 1 if is_solution_axis else n
                ny = n if is_solution_axis else 1
                
            # Check smallest cell count
            cells = self._get_grid_cells(nx, ny)
            min_p = min([len(c) for c in cells])
            
            if min_p < min_points:
                break
            limit = n
            
        return limit

    def _get_grid_cells(self, nx, ny):
        """Returns a list of center-arrays for each cell in nx * ny grid"""
        x_edges = np.linspace(self.x_bounds[0], self.x_bounds[1], nx + 1)
        y_edges = np.linspace(self.y_bounds[0], self.y_bounds[1], ny + 1)
        
        cells = []
        for r in range(ny):
            for c in range(nx):
                # Standard bounds check
                mask = (
                    (self.centers[:, 0] >= x_edges[c]) & 
                    (self.centers[:, 0] < x_edges[c+1]) & 
                    (self.centers[:, 1] >= y_edges[r]) & 
                    (self.centers[:, 1] < y_edges[r+1])
                )
                cells.append(self.centers[mask])
        return cells

    def run_multiscale_analysis(self, optimize_axis='x', max_global_split=5):
        """
        Executes the "Square then Ortho-Extend" strategy.
        """
        # 1. Determine Limits
        # max_sol: max splits along the measuring axis (usually low, e.g., 3)
        # max_ortho: max splits perpendicular (usually high, e.g., 10)
        max_sol, max_ortho = self.determine_robust_limits(optimize_axis)
        
        # Cap at user request (global 5x5 limit logic)
        max_sol = min(max_sol, max_global_split)
        max_ortho = min(max_ortho, max_global_split) # Or allow higher if desired
        
        print(f"--- Auto-Detected Limits: SolAxis={max_sol}, OrthoAxis={max_ortho} ---")
        
        results_tree = {}
        
        # 2. Strategy: Equal Increase first
        # "Increase split count equally first up to the minimum of the two"
        limit_equal = min(max_sol, max_ortho)
        
        configurations = []
        
        # Phase 1: Square (1,1) -> (L, L)
        for k in range(1, limit_equal + 1):
            if optimize_axis == 'x':
                conf = (k, k) # (nx, ny)
            else:
                conf = (k, k)
            configurations.append(conf)
            
        # Phase 2: Extend Orthogonal
        # "Keep increasing the other split count until its maximum"
        # If optimize 'x', Sol is X (cols). We keep increasing Y (rows).
        # If optimize 'y', Sol is Y (rows). We keep increasing X (cols).
        
        if max_ortho > limit_equal:
            for k in range(limit_equal + 1, max_ortho + 1):
                if optimize_axis == 'x':
                    # Sol=X (fixed at limit), Ortho=Y (increasing)
                    conf = (limit_equal, k)
                else:
                    # Sol=Y (fixed at limit), Ortho=X (increasing)
                    conf = (k, limit_equal)
                configurations.append(conf)

        # 3. Execution Loop
        for (nx, ny) in configurations:
            print(f"\nProcessing Grid Configuration: {nx}x{ny} ...")
            
            # Generate bounds
            x_edges = np.linspace(self.x_bounds[0], self.x_bounds[1], nx + 1)
            y_edges = np.linspace(self.y_bounds[0], self.y_bounds[1], ny + 1)
            
            layer_key = f"{nx}x{ny}"
            results_tree[layer_key] = []
            
            # Iterate Rows (Y) then Cols (X)
            for r in range(ny):
                for c in range(nx):
                    # Slicing
                    mask = (
                        (self.centers[:, 0] >= x_edges[c]) & 
                        (self.centers[:, 0] < x_edges[c+1]) & 
                        (self.centers[:, 1] >= y_edges[r]) & 
                        (self.centers[:, 1] < y_edges[r+1])
                    )
                    cell_data = self.centers[mask]
                    
                    # Analysis
                    res = analyze_grid_centers(cell_data, optimize_axis=optimize_axis)
                    
                    # Tagging
                    res['grid_index'] = (r, c) # (row, col)
                    res['bounds_x'] = (int(x_edges[c]), int(x_edges[c+1]))
                    res['bounds_y'] = (int(y_edges[r]), int(y_edges[r+1]))
                    
                    results_tree[layer_key].append(res)
                    
                    # Optional: Print compact status
                    status = res.get('confidence', 'FAIL')
                    per = res.get('period', 0)
                    rms = res.get('rms_error', 999) # <--- Extract RMS
                    print(f"  Cell [{r:>2},{c:<2}] ({len(cell_data):>3} pts): {status:<8} | P={round(per, 1):>4.1f} px | RMS={round(rms, 2):<6.2f}")

        return results_tree


def plot_grid_analysis(results_tree, centers, title="Grid Analysis Report"):
    """
    Generates a robust dashboard of the hierarchical grid analysis.
    
    Args:
        results_tree: The dictionary returned by 'run_multiscale_analysis'.
        centers: The original (N,2) numpy array of points (for context).
        title: Title for the figure.
    """
    # 1. Setup
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Sort layer keys by "complexity" (total number of cells)
    layer_keys = sorted(results_tree.keys(), key=lambda k: len(results_tree[k]))
    finest_layer_key = layer_keys[-1]
    finest_data = results_tree[finest_layer_key]
    
    # Extract valid periods for scaling
    valid_periods = [c['period'] for c in finest_data if c['confidence'] != 'FAIL']
    if not valid_periods:
        print("No valid data to plot.")
        return
        
    # Robust Scaling (5th-95th percentile) to ignore extreme outliers in color mapping
    p_min, p_max = np.percentile(valid_periods, [5, 95])
    
    # --- SUBPLOT 1: Spatial Distortion Map (Period) ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(f"Spatial Period Map ({finest_layer_key})\nColor shows Grid Spacing (px)", fontsize=12)
    
    # Plot background points for context
    ax1.scatter(centers[:, 0], centers[:, 1], s=1, c='lightgray', alpha=0.5)
    
    # Setup Colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=p_min, vmax=p_max)
    
    for cell in finest_data:
        bx = cell['bounds_x']
        by = cell['bounds_y']
        w, h = bx[1] - bx[0], by[1] - by[0]
        
        if cell['confidence'] == 'FAIL':
            # Draw "X" box for failed regions
            rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='red', facecolor='none', hatch='xx', alpha=0.5)
            ax1.add_patch(rect)
        else:
            # Color box by Period
            color = cmap(norm(cell['period']))
            rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='white', facecolor=color, alpha=0.6)
            ax1.add_patch(rect)
            
            # Overlay text if cell is large enough
            if w > 100 and h > 100: 
                ax1.text(bx[0]+w/2, by[0]+h/2, f"{round(cell['period'], 1):.1f}", 
                         ha='center', va='center', fontsize=8, color='white', fontweight='bold', 
                         path_effects=[path_effects.withStroke(linewidth=2, foreground='black')] if path_effects else {})

    ax1.invert_yaxis() # Image coords
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, format='%.0f')
    cbar.set_label('Measured Period (px)')

    # --- SUBPLOT 2: Reliability Map (RMS Error) ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title(f"Reliability Map (RMS Error)\nDarker Blue = Better Fit", fontsize=12)
    ax2.scatter(centers[:, 0], centers[:, 1], s=1, c='lightgray', alpha=0.5)
    
    # RMS Colormap (White -> Blue -> Red)
    # We want 0.0 to be Good (Blue), >2.0 to be Bad (Red)
    
    for cell in finest_data:
        bx = cell['bounds_x']
        by = cell['bounds_y']
        w, h = bx[1] - bx[0], by[1] - by[0]
        
        rms = cell.get('rms_error', 999)
        
        if cell['confidence'] == 'FAIL' or rms > 2.0:
            color = 'red' # Bad
            alpha = 0.3
        elif rms < 0.5:
            color = 'darkgreen' # Perfect
            alpha = 0.4
        elif rms < 1.0:
            color = 'limegreen' # Good
            alpha = 0.4
        else:
            color = 'orange' # Moderate
            alpha = 0.4
            
        rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='gray', facecolor=color, alpha=alpha)
        ax2.add_patch(rect)
        
        if w > 100 and h > 100 and rms < 10:
             ax2.text(bx[0]+w/2, by[0]+h/2, f"{rms:.2f}", ha='center', va='center', fontsize=8)

    ax2.invert_yaxis()

    # --- SUBPLOT 3: Hierarchical Stability (Boxplot) ---
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.set_title("Convergence of Estimate across Split Configurations", fontsize=12)
    
    plot_data = []
    labels = []
    
    for key in layer_keys:
        # Filter only High Confidence results for the boxplot statistics
        periods = [c['period'] for c in results_tree[key] if c['confidence'] in ['HIGH', 'MODERATE']]
        if periods:
            plot_data.append(periods)
            labels.append(key)
    
    if plot_data:
        ax3.boxplot(plot_data, labels=labels, showmeans=True, patch_artist=True, 
                    boxprops=dict(facecolor="lightblue"))
        ax3.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax3.set_ylabel("Calculated Period (px)")
        ax3.set_xlabel("Grid Configuration (Rows x Cols)")
        
        # Add horizontal line for the global median of the finest layer
        global_median = np.median(valid_periods)
        ax3.axhline(global_median, color='red', linestyle='--', linewidth=1, label=f'Finest Median: {global_median:.3f}')
        ax3.legend()
    
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def save_grid_analysis_frames(results_tree, centers, output_dir="output"):
    """
    Generates and saves two sets of JPEG images for EACH split configuration.
    1. 'summary_RowxCol.jpg': The full dashboard.
    2. 'period_map_RowxCol.jpg': Just the period map with giant labels.
    """
    # 1. Setup Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Determine Global Color Limits
    all_valid_periods = []
    for key in results_tree:
        all_valid_periods.extend([c['period'] for c in results_tree[key] if c['confidence'] != 'FAIL'])
    
    if not all_valid_periods:
        print("No valid data found in any layer.")
        return

    # Robust Scaling
    p_min, p_max = np.percentile(all_valid_periods, [5, 95])
    
    # Sort keys by complexity
    layer_keys = sorted(results_tree.keys(), key=lambda k: len(results_tree[k]))

    print(f"Generating images for {len(layer_keys)} configurations in '{output_dir}'...")

    # 3. Loop through each split configuration
    for current_key in layer_keys:
        current_data = results_tree[current_key]
        
        # --- SERIES 1: SUMMARY DASHBOARD ---
        fig_summary = plt.figure(figsize=(20, 12))
        fig_summary.suptitle(f"Grid Analysis: Configuration {current_key}", fontsize=16, fontweight='bold')
        
        # ... SUBPLOT 1: Spatial Map (Period) ...
        ax1 = fig_summary.add_subplot(2, 2, 1)
        ax1.set_title(f"Spatial Period Map ({current_key})", fontsize=12)
        ax1.scatter(centers[:, 0], centers[:, 1], s=1, c='lightgray', alpha=0.5)
        
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=p_min, vmax=p_max)
        
        for cell in current_data:
            bx = cell['bounds_x']
            by = cell['bounds_y']
            w, h = bx[1] - bx[0], by[1] - by[0]
            
            if cell['confidence'] == 'FAIL':
                rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='red', facecolor='none', hatch='xx', alpha=0.5)
                ax1.add_patch(rect)
            else:
                color = cmap(norm(cell['period']))
                rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='white', facecolor=color, alpha=0.6)
                ax1.add_patch(rect)
                
                # STYLE UPDATE: Black text, White Box, Size 16
                if w > 80 and h > 80: 
                    ax1.text(bx[0]+w/2, by[0]+h/2, f"{cell['period']:.2f}", 
                             ha='center', va='center', fontsize=16, color='black', fontweight='bold',
                             bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

        ax1.invert_yaxis()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # FORMAT UPDATE: 0 Decimals for Colorbar
        cbar = plt.colorbar(sm, ax=ax1, format='%.0f')
        cbar.set_label('Measured Period (px)')

        # ... SUBPLOT 2: Reliability Map (RMS) ...
        ax2 = fig_summary.add_subplot(2, 2, 2)
        ax2.set_title(f"Reliability Map ({current_key})", fontsize=12)
        ax2.scatter(centers[:, 0], centers[:, 1], s=1, c='lightgray', alpha=0.5)
        
        for cell in current_data:
            bx = cell['bounds_x']
            by = cell['bounds_y']
            w, h = bx[1] - bx[0], by[1] - by[0]
            rms = cell.get('rms_error', 999)
            
            if cell['confidence'] == 'FAIL' or rms > 2.0:
                color = 'red'; alpha = 0.3
            elif rms < 0.5:
                color = 'darkgreen'; alpha = 0.4
            elif rms < 1.0:
                color = 'limegreen'; alpha = 0.4
            else:
                color = 'orange'; alpha = 0.4
                
            rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='gray', facecolor=color, alpha=alpha)
            ax2.add_patch(rect)
            
            # STYLE UPDATE: 1 Decimal, Size 16, Black on White Box
            if w > 80 and h > 80 and rms < 10:
                 ax2.text(bx[0]+w/2, by[0]+h/2, f"{rms:.1f}", 
                          ha='center', va='center', fontsize=16, color='black', fontweight='bold',
                          bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

        ax2.invert_yaxis()

        # ... SUBPLOT 3: Boxplot ...
        ax3 = fig_summary.add_subplot(2, 1, 2)
        ax3.set_title(f"Convergence History (Highlighting {current_key})", fontsize=12)
        
        plot_data = []
        labels = []
        colors = []
        
        for key in layer_keys:
            periods = [c['period'] for c in results_tree[key] if c['confidence'] in ['HIGH', 'MODERATE']]
            if periods:
                plot_data.append(periods)
                labels.append(key)
                if key == current_key:
                    colors.append("orange") 
                else:
                    colors.append("lightblue")
        
        if plot_data:
            bplot = ax3.boxplot(plot_data, labels=labels, showmeans=True, patch_artist=True)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                
            ax3.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax3.set_ylabel("Calculated Period (px)")
            ax3.set_xlabel("Grid Configuration")
            
            global_median = np.median(all_valid_periods)
            ax3.axhline(global_median, color='red', linestyle='--', linewidth=1, label=f'Global Median: {global_median:.3f}')
            ax3.legend()

        # SAVE SUMMARY
        plt.tight_layout()
        summary_filename = os.path.join(output_dir, f"summary_{current_key}.jpg")
        plt.savefig(summary_filename, dpi=150)
        plt.close(fig_summary)
        print(f"  Saved {summary_filename}")

        # --- SERIES 2: BARE PERIOD MAP ---
        fig_map = plt.figure(figsize=(10, 10))
        ax_map = fig_map.add_subplot(1, 1, 1)
        ax_map.scatter(centers[:, 0], centers[:, 1], s=1, c='lightgray', alpha=0.5)
        
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=p_min, vmax=p_max)
        
        for cell in current_data:
            bx = cell['bounds_x']
            by = cell['bounds_y']
            w, h = bx[1] - bx[0], by[1] - by[0]
            
            if cell['confidence'] == 'FAIL':
                rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='red', facecolor='none', hatch='xx', alpha=0.5)
                ax_map.add_patch(rect)
            else:
                color = cmap(norm(cell['period']))
                rect = patches.Rectangle((bx[0], by[0]), w, h, linewidth=1, edgecolor='white', facecolor=color, alpha=0.6)
                ax_map.add_patch(rect)
                
                # STYLE UPDATE: Giant Font (26), Black on White Box
                if w > 80 and h > 80: 
                    ax_map.text(bx[0]+w/2, by[0]+h/2, f"{cell['period']:.2f}", 
                             ha='center', va='center', fontsize=26, color='black', fontweight='bold',
                             bbox=dict(facecolor='white', edgecolor='none', pad=4.0))

        ax_map.invert_yaxis()
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # SAVE MAP
        plt.tight_layout()
        map_filename = os.path.join(output_dir, f"period_map_{current_key}.jpg")
        plt.savefig(map_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_map)
        print(f"  Saved {map_filename}")

    print("Batch export complete.")


# ==============================================================================
#  EXAMPLE USAGE BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Synthesize dummy data for demonstration
    # True Grid: Period 40px, slightly rotated
    print("Generating Synthetic Data...")
    true_period = 40.0
    x = np.arange(0, 1000, true_period)
    y = np.arange(0, 1000, 10) # Lines made of dots
    xv, yv = np.meshgrid(x, y)
    centers = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Add Noise and Rotation (5 degrees)
    theta = np.radians(5)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centers = centers @ rot_matrix.T
    centers += np.random.normal(0, 0.5, size=centers.shape) # Jitter
    
    # --- RUN THE SOLVER ---
    solver = GridHierarchicalSolver(centers)
    
    # Solve for X-spacing (Vertical Lines)
    # This will likely split 1x1, 2x2, 3x3... 
    # And then might extend to 3x4, 3x5 (splitting Y more) because splitting X kills the period.
    results = solver.run_multiscale_analysis(optimize_axis='x', max_global_split=5)
    
    # Accessing Global Result
    print("\n--- GLOBAL RESULT ---")
    print(results['1x1'][0])

    save_grid_analysis_frames(results, centers, output_dir="output")
    plot_grid_analysis(results, centers)

"""
```
"""