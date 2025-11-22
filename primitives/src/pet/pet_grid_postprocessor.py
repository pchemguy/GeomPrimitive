"""
```
pet_grid_postprocessor.py
-------------------------

https://gemini.google.com/app/1cd765eae3be9bdb

--- USAGE EXAMPLE ---
1. Create Processor
processor = GridPostProcessor(results, centers)

2. Run Filter Pipeline
valid_cells, stats = processor.run_robust_analysis(
    rms_threshold=2.0,          # Reject noisy cells
    layer_failure_tolerance=0.25, # Reject broken 6x11 layers
    outlier_tolerance=0.25      # Reject periods > 25% off median
)

3. Generate & Plot Consensus
processor.plot_consensus()

"""

import os
import numpy as np
import matplotlib.pyplot as plt


class GridPostProcessor:
    def __init__(self, results_tree, centers):
        self.raw_tree = results_tree
        self.centers = centers
        self.x_min, self.x_max = centers[:,0].min(), centers[:,0].max()
        self.y_min, self.y_max = centers[:,1].min(), centers[:,1].max()

    def run_robust_analysis(self, 
                            rms_threshold=15.0,
                            layer_failure_tolerance=0.40,
                            outlier_tolerance=0.25):
        """
        Executes the statistical cleaning pipeline.
        """
        print(f"--- Starting Statistical Post-Processing (RMS Cutoff: {rms_threshold}) ---")
        
        initial_pool = []
        valid_layers = []
        
        for layer_key, cells in self.raw_tree.items():
            layer_total = len(cells)
            if layer_total == 0: continue

            layer_valid_cells = []
            layer_rms_values = []
            
            for c in cells:
                # Metric collection for debugging
                if 'rms_error' in c: layer_rms_values.append(c['rms_error'])

                # Check 1: Algorithm must have succeeded
                if c['confidence'] == 'FAIL': continue
                
                # Check 2: RMS Error must be reasonable (but not perfect)
                # Real world data often has RMS of 3-6px due to lens distortion.
                if c.get('rms_error', 999) > rms_threshold: continue
                
                # Check 3: Sanity check on period
                if c['period'] <= 1: continue
                
                layer_valid_cells.append(c)
            
            # Calc stats for user feedback
            avg_rms = np.mean(layer_rms_values) if layer_rms_values else 999
            failure_rate = 1.0 - (len(layer_valid_cells) / layer_total)
            
            # 2. LAYER REJECTION
            if failure_rate > layer_failure_tolerance:
                print(f"  [DROP] Layer {layer_key}: {failure_rate*100:.1f}% rejected (Avg RMS: {avg_rms:.2f})")
                continue 
            else:
                # print(f"  [KEEP] Layer {layer_key}: {failure_rate*100:.1f}% rejected (Avg RMS: {avg_rms:.2f})")
                valid_layers.append(layer_key)
                initial_pool.extend(layer_valid_cells)

        if not initial_pool:
            print("CRITICAL: No cells survived initial filtering. Check if RMS Threshold is too low.")
            return [], {}

        # 3. GLOBAL OUTLIER REJECTION (The "Median" Anchor)
        all_periods = np.array([c['period'] for c in initial_pool])
        global_median = np.median(all_periods)
        
        print(f"  Global Median Established: {global_median:.3f} px")
        
        final_pool = []
        dropped_outliers = 0
        
        for c in initial_pool:
            # Check 4: Deviation from consensus
            # This is what will catch your "15.9" cell (which is ~60% off from 43.0)
            deviation = abs(c['period'] - global_median) / global_median
            
            if deviation <= outlier_tolerance:
                final_pool.append(c)
            else:
                dropped_outliers += 1
        
        if dropped_outliers > 0:
            print(f"  [CLEANUP] Dropped {dropped_outliers} outliers deviating > {outlier_tolerance*100}% from median.")

        # 4. FINAL STATISTICS
        if not final_pool:
            print("CRITICAL: All cells were outliers.")
            return [], {}

        final_values = np.array([c['period'] for c in final_pool])
        stats = {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std_dev': np.std(final_values),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'count': len(final_values),
            'valid_layers': valid_layers
        }
        
        print(f"--- Processing Complete: Kept {len(final_pool)} cells from {len(valid_layers)} layers ---")
        print(f"    Final Mean: {stats['mean']:.3f} | StdDev: {stats['std_dev']:.3f}")
        
        self.clean_cells = final_pool
        self.stats = stats
        return final_pool, stats

    def generate_consensus_map(self, resolution_bins=(50, 50)):
        """
        Generates a high-res 'Heatmap' by averaging all overlapping valid cells.
        """
        if not hasattr(self, 'clean_cells'):
            print("Error: Run run_robust_analysis() first.")
            return None, None, None

        nx, ny = resolution_bins
        x_space = np.linspace(self.x_min, self.x_max, nx)
        y_space = np.linspace(self.y_min, self.y_max, ny)
        
        X_grid, Y_grid = np.meshgrid((x_space[:-1]+x_space[1:])/2, (y_space[:-1]+y_space[1:])/2)
        
        sum_grid = np.zeros((ny-1, nx-1))
        count_grid = np.zeros((ny-1, nx-1))
        
        for cell in self.clean_cells:
            p = cell['period']
            bx = cell['bounds_x']
            by = cell['bounds_y']
            
            col_mask = (x_space[:-1] >= bx[0]) & (x_space[1:] <= bx[1])
            row_mask = (y_space[:-1] >= by[0]) & (y_space[1:] <= by[1])
            
            mask_2d = np.outer(row_mask, col_mask)
            
            sum_grid[mask_2d] += p
            count_grid[mask_2d] += 1
            
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_grid = sum_grid / count_grid
            
        return X_grid, Y_grid, Z_grid

    def plot_consensus(self, output_file="output/consensus_map.jpg"):
        """
        Generates and saves the final consensus heatmap.
        
        Args:
            output_file: Path to save the jpeg. Defaults to 'output/consensus_map.jpg'.
        """
        # 1. Ensure Output Directory Exists
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # 2. Generate Data
        X, Y, Z = self.generate_consensus_map()
        if Z is None: return

        # 3. Setup Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        Z_masked = np.ma.masked_invalid(Z)
        
        # Calculate Robust Color Limits (Mean +/- 2 StdDev)
        # This ensures the map highlights local variations, not global outliers
        vmin = self.stats['mean'] - 2 * self.stats['std_dev']
        vmax = self.stats['mean'] + 2 * self.stats['std_dev']
        
        # Plot Heatmap
        mesh = ax.pcolormesh(X, Y, Z_masked, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        
        # Overlay Original Points (Faintly) for Context
        ax.scatter(self.centers[:,0], self.centers[:,1], s=1, c='white', alpha=0.1)
        
        ax.invert_yaxis() # Match image coordinates
        
        # Add Colorbar
        cbar = plt.colorbar(mesh, ax=ax, format='%.1f')
        cbar.set_label(f"Consensus Period (px)\nGlobal Avg: {self.stats['mean']:.2f}")
        
        ax.set_title("Statistical Consensus Map\n(Averaged from all valid overlapping layers)")
        
        # 4. Save
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        # plt.show() # Optional: Comment out if running purely in batch mode
        plt.close(fig)
        print(f"Consensus map saved to {output_file}")

"""
```
"""