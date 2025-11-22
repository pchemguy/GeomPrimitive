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
import cv2


class GridPostProcessor:
    def __init__(self, results_tree, centers):
        self.raw_tree = results_tree
        self.centers = centers
        self.x_min, self.x_max = centers[:,0].min(), centers[:,0].max()
        self.y_min, self.y_max = centers[:,1].min(), centers[:,1].max()

    # ... [run_robust_analysis method stays exactly the same] ...
    def run_robust_analysis(self, rms_threshold=15.0, layer_failure_tolerance=0.40, outlier_tolerance=0.25):
        # (Copy previous logic here or import it to save space)
        # For brevity, I assume this method is unchanged from the previous working version.
        # ...
        # (Let me know if you need the full code block for this method again)
        print(f"--- Starting Statistical Post-Processing (RMS Cutoff: {rms_threshold}) ---")
        initial_pool = []
        valid_layers = []
        for layer_key, cells in self.raw_tree.items():
            layer_total = len(cells)
            if layer_total == 0: continue
            layer_valid_cells = []
            layer_rms_values = []
            for c in cells:
                if 'rms_error' in c: layer_rms_values.append(c['rms_error'])
                if c['confidence'] == 'FAIL': continue
                if c.get('rms_error', 999) > rms_threshold: continue
                if c['period'] <= 1: continue
                layer_valid_cells.append(c)
            avg_rms = np.mean(layer_rms_values) if layer_rms_values else 999
            failure_rate = 1.0 - (len(layer_valid_cells) / layer_total)
            if failure_rate > layer_failure_tolerance:
                print(f"  [DROP] Layer {layer_key}: {failure_rate*100:.1f}% rejected (Avg RMS: {avg_rms:.2f})")
                continue 
            else:
                valid_layers.append(layer_key)
                initial_pool.extend(layer_valid_cells)

        if not initial_pool:
            return [], {}

        all_periods = np.array([c['period'] for c in initial_pool])
        global_median = np.median(all_periods)
        print(f"  Global Median Established: {global_median:.3f} px")
        
        final_pool = []
        for c in initial_pool:
            deviation = abs(c['period'] - global_median) / global_median
            if deviation <= outlier_tolerance:
                final_pool.append(c)
        
        if not final_pool: return [], {}

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
        
        self.clean_cells = final_pool
        self.stats = stats
        return final_pool, stats

    def generate_consensus_map_edges(self, resolution_bins=(50, 50)):
        """
        Generates grid EDGES (not centers) for pixel-perfect alignment.
        """
        if not hasattr(self, 'clean_cells'):
            print("Error: Run run_robust_analysis() first.")
            return None, None, None

        nx, ny = resolution_bins
        # Expand bounds slightly so we don't miss edge points
        pad = 10 
        x_edges = np.linspace(self.x_min - pad, self.x_max + pad, nx + 1)
        y_edges = np.linspace(self.y_min - pad, self.y_max + pad, ny + 1)
        
        # Accumulators for the bins
        # Shape is (ny, nx) because edges are n+1
        sum_grid = np.zeros((ny, nx))
        count_grid = np.zeros((ny, nx))
        
        for cell in self.clean_cells:
            p = cell['period']
            bx = cell['bounds_x']
            by = cell['bounds_y']
            
            # Vectorized binning
            # Find indices where this cell overlaps
            # Logic: cell covers bins where bin_center is inside cell bounds
            
            # Easier visualization logic:
            # Mark bins that are mostly covered by the cell
            col_start = np.searchsorted(x_edges, bx[0])
            col_end = np.searchsorted(x_edges, bx[1])
            row_start = np.searchsorted(y_edges, by[0])
            row_end = np.searchsorted(y_edges, by[1])
            
            # Clamp
            col_start = max(0, col_start-1); col_end = min(nx, col_end)
            row_start = max(0, row_start-1); row_end = min(ny, row_end)
            
            if col_end > col_start and row_end > row_start:
                sum_grid[row_start:row_end, col_start:col_end] += p
                count_grid[row_start:row_end, col_start:col_end] += 1
            
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_grid = sum_grid / count_grid
            
        return x_edges, y_edges, Z_grid

    def plot_consensus(self, output_dir="output", source_image=None):
        """
        Saves consensus map and overlay with strict coordinate locking.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Get Edge-aligned Grid
        x_edges, y_edges, Z = self.generate_consensus_map_edges()
        if Z is None: return
        
        Z_masked = np.ma.masked_invalid(Z)
        vmin = self.stats['mean'] - 2 * self.stats['std_dev']
        vmax = self.stats['mean'] + 2 * self.stats['std_dev']

        # --- PLOT 1: Standalone Map ---
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        
        # Use pcolormesh with EDGES. 
        # x_edges has length N+1, Z has shape N. This defines corners perfectly.
        mesh1 = ax1.pcolormesh(x_edges, y_edges, Z_masked, cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
        
        ax1.scatter(self.centers[:,0], self.centers[:,1], s=1, c='white', alpha=0.1)
        ax1.invert_yaxis()
        cbar1 = plt.colorbar(mesh1, ax=ax1, format='%.1f')
        cbar1.set_label(f"Consensus Period (px)")
        ax1.set_title("Statistical Consensus Map (Edge-Aligned)")
        
        map_path = os.path.join(output_dir, "consensus_map.jpg")
        plt.savefig(map_path, dpi=150)
        plt.close(fig1)

        # --- PLOT 2: Overlay ---
        if source_image is not None:
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            
            # 1. Show Image
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            ax2.imshow(rgb_img)
            
            # 2. Overlay Heatmap
            # We trust x_edges/y_edges to match the image pixel coordinates 
            # because 'centers' were derived from those same coordinates.
            mesh2 = ax2.pcolormesh(x_edges, y_edges, Z_masked, cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.5, shading='flat')
            
            # 3. Verification Dots (Optional but recommended)
            # Plots faint white dots where the algorithm *thinks* the grid lines are.
            # If these dots don't sit on the lines in the photo, your 'centers' array is wrong.
            # ax2.scatter(self.centers[::10,0], self.centers[::10,1], s=1, c='white', alpha=0.3)
            
            cbar2 = plt.colorbar(mesh2, ax=ax2, format='%.1f')
            cbar2.set_label("Period (px)")
            ax2.set_title("Consensus Overlay")
            ax2.set_xticks([]); ax2.set_yticks([])
            
            overlay_path = os.path.join(output_dir, "consensus_overlay.jpg")
            plt.tight_layout()
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"Overlay saved to {overlay_path}")

"""
```
"""