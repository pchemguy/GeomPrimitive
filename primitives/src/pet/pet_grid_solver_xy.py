"""
```
pet_grid_solver_xy.py
---------------------
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


from pet_grid_solver_extended import GridHierarchicalSolver
from pet_grid_postprocessor import GridPostProcessor


class GridHierarchicalXYSolver:
    """
    Orchestrates the solution of a 2D grid by treating it as 
    two independent 1D problems (Vertical lines and Horizontal lines).
    """
    def __init__(self, x_centers, y_centers):
        """
        Args:
            x_centers: (N,2) array of points belonging to Vertical Lines.
            y_centers: (M,2) array of points belonging to Horizontal Lines.
        """
        self.x_centers = x_centers
        self.y_centers = y_centers
        
        # Initialize independent solvers
        # They will optimize angles independently, handling shear/perspective.
        self.solver_x = GridHierarchicalSolver(x_centers)
        self.solver_y = GridHierarchicalSolver(y_centers)

    def run_multiscale_analysis(self, max_global_split=6):
        """
        Runs hierarchical analysis on both axes.
        """
        print(f"=== SOLVING X-AXIS (Vertical Lines, Horizontal Period) ===")
        # optimize_axis='x' means we project onto X to find spacing
        results_x = self.solver_x.run_multiscale_analysis(
            optimize_axis='x', 
            max_global_split=max_global_split
        )
        
        print(f"\n=== SOLVING Y-AXIS (Horizontal Lines, Vertical Period) ===")
        # optimize_axis='y' means we project onto Y to find spacing
        results_y = self.solver_y.run_multiscale_analysis(
            optimize_axis='y', 
            max_global_split=max_global_split
        )
        
        return {'x': results_x, 'y': results_y}


class GridPostProcessorXY:
    """
    Integrates X and Y grid solutions into a unified distortion map.
    """
    def __init__(self, xy_results, x_centers, y_centers):
        self.proc_x = GridPostProcessor(xy_results['x'], x_centers)
        self.proc_y = GridPostProcessor(xy_results['y'], y_centers)
        
        # Calculate Unified Bounds (Union of both datasets)
        all_x = np.concatenate([x_centers[:,0], y_centers[:,0]])
        all_y = np.concatenate([x_centers[:,1], y_centers[:,1]])
        
        self.bounds = {
            'x_min': all_x.min(), 'x_max': all_x.max(),
            'y_min': all_y.min(), 'y_max': all_y.max()
        }

    def run_robust_analysis(self, rms_threshold=15.0, outlier_tol=0.25):
        """
        Cleans both datasets independently using the robust statistics engine.
        """
        print("\n--- Cleaning X-Axis Data ---")
        self.proc_x.run_robust_analysis(rms_threshold, layer_failure_tolerance=0.4, outlier_tolerance=outlier_tol)
        
        print("\n--- Cleaning Y-Axis Data ---")
        self.proc_y.run_robust_analysis(rms_threshold, layer_failure_tolerance=0.4, outlier_tolerance=outlier_tol)

    def generate_unified_maps(self, resolution_bins=(50, 50)):
        """
        Generates aligned X and Y period maps and an Aspect Ratio map.
        """
        nx, ny = resolution_bins
        
        # Force both processors to use the GLOBAL unified bounds
        # This ensures pixel (0,0) in Map X corresponds exactly to pixel (0,0) in Map Y
        self._force_bounds(self.proc_x)
        self._force_bounds(self.proc_y)
        
        # Generate independent maps
        edges_x, edges_y, Z_x = self.proc_x.generate_consensus_map_edges(resolution_bins)
        _, _, Z_y = self.proc_y.generate_consensus_map_edges(resolution_bins)
        
        if Z_x is None or Z_y is None:
            print("Error: One of the dimensions failed to produce a map.")
            return None, None, None, None
            
        # Calculate Aspect Ratio (Squareness)
        # Ratio = Px / Py. Ideal square grid = 1.00.
        # Mask out areas where either X or Y data is missing
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_ratio = Z_x / Z_y
            
        return edges_x, edges_y, Z_x, Z_y, Z_ratio

    def _force_bounds(self, processor):
        """Helper to synchronize map boundaries"""
        processor.x_min = self.bounds['x_min']
        processor.x_max = self.bounds['x_max']
        processor.y_min = self.bounds['y_min']
        processor.y_max = self.bounds['y_max']

    def plot_xy_dashboard(self, output_dir="output", source_image=None):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        ex, ey, Zx, Zy, Zr = self.generate_unified_maps()
        if Zx is None: return

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Shared Plotting Helper
        def plot_subplot(ax, Z, title, label, cmap='viridis', center_val=None):
            mask = np.ma.masked_invalid(Z)
            
            if center_val:
                # For Aspect Ratio, center color map on 1.0
                # Use Diverging colormap (Red = High, Blue = Low, White = 1.0)
                vmin = center_val - 0.1
                vmax = center_val + 0.1
                cmap = 'RdBu_r' 
            else:
                # Robust auto-scaling
                valid = Z[~np.isnan(Z)]
                if len(valid) > 0:
                    vmin = np.percentile(valid, 2)
                    vmax = np.percentile(valid, 98)
                else:
                    vmin, vmax = 0, 1
            
            if source_image is not None:
                ax.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
                alpha = 0.5
            else:
                ax.invert_yaxis()
                alpha = 1.0
                
            mesh = ax.pcolormesh(ex, ey, mask, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, shading='flat')
            cbar = plt.colorbar(mesh, ax=ax, format='%.2f')
            cbar.set_label(label)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])

        # 1. X-Period Map
        plot_subplot(axes[0], Zx, "Horizontal Period (Px)", "Period (px)")
        
        # 2. Y-Period Map
        plot_subplot(axes[1], Zy, "Vertical Period (Py)", "Period (px)")
        
        # 3. Aspect Ratio Map
        # This shows Anisotropy.
        # 1.0 = Perfect Square. 
        # >1.0 = Rectangular (Wide). <1.0 = Rectangular (Tall).
        # Gradients here indicate TILT/Perspective slant.
        plot_subplot(axes[2], Zr, "Grid Aspect Ratio (Px / Py)", "Ratio", center_val=1.0)

        outfile = os.path.join(output_dir, "xy_consensus_dashboard.jpg")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"XY Dashboard saved to {outfile}")


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Assuming you have 'centers_red' (Horizontal lines) and 'centers_blue' (Vertical lines)
    # AND 'source_img' loaded
    
    # 1. Solve
    xy_solver = GridHierarchicalXYSolver(centers_blue, centers_red)
    xy_results = xy_solver.run_multiscale_analysis(max_global_split=6)
    
    # 2. Post-Process
    xy_processor = GridPostProcessorXY(xy_results, centers_blue, centers_red)
    xy_processor.run_robust_analysis()
    
    # 3. Viz
    xy_processor.plot_xy_dashboard(output_dir="output", source_image=source_img)


"""
```
"""
