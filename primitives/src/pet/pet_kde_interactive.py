"""
```
pet_kde_interactive.py
----------------------

https://gemini.google.com/app/97e64fc85d4b0264
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from scipy.stats import norm


def plot_kde_interactive(data, bw=1):
    """
    Creates an interactive plot with Scatter (Left) and KDE (Right).
    Allows rotation of the dataset to see marginal density changes.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Nx2 Array of data points (x, y).
    bw : float
        Initial bandwidth (sigma).
    """
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data must be Nx2 for scatter plot rotation.")

    # 1. Pre-calculate Geometry for Rotation
    # Center of rotation = Center of the bounding box
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    center = (min_vals + max_vals) / 2
    
    # Calculate max radius for stable axes limits (so the plot doesn't jitter while rotating)
    max_dist = np.max(np.linalg.norm(data - center, axis=1))
    limit_padding = max_dist * 1.2
    
    # limits for scatter plot
    xlim_scat = (center[0] - limit_padding, center[0] + limit_padding)
    ylim_scat = (center[1] - limit_padding, center[1] + limit_padding)

    # 2. Calculation Helper (Rotation)
    def rotate_data(points, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        # Rotation matrix
        R = np.array([[c, -s], [s, c]])
        # Translate -> Rotate -> Translate back
        centered = points - center
        rotated = centered @ R.T
        return rotated + center

    # 3. Calculation Helper (KDE)
    def calculate_kde(x_sorted, sigma):
        n = len(x_sorted)
        # Dynamic Grid: Re-calculate grid based on current x-span
        span = x_sorted[-1] - x_sorted[0]
        # Avoid zero span issues
        if span == 0: span = 1.0 
        
        pad = span * 0.2
        grid = np.linspace(x_sorted[0] - pad, x_sorted[-1] + pad, 500)
        
        if sigma <= 0: sigma = 1e-5

        # Broadcasting: (Grid, 1) vs (1, Data)
        pdfs = norm.pdf(grid[:, None], loc=x_sorted[None, :], scale=sigma)
        
        # Only calculate Density now
        y_den = np.sum(pdfs, axis=1) / n
        
        return grid, y_den

    # 4. Setup Plot Layout
    fig = plt.figure(figsize=(14, 6)) # Adjusted height since we lost a plot
    # 1 Row, 2 Columns
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig)
    plt.subplots_adjust(bottom=0.25, wspace=0.2)

    # Axes
    ax_scat = fig.add_subplot(gs[0]) # Left
    ax_den = fig.add_subplot(gs[1])  # Right

    # Initial State (Angle = 0)
    current_angle = 0.0
    current_data = data # No rotation initially
    x_current = np.sort(current_data[:, 0])
    
    # Calculate KDE
    x_grid, y_den = calculate_kde(x_current, bw)

    # --- Plot 1: Scatter ---
    scat_plot = ax_scat.scatter(current_data[:, 0], current_data[:, 1], 
                                alpha=0.6, c='purple', edgecolors='w', s=50)
    ax_scat.set_title("2D Data Source")
    ax_scat.set_xlabel("x")
    ax_scat.set_ylabel("y")
    ax_scat.set_xlim(xlim_scat)
    ax_scat.set_ylim(ylim_scat)
    ax_scat.grid(True, linestyle='--', alpha=0.4)
    ax_scat.set_aspect('equal', adjustable='box')

    # --- Plot 2: Density ---
    line_den, = ax_den.plot(x_grid, y_den, color='blue', lw=2)
    rug_lines, = ax_den.plot(x_current, np.zeros_like(x_current), '|', color='black', alpha=0.3)
    ax_den.set_title("Projected Density (Marginal X)")
    ax_den.set_ylabel("Density")
    ax_den.set_xlabel("x (Projected)")
    ax_den.grid(True, alpha=0.3)

    # 5. Sliders
    # Ax locations [left, bottom, width, height]
    ax_slider_bw = plt.axes([0.6, 0.1, 0.3, 0.05])
    ax_slider_rot = plt.axes([0.15, 0.1, 0.3, 0.05])

    slider_bw = Slider(
        ax=ax_slider_bw,
        label='Sigma (Bandwidth)',
        valmin=1.0,
        valmax=limit_padding * 0.5,
        valinit=max(bw, 1.0)
    )

    slider_rot = Slider(
        ax=ax_slider_rot,
        label='Rotation (deg)',
        valmin=-100,
        valmax=100,
        valinit=0
    )

    # 6. Update Function
    def update(val):
        sigma = slider_bw.val
        angle = slider_rot.val
        
        # A. Rotate
        rotated_data = rotate_data(data, angle)
        
        # B. Update Scatter
        # Setting offsets expects an (N, 2) array
        scat_plot.set_offsets(rotated_data)
        
        # C. Process New Projection (Sort x)
        x_new = np.sort(rotated_data[:, 0])
        
        # D. Recalculate KDE
        grid_new, den_new = calculate_kde(x_new, sigma)
        
        # E. Update KDE Plot
        line_den.set_data(grid_new, den_new)
        rug_lines.set_data(x_new, np.zeros_like(x_new))
        
        # F. Rescale Axes
        ax_den.set_xlim(grid_new[0], grid_new[-1])
        ax_den.relim()
        ax_den.autoscale_view(scalex=True, scaley=True)
        
        fig.canvas.draw_idle()

    slider_bw.on_changed(update)
    slider_rot.on_changed(update)

    plt.show()

def main():
    # 1. GENERATE DUMMY DATA (2D CLUSTERS)
    np.random.seed(42)
    # Elongated cluster 1
    c1 = np.random.normal(loc=[2, 2], scale=[0.5, 1.5], size=(100, 2))
    # Round cluster 2
    c2 = np.random.normal(loc=[6, 5], scale=[0.8, 0.8], size=(80, 2))
    
    data_points = np.vstack([c1, c2])
    
    # Initial bandwidth guess
    initial_bw = 0.5

    print(f"Launching interactive tool with {len(data_points)} points...")
    plot_kde_interactive(data_points, initial_bw)

if __name__ == "__main__":
    main()


"""
```
"""