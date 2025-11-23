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
    Splits data into 4 quartiles and calculates stats for each.

    Parameters:
    -----------
    data : numpy.ndarray
        Nx2 Array of data points (x, y).
    bw : float
        Initial bandwidth (sigma).
    """
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data must be Nx2 for scatter plot rotation.")
    
    n_points = len(data)
    if n_points < 4:
        print("Need at least 4 points to calculate quartiles.")
        return

    # Calculate scaling factor for display
    scale_exp = np.ceil(np.log10(n_points))
    scale_factor = 10 ** int(scale_exp)
    print(f"Stats Scaling Factor: {scale_factor} (based on N={n_points})")

    # 1. Pre-calculate Geometry for Rotation
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    center = (min_vals + max_vals) / 2
    
    max_dist = np.max(np.linalg.norm(data - center, axis=1))
    limit_padding = max_dist * 1.2
    
    xlim_scat = (center[0] - limit_padding, center[0] + limit_padding)
    ylim_scat = (center[1] - limit_padding, center[1] + limit_padding)

    # 2. Calculation Helper (Rotation)
    def rotate_data(points, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        centered = points - center
        rotated = centered @ R.T
        return rotated + center

    # 3. Calculation Helper (KDE)
    def calculate_kde(x_sorted, sigma):
        n = len(x_sorted)
        span = x_sorted[-1] - x_sorted[0]
        if span == 0: span = 1.0 
        
        pad = span * 0.2
        grid = np.linspace(x_sorted[0] - pad, x_sorted[-1] + pad, 500)
        
        if sigma <= 0: sigma = 1e-5

        pdfs = norm.pdf(grid[:, None], loc=x_sorted[None, :], scale=sigma)
        y_den = np.sum(pdfs, axis=1) / n
        
        return grid, y_den

    # 4. Setup Plot Layout
    fig = plt.figure(figsize=(15, 9)) # Increased height for header
    
    # Create 2 Rows: Top (Stats) and Bottom (Plots)
    # height_ratios=[0.15, 0.85] reserves top 15% for text
    gs = GridSpec(2, 2, height_ratios=[0.15, 0.85], width_ratios=[1, 1.5], figure=fig)
    plt.subplots_adjust(bottom=0.20, top=0.95, wspace=0.2, hspace=0.2)

    # --- Header Axis (Stats) ---
    ax_stats = fig.add_subplot(gs[0, :]) # Span all columns
    ax_stats.axis('off') # Hide axis lines
    
    # Initialize Text Object (Centered)
    stats_text = ax_stats.text(0.5, 0.5, "", ha='center', va='center', 
                               fontname='monospace', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))

    # --- Plot Axes (Bottom) ---
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_den = fig.add_subplot(gs[1, 1]) 

    # Initial State
    current_data = data
    x_current = np.sort(current_data[:, 0])
    x_grid, y_den = calculate_kde(x_current, bw)

    # Plot 1: Scatter
    scat_plot = ax_scat.scatter(current_data[:, 0], current_data[:, 1], 
                                alpha=0.6, c='purple', edgecolors='w', s=50)
    ax_scat.set_title("2D Data Source")
    ax_scat.set_xlim(xlim_scat)
    ax_scat.set_ylim(ylim_scat)
    ax_scat.grid(True, linestyle='--', alpha=0.4)
    ax_scat.set_aspect('equal', adjustable='box')

    # Plot 2: Density
    line_den, = ax_den.plot(x_grid, y_den, color='blue', lw=2)
    rug_lines, = ax_den.plot(x_current, np.zeros_like(x_current), '|', color='black', alpha=0.3)
    
    # Quartile Divider Lines
    vlines = [ax_den.axvline(x=0, color='red', linestyle='--', alpha=0.5) for _ in range(3)]

    ax_den.set_title("Projected Density (Marginal X)")
    ax_den.set_ylabel("Density")
    ax_den.grid(True, alpha=0.3)

    # 5. Sliders
    ax_slider_bw = plt.axes([0.6, 0.05, 0.3, 0.03])
    ax_slider_rot = plt.axes([0.15, 0.05, 0.3, 0.03])

    slider_bw = Slider(ax_slider_bw, 'Sigma', 1.0, limit_padding * 0.5, valinit=max(bw, 1.0))
    slider_rot = Slider(ax_slider_rot, 'Rotation', -100, 100, valinit=0)

    # 6. Logic to Calculate Group Stats
    def get_group_stats(x_group, grid_full, den_full):
        if len(x_group) == 0: return 0.0, 0.0, 0.0
        
        g_min, g_max = x_group[0], x_group[-1]
        mask = (grid_full >= g_min) & (grid_full <= g_max)
        local_densities = den_full[mask]
        
        if len(local_densities) > 0:
            std_val = np.std(local_densities)
            local_max = np.max(local_densities)
            top_tier = local_densities[local_densities >= 0.9 * local_max]
            avg_top_den = np.mean(top_tier)
        else:
            std_val = 0.0
            avg_top_den = 0.0
            local_max = 0.0
            
        return std_val, avg_top_den, local_max

    # 7. Update Function
    def update(val):
        sigma = slider_bw.val
        angle = slider_rot.val
        
        # A. Rotate
        rotated_data = rotate_data(data, angle)
        scat_plot.set_offsets(rotated_data)
        
        # B. Sort X
        x_new = np.sort(rotated_data[:, 0])
        
        # C. KDE
        grid_new, den_new = calculate_kde(x_new, sigma)
        line_den.set_data(grid_new, den_new)
        rug_lines.set_data(x_new, np.zeros_like(x_new))
        
        # D. Split Groups
        q_inds = np.linspace(0, len(x_new), 5, dtype=int)
        
        stats_std = []
        stats_avg_den = []
        stats_max_den = []
        boundaries = []

        for i in range(4):
            group = x_new[q_inds[i]:q_inds[i+1]]
            if i < 3: boundaries.append(x_new[q_inds[i+1]])
            
            std, avg_den, max_den = get_group_stats(group, grid_new, den_new)
            stats_std.append(std * scale_factor)
            stats_avg_den.append(avg_den * scale_factor)
            stats_max_den.append(max_den * scale_factor)

        # E. Update Visuals
        for line, x_pos in zip(vlines, boundaries):
            line.set_xdata([x_pos, x_pos])

        # Update Text Table
        header = "Group    |    Q1    |    Q2    |    Q3    |    Q4    "
        row_std = f"Std Den  | {stats_std[0]:8.4f} | {stats_std[1]:8.4f} | {stats_std[2]:8.4f} | {stats_std[3]:8.4f}"
        row_avg = f"Avg PkDen| {stats_avg_den[0]:8.4f} | {stats_avg_den[1]:8.4f} | {stats_avg_den[2]:8.4f} | {stats_avg_den[3]:8.4f}"
        row_max = f"Max PkDen| {stats_max_den[0]:8.4f} | {stats_max_den[1]:8.4f} | {stats_max_den[2]:8.4f} | {stats_max_den[3]:8.4f}"
        
        # Update text in the top panel
        stats_text.set_text(f"STATS SUMMARY (Scaled x{scale_factor})\n{header}\n{'-'*56}\n{row_std}\n{row_avg}\n{row_max}")

        ax_den.set_xlim(grid_new[0], grid_new[-1])
        ax_den.set_ylim(0, np.max(den_new) * 1.1)
        
        fig.canvas.draw_idle()

    update(0)
    slider_bw.on_changed(update)
    slider_rot.on_changed(update)

    plt.show()


def main():
    # Create a synthetic grid with a hole
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Add noise
    points += np.random.normal(0, 2, points.shape)
    
    # Remove some points to create density differences
    mask = (points[:, 0] > 30) & (points[:, 0] < 70) & (points[:, 1] > 30) & (points[:, 1] < 70)
    points = points[~mask]

    plot_kde_interactive(points, bw=5)

if __name__ == "__main__":
    main()


"""
```
"""