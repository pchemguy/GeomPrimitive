"""
```
pet_kde_interactive.py
----------------------

https://gemini.google.com/app/97e64fc85d4b0264
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.gridspec import GridSpec
from scipy.stats import norm


def plot_kde_interactive(data, bw=1):
    """
    Creates an interactive visualization tool for analyzing 2D point cloud density distributions 
    under rotation.

    The tool displays two linked plots:
    1.  **Scatter Plot (Left):** Shows the 2D point cloud. It rotates based on user input 
        to simulate projecting the data onto the X-axis from different angles.
    2.  **KDE Plot (Right):** Shows the Kernel Density Estimation of the projected 
        X-coordinates. This represents the marginal density along the current X-axis.

    Key Features
    ------------
    -   **Quartile Analysis:** The projected data is automatically split into 4 equal-count 
        quartiles (Q1, Q2, Q3, Q4) based on their X-position.
    -   **Statistical Summary:** A dedicated header panel displays scaled metrics for each group:
        -   *Std Den:* Standard deviation of the density values (measure of variation/contrast).
        -   *Avg PkDen:* Average density of the top 10% of points in that group (measure of peak concentration).
        -   *Max PkDen:* Maximum density value observed in the group.
    -   **Interactive Controls:**
        -   **Rotation Sliders:** Coarse (-100deg to 100deg) and Fine (-1.0deg to 1.0deg) controls.
        -   **Spinner Buttons:** Vertical arrows (>/<) for precise stepping (1.0deg coarse, 0.05deg fine).
            The Fine control implements "odometer" logic: crossing +/- 1.0 transfers 
            integer degrees to the Coarse control.
        -   **Bandwidth Slider:** Adjusts the sigma (smoothing) of the Gaussian KDE.
        -   **View Selector:** Radio buttons to zoom the KDE view into specific sections 
            (FULL, Halves H1/H2, or Quartiles Q1-Q4).

    Parameters
    ----------
    data : numpy.ndarray
        An (N, 2) array of Cartesian coordinates representing the point cloud.
        Must contain at least 4 points.
    bw : float, optional
        The initial bandwidth (sigma) for the Gaussian Kernel Density Estimation.
        Default is 1.
        The interactive slider range is automatically scaled based on the dataset size, 
        capped at 1% of the point count.

    Returns
    -------
    None
        The function displays a Matplotlib figure and blocks execution until closed.
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
    fig = plt.figure(figsize=(15, 9))
    
    # Height ratios: 15% Stats, 85% Plots
    gs = GridSpec(2, 2, height_ratios=[0.15, 0.85], width_ratios=[1, 1.5], figure=fig)
    plt.subplots_adjust(bottom=0.25, top=0.95, wspace=0.2, hspace=0.2)

    # --- Header Axis (Stats) ---
    ax_stats = fig.add_subplot(gs[0, :]) 
    ax_stats.axis('off') 
    
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

    # 5. Controls (Sliders + Spinner Buttons + Radio)
    
    # --- Coarse Rotation Row (y=0.08) ---
    # Reduced width to 0.22 to make room for value text
    ax_slider_rot = plt.axes([0.15, 0.08, 0.22, 0.03])
    # Buttons moved to x=0.42
    ax_rot_inc = plt.axes([0.42, 0.095, 0.02, 0.015]) # Top
    ax_rot_dec = plt.axes([0.42, 0.080, 0.02, 0.015]) # Bottom
    
    # --- Fine Rotation Row (y=0.04) ---
    # Reduced width to 0.22
    ax_slider_rot_fine = plt.axes([0.15, 0.04, 0.22, 0.03])
    # Buttons moved to x=0.42
    ax_fine_inc = plt.axes([0.42, 0.055, 0.02, 0.015]) # Top
    ax_fine_dec = plt.axes([0.42, 0.040, 0.02, 0.015]) # Bottom

    # Bandwidth Slider (Right side)
    ax_slider_bw = plt.axes([0.65, 0.08, 0.25, 0.03])
    
    # Radio Buttons (Centered, moved to 0.50)
    ax_radio = plt.axes([0.50, 0.02, 0.08, 0.20]) 

    # --- Create Widgets ---
    slider_rot = Slider(ax_slider_rot, 'Rotation', -100, 100, valinit=0)
    slider_rot_fine = Slider(ax_slider_rot_fine, 'Fine Rot', -1.0, 1.0, valinit=0) # Range -1 to 1
    
    # Vertical Spinner Buttons (Text Based)
    btn_rot_inc = Button(ax_rot_inc, '>', hovercolor='0.9')
    btn_rot_inc.label.set_fontsize(6)
    
    btn_rot_dec = Button(ax_rot_dec, '<', hovercolor='0.9')
    btn_rot_dec.label.set_fontsize(6)
    
    btn_fine_inc = Button(ax_fine_inc, '>', hovercolor='0.9')
    btn_fine_inc.label.set_fontsize(6)
    
    btn_fine_dec = Button(ax_fine_dec, '<', hovercolor='0.9')
    btn_fine_dec.label.set_fontsize(6)
    
    # Max sigma = 1% of point count (rounded up)
    sigma_max = max(2, int(np.ceil(n_points * 0.01)))
    slider_bw = Slider(ax_slider_bw, 'Sigma', 1.0, sigma_max, valinit=min(max(bw, 1.0), sigma_max))
    
    # Radio Buttons with H1 and H2
    radio = RadioButtons(ax_radio, ('FULL', 'H1', 'H2', 'Q1', 'Q2', 'Q3', 'Q4'), active=0)

    # --- Callback Logic ---
    def inc_rot(event):
        slider_rot.set_val(slider_rot.val + 1.0)
    def dec_rot(event):
        slider_rot.set_val(slider_rot.val - 1.0)
        
    def inc_fine(event):
        # Step is 0.05
        new_val = slider_rot_fine.val + 0.05
        
        # Logic: Transfer to coarse ONLY if strictly > 1.0
        # We use a small epsilon for float comparison safety
        if new_val > 1.0 + 1e-9:
            # Transfer 1.0 deg to coarse
            slider_rot.set_val(slider_rot.val + 1.0)
            # Wrap fine value (preserve the remainder)
            slider_rot_fine.set_val(new_val - 1.0)
        else:
            # Just increment fine (it will be clamped by slider if > 1.0 but <= 1.0+step)
            # Matplotlib slider set_val clamps to valmax/valmin automatically
            slider_rot_fine.set_val(new_val)

    def dec_fine(event):
        # Step is 0.05
        new_val = slider_rot_fine.val - 0.05
        
        # Logic: Transfer to coarse ONLY if strictly < -1.0
        if new_val < -1.0 - 1e-9:
            # Transfer -1.0 deg to coarse
            slider_rot.set_val(slider_rot.val - 1.0)
            # Wrap fine value
            slider_rot_fine.set_val(new_val + 1.0)
        else:
            slider_rot_fine.set_val(new_val)

    btn_rot_inc.on_clicked(inc_rot)
    btn_rot_dec.on_clicked(dec_rot)
    btn_fine_inc.on_clicked(inc_fine)
    btn_fine_dec.on_clicked(dec_fine)

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
        # Note: Automatic overflow handling removed from here.
        # Overflow is now strictly handled by Spinner Buttons.
        
        sigma = slider_bw.val
        # Combine Coarse and Fine rotation
        angle = slider_rot.val + slider_rot_fine.val
        view_mode = radio.value_selected
        
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
        
        stats_text.set_text(f"STATS SUMMARY (Scaled x{scale_factor})\n{header}\n{'-'*56}\n{row_std}\n{row_avg}\n{row_max}")

        # F. Handle View Scaling
        if view_mode == 'FULL':
            ax_den.set_xlim(grid_new[0], grid_new[-1])
            ax_den.set_ylim(0, np.max(den_new) * 1.1)
        else:
            if view_mode == 'H1':
                s_i, e_i = 0, 2 
            elif view_mode == 'H2':
                s_i, e_i = 2, 4 
            elif view_mode.startswith('Q'):
                q_num = int(view_mode[1])
                s_i, e_i = q_num - 1, q_num 
            
            start_idx = q_inds[s_i]
            end_idx = q_inds[e_i] - 1
            
            if start_idx <= end_idx:
                x_start = x_new[start_idx]
                x_end = x_new[end_idx]
                width = x_end - x_start
                pad = width * 0.05
                view_min = x_start - pad
                view_max = x_end + pad
                
                mask = (grid_new >= view_min) & (grid_new <= view_max)
                if np.any(mask):
                    local_max_y = np.max(den_new[mask])
                else:
                    local_max_y = np.max(den_new)

                ax_den.set_xlim(view_min, view_max)
                ax_den.set_ylim(0, local_max_y * 1.1)

        fig.canvas.draw_idle()

    # Link sliders to update function
    update(0)
    slider_bw.on_changed(update)
    slider_rot.on_changed(update)
    slider_rot_fine.on_changed(update)
    radio.on_clicked(update)

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