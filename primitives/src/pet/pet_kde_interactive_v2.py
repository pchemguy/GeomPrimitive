"""
pet_kde_extended.py
-------------------
Extended interactive KDE tool for 2D point clouds.

New Features:
1. Dynamic Color Mapping: Scatter points change color based on their calculated Quartile.
2. Peak Detection: Automatically marks local maxima on the density curve.
3. Data Loading: Button to load external CSV/TXT files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.signal import find_peaks
import sys
import os

# Try importing tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

def plot_kde_interactive(data, bw=1):
    """
    Interactive 2D projection analysis with Quartile Coloring and Peak Detection.
    """
    # --- Data Validation ---
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data must be Nx2.")
    
    n_points = len(data)
    if n_points < 4:
        print("Need at least 4 points.")
        return

    # --- Constants & Pre-calc ---
    scale_exp = np.ceil(np.log10(n_points))
    scale_factor = 10 ** int(scale_exp)
    
    # Color palette for Quartiles (Q1, Q2, Q3, Q4)
    # Cyan, Green, Orange, Magenta
    Q_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'] 
    
    # Geometry for axis limits
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    center = (min_vals + max_vals) / 2
    max_dist = np.max(np.linalg.norm(data - center, axis=1))
    limit_padding = max_dist * 1.2
    
    xlim_scat = (center[0] - limit_padding, center[0] + limit_padding)
    ylim_scat = (center[1] - limit_padding, center[1] + limit_padding)

    # --- Helper Functions ---
    def rotate_data(points, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        centered = points - center
        rotated = centered @ R.T
        return rotated + center

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

    # --- GUI Setup ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[0.12, 0.88], width_ratios=[1, 1.5], figure=fig)
    plt.subplots_adjust(bottom=0.25, top=0.95, wspace=0.2, hspace=0.2)

    # Header Stats
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.5, 0.5, "", ha='center', va='center', 
                               fontname='monospace', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))

    # Main Plots
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_den = fig.add_subplot(gs[1, 1])

    # Initial Plot Objects
    # We use an empty scatter initially, updated in `update()`
    scat_plot = ax_scat.scatter([], [], alpha=0.6, edgecolors='w', s=40)
    
    line_den, = ax_den.plot([], [], color='k', lw=2, label='Density')
    rug_lines, = ax_den.plot([], [], '|', color='gray', alpha=0.3)
    peak_markers, = ax_den.plot([], [], 'x', color='red', markeredgewidth=2, markersize=8, label='Peaks')
    
    # Vertical dividers for quartiles
    vlines = [ax_den.axvline(x=0, color=c, linestyle='--', alpha=0.8, lw=1.5) 
              for c in Q_COLORS[:-1]] # 3 lines divide 4 areas

    ax_scat.set_title("Rotated Point Cloud (Colored by Projection Quartile)")
    ax_scat.set_xlim(xlim_scat)
    ax_scat.set_ylim(ylim_scat)
    ax_scat.grid(True, linestyle='--', alpha=0.4)
    ax_scat.set_aspect('equal', adjustable='box')

    ax_den.set_title("Marginal Density (X-Projection)")
    ax_den.set_ylabel("Density")
    ax_den.grid(True, alpha=0.3)
    ax_den.legend(loc='upper right', fontsize='small')

    # --- Controls Layout ---
    # Row 1: Rotation
    ax_slider_rot = plt.axes([0.10, 0.10, 0.25, 0.03])
    ax_rot_inc = plt.axes([0.36, 0.115, 0.02, 0.015])
    ax_rot_dec = plt.axes([0.36, 0.100, 0.02, 0.015])
    
    # Row 2: Fine Rotation
    ax_slider_rot_fine = plt.axes([0.10, 0.06, 0.25, 0.03])
    ax_fine_inc = plt.axes([0.36, 0.075, 0.02, 0.015])
    ax_fine_dec = plt.axes([0.36, 0.060, 0.02, 0.015])

    # Right Side Controls
    ax_slider_bw = plt.axes([0.55, 0.10, 0.25, 0.03])
    ax_check_opts = plt.axes([0.82, 0.05, 0.12, 0.10]) # Checkboxes
    ax_button_load = plt.axes([0.02, 0.02, 0.08, 0.04]) # Load Button

    # Widgets
    slider_rot = Slider(ax_slider_rot, 'Rot', -100, 100, valinit=0)
    slider_rot_fine = Slider(ax_slider_rot_fine, 'Fine', -1.0, 1.0, valinit=0)
    
    btn_rot_inc = Button(ax_rot_inc, '>', hovercolor='0.9')
    btn_rot_dec = Button(ax_rot_dec, '<', hovercolor='0.9')
    btn_fine_inc = Button(ax_fine_inc, '>', hovercolor='0.9')
    btn_fine_dec = Button(ax_fine_dec, '<', hovercolor='0.9')
    
    sigma_max = max(2, int(np.ceil(n_points * 0.01)))
    slider_bw = Slider(ax_slider_bw, 'Sigma', 1.0, sigma_max, valinit=bw)
    
    check_opts = CheckButtons(ax_check_opts, ['Color Scatter', 'Show Peaks'], [True, True])
    
    btn_load = Button(ax_button_load, 'Load CSV', hovercolor='0.9')

    # --- Logic Wrappers ---
    
    def get_quartile_colors(n):
        """Generate a color array (N, 4) mapping sorted points to Q1-Q4 colors"""
        colors = np.zeros((n, 4)) # RGBA
        q_len = n // 4
        # Handle remainders by giving them to Q4 or distributing
        # Simple distribution:
        indices = np.arange(n)
        
        # Assign colors based on sorted position
        for i in range(4):
            start = int((i * n) / 4)
            end = int(((i + 1) * n) / 4)
            if i == 3: end = n # Ensure we catch all
            
            # Matplotlib colors to RGBA
            c_rgba = plt.cm.colors.to_rgba(Q_COLORS[i])
            colors[start:end] = c_rgba
            
        return colors

    def update(val):
        # 1. Get Params
        sigma = slider_bw.val
        angle = slider_rot.val + slider_rot_fine.val
        do_color = check_opts.get_status()[0]
        do_peaks = check_opts.get_status()[1]

        # 2. Transform Data
        rotated_data = rotate_data(data, angle)
        
        # 3. Sort Data by X for KDE and Quartiles
        # argsort gives us the indices to rearrange the data
        sort_idx = np.argsort(rotated_data[:, 0])
        data_sorted = rotated_data[sort_idx]
        x_sorted = data_sorted[:, 0]
        
        # 4. KDE Calculation
        grid_new, den_new = calculate_kde(x_sorted, sigma)
        
        # 5. Update Density Plot
        line_den.set_data(grid_new, den_new)
        rug_lines.set_data(x_sorted, np.zeros_like(x_sorted))
        
        # 6. Handle Peaks
        if do_peaks:
            # Find peaks with a minimum prominence relative to max density
            # height=0 means just local max, distance handles noise
            pk_idx, _ = find_peaks(den_new, height=np.max(den_new)*0.05, distance=10)
            peak_markers.set_data(grid_new[pk_idx], den_new[pk_idx])
            peak_markers.set_visible(True)
        else:
            peak_markers.set_visible(False)

        # 7. Update Scatter Plot
        # We plot the SORTED rotated data so coloring is linear (Q1->Q4 left to right)
        scat_plot.set_offsets(data_sorted)
        
        if do_color:
            c_array = get_quartile_colors(n_points)
            scat_plot.set_facecolors(c_array)
        else:
            scat_plot.set_facecolors('purple')

        # 8. Stats & Divider Lines
        q_inds = np.linspace(0, n_points, 5, dtype=int)
        boundaries = []
        stats_res = [] # Store (std, avg_pk, max_pk) tuples

        for i in range(4):
            # Get data chunk
            sub_x = x_sorted[q_inds[i]:q_inds[i+1]]
            
            # Boundary line (right side of this chunk, except for last)
            if i < 3: 
                bx = x_sorted[q_inds[i+1]]
                boundaries.append(bx)
                vlines[i].set_xdata([bx, bx])
            
            # Calc Stats
            if len(sub_x) > 0:
                g_min, g_max = sub_x[0], sub_x[-1]
                mask = (grid_new >= g_min) & (grid_new <= g_max)
                local_den = den_new[mask]
                if len(local_den) > 0:
                    s_std = np.std(local_den)
                    s_max = np.max(local_den)
                    s_avg = np.mean(local_den[local_den >= 0.9*s_max])
                else:
                    s_std = s_max = s_avg = 0
            else:
                s_std = s_max = s_avg = 0
            stats_res.append((s_std, s_avg, s_max))

        # 9. Update Text
        h_str = "Group    |    Q1 (Blue)   |   Q2 (Green)   |  Q3 (Orange)   |    Q4 (Red)    "
        r1 = "Std Den  | " + " | ".join([f"{s[0]*scale_factor:12.4f}" for s in stats_res])
        r2 = "Avg PkDen| " + " | ".join([f"{s[1]*scale_factor:12.4f}" for s in stats_res])
        r3 = "Max PkDen| " + " | ".join([f"{s[2]*scale_factor:12.4f}" for s in stats_res])
        
        stats_text.set_text(f"SUMMARY (x{scale_factor})\n{h_str}\n{'-'*75}\n{r1}\n{r2}\n{r3}")
        
        # Rescale Y axis slightly
        ax_den.set_xlim(grid_new[0], grid_new[-1])
        ax_den.set_ylim(0, np.max(den_new) * 1.1)
        
        fig.canvas.draw_idle()

    # --- Callback Assignments ---
    def inc_rot(e): slider_rot.set_val(slider_rot.val + 1.0)
    def dec_rot(e): slider_rot.set_val(slider_rot.val - 1.0)
    
    def inc_fine(e):
        v = slider_rot_fine.val + 0.05
        if v > 1.0 + 1e-9:
            slider_rot.set_val(slider_rot.val + 1.0)
            slider_rot_fine.set_val(v - 1.0)
        else: slider_rot_fine.set_val(v)
        
    def dec_fine(e):
        v = slider_rot_fine.val - 0.05
        if v < -1.0 - 1e-9:
            slider_rot.set_val(slider_rot.val - 1.0)
            slider_rot_fine.set_val(v + 1.0)
        else: slider_rot_fine.set_val(v)

    def load_file(event):
        if not HAS_TK:
            print("Tkinter not available. Cannot open file dialog.")
            return
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Bring to front
        file_path = filedialog.askopenfilename(filetypes=[("CSV/Text", "*.csv;*.txt;*.dat")])
        root.destroy()
        
        if file_path:
            try:
                # Attempt simple load
                new_data = np.loadtxt(file_path, delimiter=',')
            except:
                try:
                    new_data = np.loadtxt(file_path) # Try whitespace
                except Exception as e:
                    print(f"Error loading file: {e}")
                    return
            
            if new_data.shape[1] < 2:
                print("Data must have at least 2 columns (X, Y).")
                return
                
            print(f"Loaded {len(new_data)} points from {os.path.basename(file_path)}")
            plt.close(fig) # Close current
            plot_kde_interactive(new_data[:, :2]) # Restart with new data

    btn_rot_inc.on_clicked(inc_rot)
    btn_rot_dec.on_clicked(dec_rot)
    btn_fine_inc.on_clicked(inc_fine)
    btn_fine_dec.on_clicked(dec_fine)
    slider_bw.on_changed(update)
    slider_rot.on_changed(update)
    slider_rot_fine.on_changed(update)
    check_opts.on_clicked(update)
    btn_load.on_clicked(load_file)

    # Initial Trigger
    update(0)
    plt.show()

def generate_demo_data():
    # Two gaussian blobs and a line
    c1 = np.random.normal(loc=[20, 20], scale=5, size=(100, 2))
    c2 = np.random.normal(loc=[60, 50], scale=8, size=(150, 2))
    
    # A line structure
    x_l = np.linspace(10, 80, 50)
    y_l = -0.5 * x_l + 80 + np.random.normal(0, 1, 50)
    line = np.column_stack([x_l, y_l])
    
    data = np.vstack([c1, c2, line])
    return data

if __name__ == "__main__":
    print("Starting PET KDE Extended...")
    data = generate_demo_data()
    plot_kde_interactive(data, bw=3)