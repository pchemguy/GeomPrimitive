"""
pet_kde_optimizer_v2.py
-----------------------
Interactive KDE tool with Auto-Optimization capabilities.

Fixes:
- Adjusted GUI layout to prevent Rotation buttons from overlapping the angle text values.
- Removed deprecated Matplotlib attributes for stability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
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
    # --- Data Validation ---
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data must be Nx2.")
    
    n_points = len(data)
    if n_points < 4:
        print("Need at least 4 points.")
        return

    # --- Constants ---
    scale_exp = np.ceil(np.log10(n_points))
    scale_factor = 10 ** int(scale_exp)
    Q_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'] 
    
    # Geometry for axis limits
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    center = (min_vals + max_vals) / 2
    max_dist = np.max(np.linalg.norm(data - center, axis=1))
    limit_padding = max_dist * 1.2
    
    xlim_scat = (center[0] - limit_padding, center[0] + limit_padding)
    ylim_scat = (center[1] - limit_padding, center[1] + limit_padding)

    # --- Math Helpers ---
    def get_rotated_points(points, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        centered = points - center
        return (centered @ R.T) + center

    def compute_kde_and_grid(x_sorted, sigma):
        n = len(x_sorted)
        span = x_sorted[-1] - x_sorted[0]
        if span == 0: span = 1.0 
        pad = span * 0.2
        grid = np.linspace(x_sorted[0] - pad, x_sorted[-1] + pad, 500)
        
        sigma = max(1e-5, sigma)
        pdfs = norm.pdf(grid[:, None], loc=x_sorted[None, :], scale=sigma)
        y_den = np.sum(pdfs, axis=1) / n
        return grid, y_den

    def calculate_quartile_std(angle, target_q_idx, sigma):
        # 1. Rotate
        rot_pts = get_rotated_points(data, angle)
        # 2. Sort
        x_sorted = np.sort(rot_pts[:, 0])
        # 3. KDE
        grid, den = compute_kde_and_grid(x_sorted, sigma)
        # 4. Slice Quartile
        q_inds = np.linspace(0, len(x_sorted), 5, dtype=int)
        idx_start = q_inds[target_q_idx]
        idx_end = q_inds[target_q_idx + 1]
        
        sub_x = x_sorted[idx_start:idx_end]
        if len(sub_x) == 0: return 0.0

        g_min, g_max = sub_x[0], sub_x[-1]
        mask = (grid >= g_min) & (grid <= g_max)
        local_den = den[mask]
        
        if len(local_den) == 0: return 0.0
        return np.std(local_den)

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

    # Plot Objects
    scat_plot = ax_scat.scatter([], [], alpha=0.6, edgecolors='w', s=40)
    line_den, = ax_den.plot([], [], color='k', lw=2, label='Density')
    rug_lines, = ax_den.plot([], [], '|', color='gray', alpha=0.3)
    peak_markers, = ax_den.plot([], [], 'x', color='red', markeredgewidth=2, markersize=8)
    vlines = [ax_den.axvline(x=0, color=c, linestyle='--', alpha=0.8, lw=1.5) for c in Q_COLORS[:-1]]

    ax_scat.set_title("Rotated Point Cloud")
    ax_scat.set_xlim(xlim_scat)
    ax_scat.set_ylim(ylim_scat)
    ax_scat.grid(True, linestyle='--', alpha=0.4)
    ax_scat.set_aspect('equal', adjustable='box')

    ax_den.set_title("Marginal Density (X-Projection)")
    ax_den.set_ylabel("Density")
    ax_den.grid(True, alpha=0.3)

    # --- Controls Layout (FIXED COORDINATES) ---
    
    # Left: Rotation Sliders (0.10 to 0.35)
    ax_slider_rot = plt.axes([0.10, 0.10, 0.25, 0.03])
    ax_slider_rot_fine = plt.axes([0.10, 0.06, 0.25, 0.03])

    # Left-Middle: Rotation Spinner Buttons (Moved to 0.42 to clear text)
    # [Left, Bottom, Width, Height]
    ax_rot_inc = plt.axes([0.42, 0.115, 0.02, 0.015])
    ax_rot_dec = plt.axes([0.42, 0.100, 0.02, 0.015])
    
    ax_fine_inc = plt.axes([0.42, 0.075, 0.02, 0.015])
    ax_fine_dec = plt.axes([0.42, 0.060, 0.02, 0.015])

    # Center-Right: Sigma & Load
    ax_slider_bw = plt.axes([0.55, 0.10, 0.15, 0.03])
    ax_button_load = plt.axes([0.55, 0.05, 0.08, 0.04]) 
    ax_check_opts = plt.axes([0.65, 0.04, 0.10, 0.06]) 

    # Far Right: Optimization Panel
    ax_panel_bg = plt.axes([0.78, 0.02, 0.18, 0.12]) 
    ax_panel_bg.axis('off')
    ax_panel_bg.text(0.5, 0.9, "Optimization", ha='center', transform=ax_panel_bg.transAxes, fontsize=9, weight='bold')
    
    ax_opt_radio = plt.axes([0.80, 0.03, 0.06, 0.08])
    ax_opt_btn = plt.axes([0.87, 0.05, 0.08, 0.05])

    # --- Widgets ---
    slider_rot = Slider(ax_slider_rot, 'Rot', -100, 100, valinit=0)
    slider_rot_fine = Slider(ax_slider_rot_fine, 'Fine', -1.0, 1.0, valinit=0)
    
    btn_rot_inc = Button(ax_rot_inc, '>', hovercolor='0.9')
    btn_rot_dec = Button(ax_rot_dec, '<', hovercolor='0.9')
    btn_fine_inc = Button(ax_fine_inc, '>', hovercolor='0.9')
    btn_fine_dec = Button(ax_fine_dec, '<', hovercolor='0.9')
    
    sigma_max = max(2, int(np.ceil(n_points * 0.01)))
    slider_bw = Slider(ax_slider_bw, 'Sigma', 1.0, sigma_max, valinit=bw)
    
    check_opts = CheckButtons(ax_check_opts, ['Color', 'Peaks'], [True, True])
    btn_load = Button(ax_button_load, 'Load', hovercolor='0.9')

    # Optimization Widgets
    radio_opt = RadioButtons(ax_opt_radio, ('Q1', 'Q2', 'Q3', 'Q4'), active=0)
    btn_opt = Button(ax_opt_btn, 'Opt +/-10deg', color='lightblue', hovercolor='skyblue')

    # --- Core Logic ---
    def get_quartile_colors(n):
        colors = np.zeros((n, 4)) 
        for i in range(4):
            start = int((i * n) / 4)
            end = int(((i + 1) * n) / 4)
            if i == 3: end = n 
            colors[start:end] = plt.cm.colors.to_rgba(Q_COLORS[i])
        return colors

    def update(val):
        sigma = slider_bw.val
        angle = slider_rot.val + slider_rot_fine.val
        do_color, do_peaks = check_opts.get_status()

        # Calculation
        rotated_data = get_rotated_points(data, angle)
        sort_idx = np.argsort(rotated_data[:, 0])
        data_sorted = rotated_data[sort_idx]
        x_sorted = data_sorted[:, 0]
        
        grid_new, den_new = compute_kde_and_grid(x_sorted, sigma)
        
        # Plot Density
        line_den.set_data(grid_new, den_new)
        rug_lines.set_data(x_sorted, np.zeros_like(x_sorted))
        
        # Plot Scatter
        scat_plot.set_offsets(data_sorted)
        if do_color:
            scat_plot.set_facecolors(get_quartile_colors(n_points))
        else:
            scat_plot.set_facecolors('purple')

        # Peaks
        if do_peaks:
            pk_idx, _ = find_peaks(den_new, height=np.max(den_new)*0.05, distance=10)
            peak_markers.set_data(grid_new[pk_idx], den_new[pk_idx])
            peak_markers.set_visible(True)
        else:
            peak_markers.set_visible(False)

        # Stats & Boundaries
        q_inds = np.linspace(0, n_points, 5, dtype=int)
        stats_res = []

        for i in range(4):
            sub_x = x_sorted[q_inds[i]:q_inds[i+1]]
            if i < 3: 
                bx = x_sorted[q_inds[i+1]]
                vlines[i].set_xdata([bx, bx])
            
            # Recalc stats for display
            if len(sub_x) > 0:
                g_min, g_max = sub_x[0], sub_x[-1]
                mask = (grid_new >= g_min) & (grid_new <= g_max)
                local_den = den_new[mask]
                if len(local_den) > 0:
                    s_std = np.std(local_den)
                    s_max = np.max(local_den)
                    s_avg = np.mean(local_den[local_den >= 0.9*s_max])
                else: s_std = s_max = s_avg = 0
            else: s_std = s_max = s_avg = 0
            stats_res.append((s_std, s_avg, s_max))

        # Text Update
        h_str = "Group    |    Q1 (Blue)   |   Q2 (Green)   |  Q3 (Orange)   |    Q4 (Red)    "
        r1 = "Std Den  | " + " | ".join([f"{s[0]*scale_factor:12.4f}" for s in stats_res])
        r2 = "Avg PkDen| " + " | ".join([f"{s[1]*scale_factor:12.4f}" for s in stats_res])
        r3 = "Max PkDen| " + " | ".join([f"{s[2]*scale_factor:12.4f}" for s in stats_res])
        stats_text.set_text(f"SUMMARY (x{scale_factor})\n{h_str}\n{'-'*75}\n{r1}\n{r2}\n{r3}")
        
        ax_den.set_xlim(grid_new[0], grid_new[-1])
        ax_den.set_ylim(0, np.max(den_new) * 1.1)
        fig.canvas.draw_idle()

    # --- Callbacks ---
    def run_optimization(event):
        current_angle = slider_rot.val + slider_rot_fine.val
        sigma = slider_bw.val
        
        q_label = radio_opt.value_selected
        q_idx = int(q_label[1]) - 1 
        
        print(f"Optimizing for {q_label} (Max Std Dev) within +/- 10 degrees...")
        btn_opt.label.set_text("Busy...")
        fig.canvas.draw()
        
        def objective(a):
            return -1 * calculate_quartile_std(a, q_idx, sigma)
        
        b_min = max(-100, current_angle - 10)
        b_max = min(100, current_angle + 10)
        
        res = minimize_scalar(objective, bounds=(b_min, b_max), method='bounded')
        
        best_angle = res.x
        final_score = -res.fun * scale_factor
        
        print(f" Optimization Complete: Found angle {best_angle:.2f} deg (Score: {final_score:.4f})")
        
        coarse = int(np.round(best_angle))
        fine = best_angle - coarse
        
        if fine > 1.0: 
            coarse += 1; fine -= 1.0
        elif fine < -1.0: 
            coarse -= 1; fine += 1.0
            
        slider_rot.set_val(coarse)
        slider_rot_fine.set_val(fine)
        btn_opt.label.set_text("Opt +/-10deg")

    def load_file(event):
        if not HAS_TK: 
            print("Tkinter not available. Cannot open file dialog.")
            return
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        fp = filedialog.askopenfilename(filetypes=[("CSV/Text", "*.csv;*.txt;*.dat")])
        root.destroy()
        if fp:
            try:
                nd = np.loadtxt(fp, delimiter=',')
            except:
                try: nd = np.loadtxt(fp)
                except Exception as e: 
                    print(f"Failed to load: {e}")
                    return
            print(f"Loaded {len(nd)} points.")
            plt.close(fig)
            plot_kde_interactive(nd[:, :2])

    slider_bw.on_changed(update)
    slider_rot.on_changed(update)
    slider_rot_fine.on_changed(update)
    check_opts.on_clicked(update)
    btn_load.on_clicked(load_file)
    btn_opt.on_clicked(run_optimization)
    
    # Helper buttons
    def i_r(e): slider_rot.set_val(slider_rot.val + 1)
    def d_r(e): slider_rot.set_val(slider_rot.val - 1)
    def i_f(e): 
        v = slider_rot_fine.val + 0.05
        if v > 1: slider_rot.set_val(slider_rot.val+1); slider_rot_fine.set_val(v-1)
        else: slider_rot_fine.set_val(v)
    def d_f(e):
        v = slider_rot_fine.val - 0.05
        if v < -1: slider_rot.set_val(slider_rot.val-1); slider_rot_fine.set_val(v+1)
        else: slider_rot_fine.set_val(v)
        
    btn_rot_inc.on_clicked(i_r)
    btn_rot_dec.on_clicked(d_r)
    btn_fine_inc.on_clicked(i_f)
    btn_fine_dec.on_clicked(d_f)

    update(0)
    plt.show()

def generate_demo_data():
    c1 = np.random.normal([30, 50], [4, 15], (200, 2)) 
    c2 = np.random.normal([70, 50], [4, 15], (200, 2))
    x = np.linspace(20, 80, 100)
    y = x + np.random.normal(0, 2, 100)
    line = np.column_stack([x, y])
    return np.vstack([c1, c2, line])

if __name__ == "__main__":
    data = generate_demo_data()
    plot_kde_interactive(data, bw=2)
