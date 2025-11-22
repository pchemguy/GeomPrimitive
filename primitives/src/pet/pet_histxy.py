"""
```
pet_histxy.py
-------------
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def plot_interactive_histogram(data, axis='x', bin_size=None, alpha=0.0, title="Interactive Analysis", color='skyblue'):
    """
    Creates a three-panel interactive tool:
    - Left: XY Scatter plot (showing rotation).
    - Center: Standard Histogram (projected data).
    - Right: Sorted Histogram (descending frequency) with Frequency Stats.
    
    Includes dynamic bin sizing and rotation sliders.
    
    Args:
        data (np.array): (N, 2) array of points or 1D array.
        axis (str): Initial axis ('x' or 'y').
        bin_size (float): Optional initial bin size.
        alpha (float): Initial rotation angle in degrees (CCW).
        title (str): Plot title.
        color (str): Bar color.
    """
    # --- 1. Data Preparation ---
    original_data = np.array(data)
    is_2d = (original_data.ndim == 2 and original_data.shape[1] >= 2)
    
    # Determine pivot point
    if is_2d:
        min_vals = np.min(original_data, axis=0)
        max_vals = np.max(original_data, axis=0)
        pivot_center = (min_vals + max_vals) / 2
    else:
        pivot_center = np.array([0, 0])

    # Local state container
    state = {
        'rotated_points': None, 
        'hist_data': None,      
        'xlabel': "",
        'axis_key': axis.lower() if is_2d else 'x',
        'alpha': float(alpha) if is_2d else 0.0,
        'current_bin_size': bin_size 
    }

    slider_bin = None

    # --- Helper: Rotation ---
    def get_rotated_data(points, angle_deg):
        if not is_2d: return points
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        
        centered = points - pivot_center
        x = centered[:, 0]
        y = centered[:, 1]
        
        x_rot = x * c - y * s
        y_rot = x * s + y * c
        
        return np.column_stack((x_rot, y_rot)) + pivot_center

    # --- Helper: Pipeline ---
    def refresh_data_state():
        if is_2d:
            rotated = get_rotated_data(original_data, state['alpha'])
            state['rotated_points'] = rotated
        else:
            state['rotated_points'] = original_data
            
        if is_2d:
            if state['axis_key'] == 'x':
                vals = state['rotated_points'][:, 0]
                lbl = f"X Coordinate (Rotated {state['alpha']:.1f}deg)"
            else:
                vals = state['rotated_points'][:, 1]
                lbl = f"Y Coordinate (Rotated {state['alpha']:.1f}deg)"
        else:
            vals = original_data.flatten()
            lbl = "Value"
            
        clean = vals[~np.isnan(vals)]
        state['hist_data'] = clean
        state['xlabel'] = lbl

    refresh_data_state()

    if len(state['hist_data']) == 0:
        print("No valid data to plot.")
        return

    def get_data_stats(clean_data):
        d_min = np.floor(clean_data.min())
        d_max = np.ceil(clean_data.max())
        d_range = d_max - d_min
        return d_min, d_max, d_range

    # --- 3. Plot Setup (3 Panels) ---
    fig, (ax_scatter, ax_hist, ax_sorted) = plt.subplots(1, 3, figsize=(18, 7))
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.25, left=0.10, right=0.95, wspace=0.25)

    def draw_stats(ax_target, vals):
        """Stats for original data values (Vertical lines)"""
        mean_val = np.mean(vals)
        median_val = np.median(vals)
        std_dev = np.std(vals)
        
        ax_target.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax_target.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
        
        stats_text = (f"Count: {len(vals)}\n"
                      f"Mean: {mean_val:.2f}\n"
                      f"Median: {median_val:.2f}\n"
                      f"StdDev: {std_dev:.2f}")
        
        ax_target.text(0.95, 0.95, stats_text, transform=ax_target.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_target.legend(loc='upper left')

    def draw_stats_y(ax_target, counts):
        """Stats for bin frequencies (Horizontal lines)"""
        # Calculate stats on the bin counts themselves
        mean_freq = np.mean(counts)
        median_freq = np.median(counts)
        std_freq = np.std(counts)
        
        ax_target.axhline(mean_freq, color='red', linestyle='dashed', linewidth=2, label=f'Mean Freq: {mean_freq:.2f}')
        ax_target.axhline(median_freq, color='green', linestyle='dashed', linewidth=2, label=f'Med Freq: {median_freq:.2f}')
        
        stats_text = (f"Bin Count: {len(counts)}\n"
                      f"Mean Freq: {mean_freq:.2f}\n"
                      f"Med Freq: {median_freq:.2f}\n"
                      f"StdDev: {std_freq:.2f}")
        
        # Place text top right (usually empty for descending chart)
        ax_target.text(0.95, 0.95, stats_text, transform=ax_target.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_target.legend(loc='center right')

    # --- 4. Update Plot Function ---
    def update_plot():
        # --- A. SCATTER PLOT (Left) ---
        ax_scatter.cla()
        pts = state['rotated_points']
        
        if is_2d:
            ax_scatter.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=10, color='gray')
            ax_scatter.plot(pivot_center[0], pivot_center[1], 'rx', markersize=10, label='Pivot')
            ax_scatter.set_aspect('equal', 'datalim')
            ax_scatter.set_title(f"Source Data (Rotated {state['alpha']:.1f}deg)")
            ax_scatter.set_xlabel("X"); ax_scatter.set_ylabel("Y")
            ax_scatter.grid(True, alpha=0.3)
        else:
            ax_scatter.text(0.5, 0.5, "1D Data - No Scatter", ha='center')

        # --- B. STANDARD HISTOGRAM (Center) ---
        ax_hist.cla()
        vals = state['hist_data']
        
        d_min, d_max, d_range = get_data_stats(vals)
        
        if d_range == 0:
            return

        # --- Dynamic Slider Logic ---
        bin_min = d_range * 0.01
        bin_max = d_range * 0.10
        
        current_b = state['current_bin_size']
        if current_b is None: current_b = (bin_min + bin_max) / 2
        
        if current_b < bin_min: current_b = bin_min
        if current_b > bin_max: current_b = bin_max
        state['current_bin_size'] = current_b
        
        if slider_bin is not None:
            slider_bin.eventson = False
            slider_bin.valmin = bin_min; slider_bin.valmax = bin_max
            slider_bin.set_val(current_b); slider_bin.ax.set_xlim(bin_min, bin_max)
            slider_bin.eventson = True

        # Calculate Histogram Data manually to reuse for sorting
        bins = np.arange(d_min, d_max + current_b, current_b)
        counts, edges = np.histogram(vals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        # Plot Standard
        ax_hist.bar(centers, counts, width=current_b*0.9, color=color, edgecolor='black', alpha=0.7)
        draw_stats(ax_hist, vals)
        
        ax_hist.set_title(f"Projection: {state['axis_key'].upper()}-Axis\n(Bin Size: {current_b:.2f})")
        ax_hist.set_xlabel(state['xlabel'])
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(axis='y', linestyle='--', alpha=0.3)

        # --- C. SORTED HISTOGRAM (Right) ---
        ax_sorted.cla()
        
        # Sort counts descending
        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        
        # Plot bars ranked by frequency
        # X-axis is just rank 0, 1, 2...
        ranks = np.arange(len(sorted_counts))
        ax_sorted.bar(ranks, sorted_counts, width=0.8, color=color, edgecolor='black', alpha=0.7)
        
        # Draw stats based on FREQUENCIES (horizontal lines)
        draw_stats_y(ax_sorted, counts)
        
        ax_sorted.set_title("Sorted Distribution (Pareto)")
        ax_sorted.set_xlabel("Bin Rank (Desc. Freq)")
        ax_sorted.set_ylabel("Frequency")
        ax_sorted.grid(axis='y', linestyle='--', alpha=0.3)
        
        if len(ranks) > 50:
            ax_sorted.set_xlim(-0.5, 50.5)
            ax_sorted.text(0.5, -0.15, "(Showing top 50 bins)", transform=ax_sorted.transAxes, ha='center')

        fig.canvas.draw_idle()

    # --- 5. Widgets ---
    
    # A. Bin Size Slider
    ax_bin = plt.axes([0.20, 0.1, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    _, _, init_rng = get_data_stats(state['hist_data'])
    init_b_min = init_rng * 0.01; init_b_max = init_rng * 0.10
    init_b_val = state['current_bin_size'] if state['current_bin_size'] else (init_b_min + init_b_max)/2

    slider_bin = Slider(ax_bin, 'Bin Size ', init_b_min, init_b_max, valinit=init_b_val, valfmt='%0.2f')
    slider_bin.on_changed(lambda val: [state.update({'current_bin_size': val}), update_plot()])

    # B. Alpha Slider (Only if 2D)
    if is_2d:
        ax_alpha = plt.axes([0.20, 0.05, 0.70, 0.03], facecolor='#e6e6fa')
        slider_alpha = Slider(ax_alpha, 'Rotation (deg) ', -90.0, 90.0, valinit=state['alpha'], valfmt='%0.1f')
        slider_alpha.on_changed(lambda val: [state.update({'alpha': val}), refresh_data_state(), update_plot()])

    # C. Radio Buttons (Only if 2D)
    if is_2d:
        ax_radio = plt.axes([0.02, 0.4, 0.06, 0.15], facecolor='#f0f0f0')
        radio = RadioButtons(ax_radio, ('X', 'Y'), active=(0 if state['axis_key'] == 'x' else 1))
        radio.on_clicked(lambda label: [state.update({'axis_key': label.lower()}), refresh_data_state(), update_plot()])

    update_plot()
    
    print("Three-panel interactive plot launched.")
    if is_2d: print(f"Rotation Pivot: {pivot_center}")
    plt.show()

# --- Usage Example ---
if __name__ == "__main__":
    # Generate 2D data (Cross shape)
    line1_x = np.random.normal(50, 2, 500)
    line1_y = np.random.normal(50, 20, 500)
    line2_x = np.random.normal(50, 20, 500)
    line2_y = np.random.normal(50, 2, 500)
    pts = np.vstack([np.column_stack((line1_x, line1_y)), np.column_stack((line2_x, line2_y))])
    
    print("Testing Three-Panel Layout...")
    plot_interactive_histogram(pts, axis='x', alpha=0.0, title="Cross Distribution Analysis")


"""
```
"""
