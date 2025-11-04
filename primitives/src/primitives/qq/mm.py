import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def cubic_arc_segment(start_deg=0, 
                      end_deg=90, 
                      hand_drawn: bool = False, 
                      jitter_scale: float = 0.01):
    """
    Return a cubic Bezier Path approximating a circular arc
    between start_deg and end_deg (degrees). Works best for spans <= 90deg.
    
    If hand_drawn is True, adds random jitter to the control and end points.
    """
    a0, a1 = np.radians(start_deg), np.radians(end_deg)
    delta = a1 - a0
    
    # Check for absolute span, as this works for negative (clockwise) arcs too
    if abs(delta) > np.pi / 2 + 0.001: # Add small tolerance
        raise ValueError(
            f"Span too large ({np.degrees(delta):.1f} deg) for single cubic Bezier; "
            "split into <=90deg segments."
        )

    t = 4 / 3 * np.tan(delta / 4)

    # Start, end, and control points
    P0 = [np.cos(a0), np.sin(a0)]
    P3 = [np.cos(a1), np.sin(a1)]
    P1 = [np.cos(a0) - t * np.sin(a0), np.sin(a0) + t * np.cos(a0)]
    P2 = [np.cos(a1) + t * np.sin(a1), np.sin(a1) - t * np.cos(a1)]

    verts = [P0, P1, P2, P3]

    # --- NEW: Add jitter if hand_drawn is True ---
    if hand_drawn and jitter_scale > 0:
        # We DON'T jitter P0, so the arc stays connected to the previous segment.
        
        # Jitter P1 (control point 1)
        jitter_p1 = np.random.normal(0.0, jitter_scale, 2)
        P1 = [P1[0] + jitter_p1[0], P1[1] + jitter_p1[1]]
        
        # Jitter P2 (control point 2)
        jitter_p2 = np.random.normal(0.0, jitter_scale, 2)
        P2 = [P2[0] + jitter_p2[0], P2[1] + jitter_p2[1]]

        # Jitter P3 (end point)
        jitter_p3 = np.random.normal(0.0, jitter_scale, 2)
        P3 = [P3[0] + jitter_p3[0], P3[1] + jitter_p3[1]]
        
        # Update the vertices list with jittered points
        verts = [P0, P1, P2, P3]
    # --- End of new code ---

    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def create_arbitrary_arc(start_deg, 
                         end_deg, 
                         max_segment_deg=90, 
                         hand_drawn: bool = False, 
                         jitter_scale: float = 0.01):
    """
    Generates a Matplotlib Path for a circular arc from start_deg to end_deg.
    
    Passes hand_drawn and jitter_scale parameters to the segment generator.
    """
    start_deg = float(start_deg)
    end_deg = float(end_deg)
    
    total_span = end_deg - start_deg
    
    if np.isclose(total_span, 0):
         return Path([], [])

    if total_span > 0:
        step = float(max_segment_deg)
    else:
        step = -float(max_segment_deg)

    all_verts = []
    all_codes = []
    is_first_segment = True
    
    current_deg = start_deg
    
    while True:
        if total_span > 0:
            next_deg = min(current_deg + step, end_deg)
        else:
            next_deg = max(current_deg + step, end_deg)

        # --- MODIFIED: Pass parameters down ---
        segment_path = cubic_arc_segment(
            current_deg, 
            next_deg,
            hand_drawn=hand_drawn,
            jitter_scale=jitter_scale
        )
        # --- End of modification ---

        v, c = segment_path.vertices, segment_path.codes
        
        if is_first_segment:
            all_verts.extend(v)
            all_codes.extend(c)
            is_first_segment = False
        else:
            all_verts.extend(v[1:])
            all_codes.extend(c[1:])

        if np.isclose(next_deg, end_deg):
            break
            
        current_deg = next_deg
        
    return Path(all_verts, all_codes)


# ===================================================================
# ---                         Demo Plot                           ---
# ===================================================================

fig, ax = plt.subplots(figsize=(6, 6))

# 1. A simple smooth arc
#arc_1 = create_arbitrary_arc(0, 45)
#ax.add_patch(PathPatch(arc_1, edgecolor="blue", lw=2, facecolor="none", label="0 to 45 deg (Smooth)"))

# 2. A smooth arc > 90 deg
#arc_2 = create_arbitrary_arc(30, 150)
#ax.add_patch(PathPatch(arc_2, edgecolor="red", lw=2, facecolor="none", label="30 to 150 deg (Smooth)"))

# 3. A clockwise (negative) smooth arc
#arc_3 = create_arbitrary_arc(90, -90)
#ax.add_patch(PathPatch(arc_3, edgecolor="green", lw=2, facecolor="none", label="90 to -90 deg (Smooth)"))

# 4. A HAND-DRAWN version of the green arc
#    We use a larger jitter_scale (0.02) to make the effect obvious
arc_4 = create_arbitrary_arc(
    90, -90, 
    hand_drawn=True, 
    jitter_scale=0.02
)
ax.add_patch(PathPatch(arc_4, edgecolor="black", lw=2, ls="--", facecolor="none", label="90 to -90 deg (Hand-Drawn)"))


# --- Plot settings ---
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.legend(fontsize="small")
ax.grid(True, ls="--", alpha=0.5)
plt.title("Hand-Drawn vs. Smooth Arcs")
plt.show()
