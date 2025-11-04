import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def cubic_arc_segment(start_deg=0, end_deg=90):
    """
    Return a cubic Bezier Path approximating a circular arc
    between start_deg and end_deg (degrees). Works best for spans <= 90deg.
    """
    a0, a1 = np.radians(start_deg), np.radians(end_deg)
    delta = a1 - a0
    
    # Check for absolute span, as this works for negative (clockwise) arcs too
    if abs(delta) > np.pi / 2 + 0.001: # Add small tolerance
        raise ValueError(
            f"Span too large ({np.degrees(delta):.1f} deg) for single cubic Bezier; "
            "split into <=90deg segments."
        )

    # 4/3 * tan(delta/4) gives the correct control handle length
    # This formula works for both positive and negative delta
    t = 4 / 3 * np.tan(delta / 4)

    # Start, end, and control points
    P0 = [np.cos(a0), np.sin(a0)]
    P3 = [np.cos(a1), np.sin(a1)]
    P1 = [np.cos(a0) - t * np.sin(a0), np.sin(a0) + t * np.cos(a0)]
    P2 = [np.cos(a1) + t * np.sin(a1), np.sin(a1) - t * np.cos(a1)]

    verts = [P0, P1, P2, P3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def create_arbitrary_arc(start_deg, end_deg, max_segment_deg=90):
    """
    Generates a Matplotlib Path for a circular arc from start_deg to end_deg.
    
    This function splits the arc into multiple cubic_arc_segment calls
    to handle spans greater than 90 degrees or clockwise arcs.
    """
    # Ensure inputs are floats
    start_deg = float(start_deg)
    end_deg = float(end_deg)
    
    total_span = end_deg - start_deg
    
    # If no arc, return an empty path
    if np.isclose(total_span, 0):
         return Path([], [])

    # Determine step direction (clockwise or counter-clockwise)
    if total_span > 0:
        step = float(max_segment_deg)
    else:
        step = -float(max_segment_deg)

    all_verts = []
    all_codes = []
    is_first_segment = True
    
    current_deg = start_deg
    
    while True:
        # Determine the end angle for this segment
        if total_span > 0: # Counter-clockwise
            next_deg = min(current_deg + step, end_deg)
        else: # Clockwise
            next_deg = max(current_deg + step, end_deg)

        # Create the path for this small segment
        segment_path = cubic_arc_segment(current_deg, next_deg)
        v, c = segment_path.vertices, segment_path.codes
        
        if is_first_segment:
            # For the first segment, include the MOVETO
            all_verts.extend(v)
            all_codes.extend(c)
            is_first_segment = False
        else:
            # For subsequent segments, skip the first vertex (it's a duplicate)
            # and skip the MOVETO code to create a single continuous path.
            all_verts.extend(v[1:])
            all_codes.extend(c[1:])

        # Check if we've reached the end
        if np.isclose(next_deg, end_deg):
            break
            
        current_deg = next_deg
        
    return Path(all_verts, all_codes)


# ===================================================================
# ---                         Demo Plot                           ---
# ===================================================================

fig, ax = plt.subplots(figsize=(6, 6))

# 1. A simple arc (<= 90 deg)
#arc_1 = create_arbitrary_arc(0, 45)
#ax.add_patch(PathPatch(arc_1, edgecolor="blue", lw=2, facecolor="none", label="0 to 45 deg"))

# 2. An arc that spans > 90 deg
arc_2 = create_arbitrary_arc(-30, 150)
ax.add_patch(PathPatch(arc_2, edgecolor="red", lw=2, facecolor="none", label="30 to 150 deg"))

# 3. A clockwise (negative) arc
#arc_3 = create_arbitrary_arc(90, -90)
#ax.add_patch(PathPatch(arc_3, edgecolor="green", lw=2, facecolor="none", label="90 to -90 deg"))

# 4. A multi-revolution "spiral" arc
# (We scale this one down so it's visible)
# arc_4_path = create_arbitrary_arc(0, 450)
# arc_4_verts = arc_4_path.vertices * 0.5 # Scale down by 50%
# arc_4 = Path(arc_4_verts, arc_4_path.codes)
# ax.add_patch(PathPatch(arc_4, edgecolor="purple", lw=2, facecolor="none", label="0 to 450 deg (scaled)"))


# --- Plot settings ---
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
plt.title("Using create_arbitrary_arc()")
plt.show()