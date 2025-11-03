import numpy as np
import random
import math
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath
from matplotlib.transforms import Affine2D


def elliptical_arc(hrange: tuple[float, float] = (0, 1023),
                   vrange: tuple[float, float] = (0, 1023),
                   start_deg: Optional[float] = None,
                   end_deg: Optional[float] = None,
                   aspect_ratio: Optional[float] = None,
                   angle_deg: Optional[int] = None,
                   jitter_amp: Optional[float] = 0.02,
                   jitter_aspect: float = 0.1,
                   jitter_angle_deg: int = 5,
                   max_angle_delta_deg: Optional[int] = 20,
                   min_angle_steps: Optional[int] = 3,
                  ) -> mplPath:
    """ Creates a generalized elliptical arc or an eelipse.

    The code first creates a unit cicular arc using piecewise cubic Bezier
    curves provided by Matplotlib. In priciple, a 90 deg arc can be approximated
    by a single cubic curve very well. However, to imitate hand drawing, smaller
    steps are used, set at the default value of 20 deg `max_angle_delta_deg`.

    For smaller arcs, the smallest number of sections is set to 3 (`min_angle_steps`).

    Once the unit arc or a full circle is created, Jitter is applied to individual
    points (magnitude controlled by `jitter_amp`), as well as to aspect ratio
    (`jitter_aspect` controls scaling of the y coordinates only. The latter is only
    usefull for creating non-ideal circular arcs, as elliptical transform will absorb
    this factor.

    The circular arc is scaled to yeild an ellipse, rotated (with angle jitter), and
    translated to yield the final generalized elliptical arc with jitter.
    """
    # 1. Create a unit circular arc using piecewise cubic Bezier curves.
    #
    if start_deg is None:
        start_deg = random.uniform(0, 270)
        end_deg = random.uniform(end_deg + 5, 360)
    delta_deg: float = end_deg - start_deg
    if delta_deg < 1 or delta_deg > 359:
        start_deg = 0
        end_deg = 360
        delta_deg = 360
        closed = True
    else:
        closed = False

    step_count: int = int(max(min_angle_steps, round(delta_deg / max_angle_delta_deg)))
    start: float = np.radians(start_deg)
    end: float = np.radians(end_deg)
    delta: float = end - start
    delta_step: float = delta / step_count
    t = 4 / 3 * np.tan(delta_step / 4)

    P0 = [float(np.cos(start)), float(np.sin(start))]
    verts = [P0]
    start_section = start
    end_section = start_section + delta_step
    for i in range(step_count):
        P1 = [float(np.cos(start_section) - t * np.sin(start_section)),
              float(np.sin(start_section) + t * np.cos(start_section))]
        P2 = [float(np.cos(end_section)   + t * np.sin(end_section)),
              float(np.sin(end_section)   - t * np.cos(end_section))]
        P3 = [float(np.cos(end_section)),
              float(np.sin(end_section))]
        verts.extend([P1, P2, P3])
        start_section += delta_step
        end_section += delta_step
    
    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * step_count
    if closed:
        codes.append(mplPath.CLOSEPOLY)
        verts.append(P0)

    # -------------------------
    # 2. Convert to NumPy array
    # -------------------------
    verts_array = np.array(verts)

    # -------------------------
    # 3. Apply Aspect Jitter (Multiplicative Scale)
    # -------------------------
    if jitter_aspect:
        # We apply one random scale to the y-axis of all vertices
        verts_array[:, 1] *= 1 - jitter_aspect * random.uniform(0, 1)  # Scales the entire y-column

    # -------------------------
    # 4. Apply Additive Jitter
    # -------------------------
    if jitter_amp:
        verts_array += np.random.uniform(-1, 1, size=verts_array.shape) * jitter_amp

    # -------------------------
    # 5. Scale.
    # -------------------------
    xmin, xmax = hrange
    ymin, ymax = vrange
    dx, dy = ymax - ymin, xmax - xmin
    print(f"dx: {dx}\ndy: {dy}")
    bbox_side = min(dx, dy) * (1 - random.uniform(0, 0.8))
    bbox_diag = bbox_side * math.sqrt(2)
    print(f"bbox_side: {bbox_side}\nbbox_diag: {bbox_diag}")

    if aspect_ratio is None or not isinstance(aspect_ratio, (float, int)):
        aspect_ratio = 1 - abs(random.normalvariate(0, 0.25))
    aspect_ratio = max(0.25, min(aspect_ratio, 1))

    x_scale = 0.5 * bbox_side
    y_scale = 0.5 * bbox_side * aspect_ratio
    verts_array *= [x_scale, y_scale]
    print(f"===== SCALE =====")
    print(f"[x_scale, y_scale]: {[x_scale, y_scale]}.")

    # -------------------------
    # 6. Rotate.
    # -------------------------
    if not isinstance(angle_deg, (int, float)):
        angle_deg = random.uniform(-90, 90)
    else:
        angle_deg += jitter_angle_deg * random.uniform(-1, 1)
    angle = np.radians(angle_deg)
    rotation_matrix = np.array([
        [+np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    verts_array @= rotation_matrix
    print(f"===== ROTATE =====")
    print(f"angle_deg: {angle_deg}.")
    
    # -------------------------
    # 6. Translate
    # -------------------------
    x0, y0 = (xmin + xmax) / 2, (ymin + ymax) / 2
    x_translate = x0 + max(0, 0.5 * (dx - bbox_diag)) * random.uniform(-1, 1)
    y_translate = y0 + max(0, 0.5 * (dy - bbox_diag)) * random.uniform(-1, 1)
    verts_array += [x_translate, y_translate]
    print(f"===== TRANSLATE =====")
    print(f"[x_translate, y_translate]: {[x_translate, y_translate]}.")

    # -------------------------
    # Note. Library-based SRT (Scale, Rotate, Translate)
    # -------------------------
    # trans = (
    #     Affine2D()
    #     .scale(x_scale, y_scale)
    #     .rotate_deg(angle_deg)
    #     .translate(x_translate, y_translate)
    # )
    # verts_array = trans.transform(verts_array)
    
    # -------------------------
    # 7. Create Path
    # -------------------------
    
    arc_path: mplPath = mplPath(verts_array, codes)

    return arc_path


hrange=(-10, 30)
vrange=(-10, 20)

arc = elliptical_arc(hrange=hrange, vrange=vrange, start_deg=0, end_deg=360, angle_deg=0)

fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(arc, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))

ax.set_aspect("equal")
ax.grid(True, ls="--", alpha=0.5)
ax.set_xlim(*hrange)
ax.set_ylim(*vrange)
plt.show()



def cubic_arc_segment(start_deg=0, end_deg=90, amp=0.02):
    """
    Return a cubic Bezier Path approximating a circular arc
    between start_deg and end_deg (degrees). Works best for spans <= 90deg.
    """
    a0, a1 = np.radians(start_deg), np.radians(end_deg)
    delta = a1 - a0
    if abs(delta) > np.pi / 2:
        raise ValueError("Span too large for single cubic Bezier; split into <=90deg segments.")

    # 4/3 * tan(delta/4) gives the correct control handle length
    t = 4 / 3 * np.tan(delta / 4)

    # Start, end, and control points
    P0 = (np.cos(a0), np.sin(a0))
    P3 = [np.cos(a1), np.sin(a1)]
    P1 = [np.cos(a0) - t * np.sin(a0) + random.uniform(-1, 1) * amp, np.sin(a0) + t * np.cos(a0) + random.uniform(-1, 1) * amp]
    P2 = [np.cos(a1) + t * np.sin(a1) + random.uniform(-1, 1) * amp, np.sin(a1) - t * np.cos(a1) + random.uniform(-1, 1) * amp]

    verts = [P0, P1, P2, P3]
    codes = [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]
    return mplPath(verts, codes)


def join_paths(path1: mplPath, path2: mplPath) -> mplPath:
    """
    Joins two paths into a single continuous path.
    Assumes the end of path1 is the start of path2.
    """
    # If path1 is empty, just return path2
    if not path1.vertices.size:
        return path2
    # If path2 is empty, just return path1
    if not path2.vertices.size:
        return path1

    # Get vertices and codes, skipping the first (MOVETO) from path2
    v1, c1 = path1.vertices, path1.codes
    v2, c2 = path2.vertices[1:], path2.codes[1:]

    # Concatenate them
    verts = np.concatenate((v1, v2))
    codes = np.concatenate((c1, c2))
    
    return mplPath(verts, codes)



#arc1 = cubic_arc_segment(0, 30, 0.04)  # any angle span <=90deg
#arc2 = cubic_arc_segment(30, 60, 0.04)
#arc3 = cubic_arc_segment(60, 90, 0.04)
#arc4 = cubic_arc_segment(90, 120, 0.04)
#arc5 = cubic_arc_segment(120, 150, 0.04)
#arc6 = cubic_arc_segment(150, 180, 0.04)
#arc = join_paths(join_paths(join_paths(arc1, arc2), join_paths(arc3, arc4)), join_paths(arc5, arc6))
#print(arc)


def print_locals(local_vars: dict):
    """
    Pretty-prints the top level of a dict with sorted keys.
    Collapses non-scalar values to a single-line string.
    """
    max_key_len = max(len(k) for k in local_vars) + 1 # +1 for a space
    scalar_types = (int, float, str, bool, type(None))

    for key, value in sorted(local_vars.items()):        
        if key[0] == "_":
            pass
        elif isinstance(value, scalar_types):
            print(f"{key:<{max_key_len}}: {value}")
        else:
            value_str = str(value).replace('\n', ' ')
            print(f"{key:<{max_key_len}}: {value_str}")



