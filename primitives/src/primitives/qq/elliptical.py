import numpy as np
import random
import math
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath
from matplotlib.transforms import Affine2D


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



def elliptical_arc(hrange: tuple[float, float] = (0, 1023),
                   vrange: tuple[float, float] = (0, 1023),
                   start_deg: Optional[float] = None,
                   end_deg: Optional[float] = None,
                   aspect_ratio: Optional[float] = None,
                   angle_deg: Optional[int] = None,
                   jitter_amp: Optional[float] = 0.02,
                   jitter_aspect: float = 0.1,
                   max_angle_delta_deg: Optional[int] = 20,
                   min_angle_steps: Optional[int] = 3,
                  ) -> mplPath:
    xmin, xmax = hrange
    ymin, ymax = vrange
    if aspect_ratio is None or not isinstance(aspect_ratio, (float, int)):
        aspect_ratio = random.uniform(0, 1)
    aspect_ratio = min(0.1, max(aspect_ratio, 1))
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

    jitter_y = 1 - (jitter_aspect * random.uniform(0, 1) if jitter_aspect else 0)

    P0 = [float(np.cos(start)), float(np.sin(start))]
    verts = [P0]
    start_section = start
    end_section = start_section + delta_step
    for i in range(step_count):
        P1 = [float(np.cos(start_section) - t * np.sin(start_section)),
              float((np.sin(start_section) + t * np.cos(start_section)) * jitter_y)]
        P2 = [float(np.cos(end_section)   + t * np.sin(end_section)),
              float((np.sin(end_section)   - t * np.cos(end_section)) * jitter_y)]
        P3 = [float(np.cos(end_section)),
              float((np.sin(end_section)) * jitter_y)]
        verts.extend([P1, P2, P3])
        start_section += delta_step
        end_section += delta_step
    codes = [mplPath.MOVETO] + [mplPath.CURVE4] * 3 * step_count

    if jitter_amp:
        p0 = verts[0]
        jittered_verts = [p0]
        for vert in verts[1:]:
            jittered_verts.append([
                vert[0] + jitter_amp * random.uniform(-1, 1),
                vert[1] + jitter_amp * random.uniform(-1, 1)
            ])
        verts = jittered_verts

    if closed:
        codes.append(mplPath.CLOSEPOLY)
        verts.append(P0)
    
    arc_path: mplPath = mplPath(verts, codes)

    return arc_path

arcarc = elliptical_arc(hrange=(-1, 1), vrange=(-1, 1), start_deg=0, end_deg=360)
# print(arcarc)


hrange=(-10, 20)
vrange=(-10, 30)

xmin, xmax = hrange
ymin, ymax = vrange
dx, dy = ymax - ymin, xmax - xmin
x0, y0 = (xmin + xmax) / 2, (ymin + ymax) / 2

scale_factor = min(dx, dy) * (1 - random.uniform(0.2, 0.9)) * 0.5
aspect_ratio = None
if aspect_ratio is None or not isinstance(aspect_ratio, (float, int)):
    aspect_ratio = max(0.25, 1 - abs(random.normalvariate(0, 0.25)))
x_scale = scale_factor
y_scale = scale_factor * aspect_ratio

shift_amp_x = max(0, (dx - scale_factor * math.sqrt(2)) / 2)
shift_amp_y = max(0, (dy - scale_factor * math.sqrt(2)) / 2)
x_translate = x0 + shift_amp_x * random.uniform(-1, 1) / 2
y_translate = y0 + shift_amp_y * random.uniform(-1, 1) / 2

angle_deg = None
if not isinstance(angle_deg, (int, float)):
    angle_deg = random.uniform(-90, 90)
angle = np.radians(angle_deg)
rotation_matrix = np.array([
    [+np.cos(angle), np.sin(angle)],
    [-np.sin(angle), np.cos(angle)]
])

verts = arcarc.vertices
# Apply SRT (Scale, Rotate, Translate) in one line
# (verts * scale) @ rotate + translate
verts = (verts * [x_scale, y_scale]) @ rotation_matrix + [x_translate, y_translate]    
arcarc = mplPath(verts, arcarc.codes)


print(
    f"x0         : {x0}         \n"
    f"y0         : {y0}         \n"
    f"dx         : {dx}         \n"
    f"dy         : {dy}         \n"
    f"x_scale    : {x_scale}    \n"
    f"y_scale    : {y_scale}    \n"
    f"x_translate: {x_translate}\n"
    f"y_translate: {y_translate}\n"
)


#_context = locals()
#_context.pop("arcarc")
#print_locals(_context)

#exit()


fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(PathPatch(arcarc, edgecolor="blue", lw=2, facecolor="none", linestyle="--"))

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



arc1 = cubic_arc_segment(0, 30, 0.04)  # any angle span <=90deg
arc2 = cubic_arc_segment(30, 60, 0.04)
arc3 = cubic_arc_segment(60, 90, 0.04)
arc4 = cubic_arc_segment(90, 120, 0.04)
arc5 = cubic_arc_segment(120, 150, 0.04)
arc6 = cubic_arc_segment(150, 180, 0.04)
arc = join_paths(join_paths(join_paths(arc1, arc2), join_paths(arc3, arc4)), join_paths(arc5, arc6))
print(arc)

