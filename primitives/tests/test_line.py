"""
-------
test_line.py
-------
"""

import pytest
from primitives.line import Line


# A. Metadata structure
# -----------------------------------------------------------------------------

from matplotlib._enums import CapStyle, JoinStyle

def test_make_geometry_structure(fig_ax):
  _, ax = fig_ax
  line = Line(ax)
  meta = line.meta
  required = {
      "x", "y", "linewidth", "linestyle", "color", "alpha",
      "orientation", "hand_drawn", "solid_capstyle",
      "solid_joinstyle", "dash_capstyle", "dash_joinstyle"
  }
  assert set(meta.keys()) == required
  assert all(
      isinstance(
          meta[k],
          (float, int, str, list, tuple, bool, type(None), CapStyle, JoinStyle)
      )
      or hasattr(meta[k], "__iter__")
      for k in meta
  )
  
def test_make_geometry_populates_meta(fig_ax):
  _, ax = fig_ax
  line = Line(ax)
  assert isinstance(line.meta, dict)
  assert "x" in line.meta
  assert isinstance(line.meta["x"], list)

# B. Orientation handling
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", [
    "horizontal", "vertical", "diagonal_primary", "diagonal_auxiliary",
    45, -30, None,
])
def test_make_geometry_orientation_variants(fig_ax, orientation):
  _, ax = fig_ax
  line = Line(ax, orientation=orientation)
  meta = line.meta
  assert "x" in meta and "y" in meta
  assert len(meta["x"]) == 2
  assert len(meta["y"]) == 2

# C. Hand-drawn effect
# -----------------------------------------------------------------------------

def test_make_geometry_hand_drawn_effect(fig_ax):
  _, ax = fig_ax
  line1 = Line(ax, hand_drawn=False)
  meta1 = line1.meta
  line2 = Line(ax, hand_drawn=True)
  meta2 = line2.meta
  assert meta1["hand_drawn"] is False
  assert meta2["hand_drawn"] is True
  assert meta1["linestyle"] != meta2["linestyle"]

# D. Color and alpha handling
# -----------------------------------------------------------------------------

def test_make_geometry_color_variants(fig_ax):
  _, ax = fig_ax
  line1 = Line(ax, color="skyblue")
  line2 = Line(ax, color=(0.1, 0.2, 0.3))
  meta1, meta2 = line1.meta, line2.meta
  assert isinstance(meta1["color"], str)
  assert isinstance(meta2["color"], tuple)
  assert all(0 <= c <= 1 for c in meta2["color"])

def test_make_geometry_alpha_bounds(fig_ax):
  _, ax = fig_ax
  line = Line(ax, alpha=1.5)
  meta = line.meta
  assert 0.0 <= meta["alpha"] <= 1.0

# E. Line style and dash patterns
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["--", "-.", " ", "_--_.", None])
def test_make_geometry_dash_pattern_validity(fig_ax, pattern):
  _, ax = fig_ax
  line = Line(ax, pattern=pattern)
  meta = line.meta
  ls = meta["linestyle"]
  assert isinstance(ls, (str, tuple))
  if isinstance(ls, tuple):
    offset, seq = ls
    assert isinstance(offset, int)
    assert isinstance(seq, tuple)
    assert all(isinstance(x, (float, int)) and x > 0 for x in seq)

# F. Coordinate validation
# -----------------------------------------------------------------------------

def test_make_geometry_respects_limits(fig_ax):
  _, ax = fig_ax
  ax.set_xlim(0, 10)
  ax.set_ylim(0, 10)
  line = Line(ax)
  meta = line.meta
  assert all(0 <= x <= 10 for x in meta["x"])
  assert all(0 <= y <= 10 for y in meta["y"])
