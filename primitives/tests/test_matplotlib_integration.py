import pytest
from primitives.base import Primitive
from primitives.line import Line
from matplotlib.axes import Axes

def test_draw_adds_line_to_axes(fig_ax, line_instance):
  fig, ax = fig_ax
  before = len(ax.lines)
  line_instance.make_geometry(ax).draw(ax)
  after = len(ax.lines)
  assert after == before + 1
