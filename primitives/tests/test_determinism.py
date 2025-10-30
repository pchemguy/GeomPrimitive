import pytest
from primitives.base import Primitive
from primitives.line import Line
from matplotlib.axes import Axes


def test_determinism_with_reseed(fig_ax, line_instance):
  _, ax = fig_ax
  Line.reseed(123)
  meta1 = line_instance.make_geometry(ax)
  Line.reseed(123)
  meta2 = line_instance.make_geometry(ax)
  assert meta1 == meta2
