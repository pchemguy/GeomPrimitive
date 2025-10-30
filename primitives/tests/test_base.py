import pytest
from primitives.base import Primitive
from primitives.line import Line
from matplotlib.axes import Axes


def test_reset_regenerates_metadata(fig_ax, line_instance):
  _, ax = fig_ax
  m1 = line_instance.make_geometry(ax)
  line_instance.meta = m1.copy()
  m2 = line_instance.reset(ax).meta
  assert isinstance(m2, dict)
  assert m2 != m1  # randomness introduces difference


def test_to_dict_returns_deepcopy(fig_ax, line_instance):
  _, ax = fig_ax
  meta = line_instance.make_geometry(ax)
  d = line_instance.to_dict()
  d["x"][0] = -999
  assert line_instance.meta["x"][0] != -999


def test_repr_contains_keys(fig_ax, line_instance):
  _, ax = fig_ax
  meta = line_instance.make_geometry(ax)
  s = repr(line_instance)
  for key in meta.keys():
    assert key in s


def test_reseed_changes_rng_output(fig_ax, line_instance):
  _, ax = fig_ax
  Line.reseed(1)
  meta1 = line_instance.make_geometry(ax)
  Line.reseed(2)
  meta2 = line_instance.make_geometry(ax)
  assert meta1 != meta2
