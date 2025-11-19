"""
-------
test_determinism.py
-------
"""

import pytest
from primitives.line import Line


def test_determinism_with_reseed(fig_ax):
  """
  Re-seeding the RNG with the same seed must produce
  identical metadata for identical inputs.
  """
  _, ax = fig_ax

  Line.reseed(123)
  line1 = Line(ax)
  m1 = line1.meta

  Line.reseed(123)
  line2 = Line(ax)
  m2 = line2.meta

  assert m1 == m2
