"""
-------
test_matplotlib_integration.py
-------
"""

import pytest
from primitives.line import Line


def test_draw_adds_line_to_axes(fig_ax):
  """Ensure that calling draw() actually adds a Line2D object to the Axes."""
  fig, ax = fig_ax
  before = len(ax.lines)
  line = Line(ax)  # auto-generates metadata
  line.draw(ax)
  after = len(ax.lines)
  assert after == before + 1
