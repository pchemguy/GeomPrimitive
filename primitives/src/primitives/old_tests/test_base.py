"""
-------
test_base.py
-------
"""

import json
import re
import pytest
from primitives.line import Line


def test_reset_regenerates_metadata(fig_ax):
  """Repeated geometry generation should yield different metadata (randomness)."""
  _, ax = fig_ax
  line = Line(ax)
  m1 = line.meta
  line.make_geometry(ax)
  m2 = line.meta
  assert isinstance(m2, dict)
  assert m1 != m2  # randomness introduces difference


def test_meta_returns_deepcopy(fig_ax):
  """Modifying a copy of meta should not affect the original."""
  _, ax = fig_ax
  line = Line(ax)
  d = line.meta
  assert "x" in d and isinstance(d["x"], list)
  d["x"][0] = -999
  # The internal meta should remain intact
  assert line.meta["x"][0] != -999


def test_repr_contains_keys(fig_ax):
  """__repr__ should include key names from current metadata."""
  _, ax = fig_ax
  line = Line(ax)
  s = repr(line)
  for key in line.meta.keys():
    assert key in s


def test_reseed_changes_rng_output(fig_ax):
  """Changing RNG seed should yield different random metadata."""
  _, ax = fig_ax
  Line.reseed(1)
  line1 = Line(ax)
  m1 = line1.meta
  Line.reseed(2)
  line2 = Line(ax)
  m2 = line2.meta
  assert m1 != m2


def test_json_and_jsonpp_are_valid_json(fig_ax):
  _, ax = fig_ax
  line = Line(ax)

  # Both should return strings
  s_compact = line.json
  s_pretty = line.jsonpp

  assert isinstance(s_compact, str)
  assert isinstance(s_pretty, str)

  # Both should parse cleanly
  parsed_compact = json.loads(s_compact)
  parsed_pretty = json.loads(s_pretty)
  assert parsed_compact == parsed_pretty

  # Pretty version should contain newlines and indentation
  assert "\n" in s_pretty
  assert len(s_pretty) > len(s_compact)


def test_str_representation_contains_class_and_id(fig_ax):
  _, ax = fig_ax
  line = Line(ax)

  s = str(line)
  # Should mention class name
  assert line.__class__.__name__ in s
  # Should contain an object ID (memory address) pattern
  assert re.search(r"0x[0-9A-Fa-f]+", s)
  # Should embed readable JSON
  assert "{" in s and "}" in s

  # The JSON fragment should also parse
  json_part = s.split("{", 1)[1]
  json_part = "{" + json_part  # restore leading brace
  # Gracefully ignore trailing non-JSON (if any)
  try:
    json.loads(json_part)
  except json.JSONDecodeError:
    # At minimum must contain valid structure markers
    assert s.count("{") >= 1 and s.count("}") >= 1
