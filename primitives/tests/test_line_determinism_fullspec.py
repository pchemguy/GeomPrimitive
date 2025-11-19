"""
-------
test_line_determinism_fullspec.py
-------

Ensures that when all stylistic parameters are explicitly specified
and hand_drawn=False, the resulting Line metadata is deterministic
except for coordinates and style enums.
"""

import json
import re
from primitives.line import Line


def test_line_deterministic_with_fixed_params(fig_ax):
    _, ax = fig_ax

    params = dict(
        linewidth=2.5,
        pattern="--",
        color="navy",
        alpha=0.8,
        orientation="horizontal",
        hand_drawn=False,
    )

    # --- Case 1: Same seed — everything identical (full determinism) ---
    Line.reseed(999)
    line1 = Line(ax, **params)
    Line.reseed(999)
    line2 = Line(ax, **params)

    meta1, meta2 = line1.meta, line2.meta
    assert meta1 == meta2, "All fields (including coords) must match when reseeded"

    # --- Case 2: Continuous RNG (no reseed) ---
    # Only explicitly provided fields must remain identical.
    # Cap/join styles and coordinates are free to vary.
    line3 = Line(ax, **params)
    line4 = Line(ax, **params)

    meta3, meta4 = line3.meta, line4.meta

    deterministic_fields = {"linewidth", "linestyle", "color", "alpha", "orientation", "hand_drawn"}
    for k in deterministic_fields:
        assert meta3[k] == meta4[k], f"Unexpected variation in deterministic field '{k}'"

    # The following may vary
    variable_fields = {
        "x",
        "y",
        "solid_capstyle",
        "solid_joinstyle",
        "dash_capstyle",
        "dash_joinstyle",
    }
    varying = [k for k in variable_fields if meta3[k] != meta4[k]]
    assert varying, "Expected at least one variable field (e.g., coords or capstyle) to differ"

    # --- JSON / JSONPP / __str__ checks ---
    j = line1.json
    jp = line1.jsonpp
    d = json.loads(j)
    assert d["color"] == "navy"
    assert isinstance(d["linewidth"], (float, int))
    assert "\n" in jp and len(jp) > len(j)

    s = str(line1)
    assert "Line" in s
    assert re.search(r"0x[0-9A-Fa-f]+", s)
    assert "navy" in s
    assert "--" in s
    assert "horizontal" in s or "orientation" in s
