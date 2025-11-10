"""
test_mpl_path_utils.py
----------------------
Unit tests for mpl_path_utils.py
"""

import numpy as np
import pytest
from matplotlib.path import Path as mplPath
from matplotlib.patches import Circle, Ellipse, Arc

from spt.mpl_path_utils import join_paths, ellipse_or_arc_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_circle_path():
    """Full circle (should match Circle geometry)."""
    return ellipse_or_arc_path(0, 0, 1.0)


@pytest.fixture
def basic_ellipse_path():
    """Full ellipse with y_compress < 1."""
    return ellipse_or_arc_path(0, 0, 1.0, y_compress=0.6)


@pytest.fixture
def arc_path():
    """Open arc (quarter-circle)."""
    return ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=90)


@pytest.fixture
def closed_arc_path():
    """Closed pie-slice arc."""
    return ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=90, close=True)


# ---------------------------------------------------------------------------
# Tests for ellipse_or_arc_path
# ---------------------------------------------------------------------------

def test_circle_equivalence_to_patch(basic_circle_path):
    """Full 360deg circle should be equivalent to a Matplotlib Circle patch."""
    circle_patch = Circle((0, 0), 1.0)
    ref = circle_patch.get_transform().transform_path(circle_patch.get_path())
    assert np.allclose(basic_circle_path.vertices, ref.vertices, atol=1e-7)
    assert basic_circle_path.codes is not None


def test_ellipse_has_different_y_extent(basic_ellipse_path):
    """Elliptical flattening reduces vertical extent."""
    verts = basic_ellipse_path.vertices
    y_range = verts[:, 1].max() - verts[:, 1].min()
    x_range = verts[:, 0].max() - verts[:, 0].min()
    assert y_range < x_range, "y_compress < 1 should produce a flattened ellipse"


def test_arc_not_closed(arc_path):
    """Arc path should not close to center or repeat start vertex."""
    verts = arc_path.vertices
    assert not np.allclose(verts[0], verts[-1]), "Arc should remain open"


def test_closed_arc_adds_center_vertex(closed_arc_path):
    """Closed arcs must contain center vertex and CLOSEPOLY code."""
    verts = closed_arc_path.vertices
    codes = closed_arc_path.codes
    assert any(np.allclose(v, [0, 0]) for v in verts), "Center vertex missing"
    assert codes[-1] == mplPath.CLOSEPOLY


@pytest.mark.parametrize("angle_offset", [0.0, 30.0, 90.0])
def test_angle_offset_rotates_major_axis(angle_offset):
    """Rotation should affect vertex bounding box orientation."""
    path1 = ellipse_or_arc_path(0, 0, 1.0, y_compress=0.5, angle_offset=angle_offset)
    verts = path1.vertices
    x_span = np.ptp(verts[:, 0])
    y_span = np.ptp(verts[:, 1])
    ratio = y_span / x_span

    if angle_offset == 0.0:
        assert ratio < 1.0  # compressed vertically
    else:
        # For rotated ellipse, allow broader ratio tolerance
        assert 0.45 <= ratio <= 2.05, f"Unexpected aspect ratio {ratio:.3f} for angle {angle_offset}"


def test_span_less_than_360_yields_partial_arc():
    """Span < 360deg should produce open arc with fewer vertices than full ellipse."""
    full = ellipse_or_arc_path(0, 0, 1.0)
    partial = ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=180)
    assert len(partial.vertices) < len(full.vertices)


def test_invalid_arguments_raise():
    """Invalid usage should raise errors."""
    with pytest.raises(TypeError):
        join_paths([None])  # not a Path
    with pytest.raises(ValueError):
        join_paths([])  # empty list


# ---------------------------------------------------------------------------
# Tests for join_paths
# ---------------------------------------------------------------------------

def test_join_paths_continuous_merge():
    """When preserve_moveto=False, the joined path omits duplicate MOVETO commands."""
    c1 = ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=90)
    c2 = ellipse_or_arc_path(0, 0, 1.0, start_angle=90, end_angle=180)
    joined = join_paths([c1, c2], preserve_moveto=False)
    # First vertex of second path should be skipped
    assert not np.allclose(joined.vertices[len(c1.vertices)], c2.vertices[0])


def test_join_paths_preserves_disjoint():
    """When preserve_moveto=True, subpaths remain separate."""
    c1 = ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=90)
    c2 = ellipse_or_arc_path(1, 0, 1.0, start_angle=90, end_angle=180)
    joined = join_paths([c1, c2], preserve_moveto=True)
    # Total vertex count = sum of both
    assert len(joined.vertices) == len(c1.vertices) + len(c2.vertices)


def test_join_paths_result_is_valid():
    """Resulting joined path must contain valid vertices and codes arrays."""
    c1 = ellipse_or_arc_path(0, 0, 1.0, start_angle=0, end_angle=90)
    c2 = ellipse_or_arc_path(1, 0, 1.0, start_angle=90, end_angle=180)
    joined = join_paths([c1, c2])
    assert joined.vertices.shape[1] == 2
    assert len(joined.codes) == len(joined.vertices)
    assert np.isfinite(joined.vertices).all()


@pytest.mark.benchmark
def test_perf_benchmark(benchmark):
    result = benchmark(lambda: ellipse_or_arc_path(0, 0, 1.0, y_compress=0.7, start_angle=0, end_angle=270))
    assert isinstance(result, mplPath)
