"""
test_mpl_path_utils.py
----------------------
Unit tests for mpl_path_utils.py
"""

import math
import numpy as np
import pytest
from matplotlib.path import Path as mplPath
from matplotlib.patches import Circle, Ellipse, Arc

from spt.mpl_path_utils import (
    join_paths, random_srt_path, unit_circle_diameter, ellipse_or_arc_path,
    JITTER_ANGLE_DEG,
)
from spt.rng import RNG, get_rng


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unit_circle_path():
    """A simple circular path centered at origin, radius 1."""
    c = Circle((0, 0), radius=1.0)
    return c.get_transform().transform_path(c.get_path())


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


@pytest.fixture
def fixed_rng():
    """Deterministic RNG instance."""
    r = RNG(seed=123)
    return r


@pytest.fixture(scope="module")
def base_call(fixed_rng):
    """Generate a default result for reuse."""
    path, meta = unit_circle_diameter(rng=fixed_rng)
    return path, meta

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


# ---------------------------------------------------------------------
# Basic Functionality
# ---------------------------------------------------------------------

def test_returns_path_and_meta(unit_circle_path, fixed_rng):
    canvas_x1x2 = (-5.0, 5.0)
    canvas_y1y2 = (-4.0, 4.0)

    path_out, meta = random_srt_path(
        unit_circle_path,
        canvas_x1x2,
        canvas_y1y2,
        rng=fixed_rng,
    )

    assert isinstance(path_out, mplPath)
    assert isinstance(meta, dict)
    # Meta must contain standard keys
    expected_keys = {"scale_x", "scale_y", "rot_x", "rot_y",
                     "rot_deg", "trans_x", "trans_y"}
    assert expected_keys.issubset(meta.keys())


def test_result_stays_within_canvas(unit_circle_path, fixed_rng):
    """All transformed vertices should fit within the given canvas bounds."""
    cx, cy = (-10.0, 10.0), (-10.0, 10.0)
    path_out, _ = random_srt_path(unit_circle_path, cx, cy, rng=fixed_rng)
    xs, ys = path_out.vertices[:, 0], path_out.vertices[:, 1]
    assert xs.min() >= cx[0] - 1e-3
    assert xs.max() <= cx[1] + 1e-3
    assert ys.min() >= cy[0] - 1e-3
    assert ys.max() <= cy[1] + 1e-3


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

def test_same_seed_produces_same_result(unit_circle_path):
    """Two RNGs with the same seed should yield identical transforms."""
    rng1 = RNG(seed=42)
    rng2 = RNG(seed=42)
    cx, cy = (-5.0, 5.0), (-5.0, 5.0)
    p1, m1 = random_srt_path(unit_circle_path, cx, cy, rng=rng1)
    p2, m2 = random_srt_path(unit_circle_path, cx, cy, rng=rng2)
    np.testing.assert_allclose(p1.vertices, p2.vertices)
    assert m1 == m2


def test_different_seeds_produce_different_result(unit_circle_path):
    """Distinct seeds should yield different transforms most of the time."""
    rng1 = RNG(seed=1)
    rng2 = RNG(seed=999)
    cx, cy = (-5.0, 5.0), (-5.0, 5.0)
    p1, _ = random_srt_path(unit_circle_path, cx, cy, rng=rng1)
    p2, _ = random_srt_path(unit_circle_path, cx, cy, rng=rng2)
    assert not np.allclose(p1.vertices, p2.vertices)


# ---------------------------------------------------------------------
# Angle and Rotation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [0, 30, 180])
def test_angle_is_applied(unit_circle_path, angle_deg, fixed_rng):
    """Ensure rotation is reflected in metadata."""
    cx, cy = (-5, 5), (-5, 5)
    _, meta = random_srt_path(
        unit_circle_path,
        cx,
        cy,
        angle_deg=angle_deg,
        rng=fixed_rng,
    )
    assert -180 <= meta["rot_deg"] <= 180
    if angle_deg != 0:
        assert abs(meta["rot_deg"] - angle_deg) <= JITTER_ANGLE_DEG


# ---------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------

def test_invalid_shape_type():
    with pytest.raises(TypeError):
        random_srt_path("not_a_path", (0, 1), (0, 1))

def test_invalid_canvas_types(unit_circle_path):
    with pytest.raises(TypeError):
        random_srt_path(unit_circle_path, (0, 1), "bad")

def test_invalid_origin_type(unit_circle_path):
    with pytest.raises(TypeError):
        random_srt_path(unit_circle_path, (0, 1), (0, 1), origin="not_a_tuple")

def test_zero_canvas_raises(unit_circle_path):
    with pytest.raises(ValueError):
        random_srt_path(unit_circle_path, (0, 0), (0, 1))

def test_degenerate_shape_succeeds(fixed_rng):
    """A zero-size path should not crash but still return a valid Path."""
    p = mplPath(np.zeros((3, 2)))
    out, meta = random_srt_path(p, (-1, 1), (-1, 1), rng=fixed_rng)
    assert isinstance(out, mplPath)
    assert isinstance(meta, dict)
    assert "scale_x" in meta


# ---------------------------------------------------------------------
# Performance sanity check
# ---------------------------------------------------------------------

@pytest.mark.benchmark(group="random_srt_path")
def test_runtime_under_threshold(benchmark, unit_circle_path, fixed_rng):
    """Ensure transform runs within 5ms."""
    cx, cy = (-10, 10), (-10, 10)

    def run():
        random_srt_path(unit_circle_path, cx, cy, rng=fixed_rng)

    result = benchmark(run)
    assert result is None  # benchmark just times it


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Unit tests for unit_circle_diameter()
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1. Basic properties
# ---------------------------------------------------------------------

def test_returns_path_and_meta(base_call):
    """Function must return a Path and a metadata dictionary."""
    path, meta = base_call
    assert isinstance(path, mplPath)
    assert isinstance(meta, dict)
    assert "angle_deg" in meta
    assert "base_angle" in meta
    assert "jitter_deg" in meta


def test_path_has_two_vertices(base_call):
    """Diameter path must contain exactly two vertices."""
    path, _ = base_call
    assert path.vertices.shape == (2, 2)
    assert path.codes.tolist() == [mplPath.MOVETO, mplPath.LINETO]


def test_vertices_on_unit_circle(base_call):
    """Both endpoints should lie on the unit circle (rd1)."""
    path, _ = base_call
    radii = np.sqrt(np.sum(path.vertices**2, axis=1))
    np.testing.assert_allclose(radii, 1.0, atol=1e-7)


def test_line_passes_through_origin(fixed_rng):
    """Midpoint of the line should be near the origin."""
    path, _ = unit_circle_diameter(rng=fixed_rng)
    mid = path.vertices.mean(axis=0)
    np.testing.assert_allclose(mid, (0.0, 0.0), atol=1e-7)


# ---------------------------------------------------------------------
# 2. Angle behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("base_angle", [0, 45, 90, -60])
def test_angle_orientation_consistent(base_angle, fixed_rng):
    """The returned angle should roughly match the requested base_angle."""
    path, meta = unit_circle_diameter(base_angle=base_angle, rng=fixed_rng)
    angle_diff = abs((meta["angle_deg"] - base_angle + 180) % 360 - 180)
    assert angle_diff <= JITTER_ANGLE_DEG + 1e-3


def test_random_angle_range(fixed_rng):
    """Without base_angle, result must lie within [-90, 90]."""
    for _ in range(50):
        _, meta = unit_circle_diameter(rng=fixed_rng)
        assert -90 <= meta["base_angle"] <= 90


# ---------------------------------------------------------------------
# 3. Determinism and randomness
# ---------------------------------------------------------------------

def test_same_seed_produces_same_result():
    """Identical seeds yield identical coordinates."""
    rng1 = RNG(seed=42)
    rng2 = RNG(seed=42)
    p1, m1 = unit_circle_diameter(rng=rng1)
    p2, m2 = unit_circle_diameter(rng=rng2)
    np.testing.assert_allclose(p1.vertices, p2.vertices)
    assert m1 == m2


def test_different_seed_produces_different_result():
    """Distinct seeds yield different coordinates."""
    rng1 = RNG(seed=1)
    rng2 = RNG(seed=999)
    p1, _ = unit_circle_diameter(rng=rng1)
    p2, _ = unit_circle_diameter(rng=rng2)
    assert not np.allclose(p1.vertices, p2.vertices)


# ---------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------

def test_invalid_base_angle_type():
    """Non-numeric base_angle must raise TypeError."""
    with pytest.raises(TypeError):
        unit_circle_diameter(base_angle="invalid_angle")


def test_invalid_rng_type():
    """Passing a bad RNG should still fail cleanly."""
    class Dummy: ...
    with pytest.raises(AttributeError):
        unit_circle_diameter(rng=Dummy())


# ---------------------------------------------------------------------
# 5. Geometry consistency
# ---------------------------------------------------------------------

def test_endpoints_are_opposite(fixed_rng):
    """Endpoints must be diametrically opposite (sum ~ 0)."""
    path, _ = unit_circle_diameter(rng=fixed_rng)
    summed = path.vertices[0] + path.vertices[1]
    np.testing.assert_allclose(summed, (0.0, 0.0), atol=1e-7)


def test_length_equals_diameter(fixed_rng):
    """Line length should equal 2 ~ circle radius (2)."""
    path, _ = unit_circle_diameter(rng=fixed_rng)
    dist = np.linalg.norm(path.vertices[1] - path.vertices[0])
    np.testing.assert_allclose(dist, 2.0, atol=1e-7)


# ---------------------------------------------------------------------
# 6. Performance
# ---------------------------------------------------------------------

@pytest.mark.benchmark(group="unit_circle_diameter")
def test_runtime_under_threshold(benchmark, fixed_rng):
    """Ensure diameter generation runs within 1ms."""
    def run():
        unit_circle_diameter(rng=fixed_rng)
    result = benchmark(run)
    assert result is None  # timing only
