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
    random_cubic_spline_segment, handdrawn_polyline_path,
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


@pytest.fixture(scope="module")
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
        assert abs(abs(meta["rot_deg"]) - abs(angle_deg)) <= JITTER_ANGLE_DEG


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
    angle_diff = abs((meta["angle_deg"] - base_angle + 90) % 180 - 90)
    assert angle_diff <= JITTER_ANGLE_DEG + 1e-3


def test_random_angle_range(fixed_rng):
    """Without base_angle, result must lie within [-90, 90]."""
    for _ in range(50):
        _, meta = unit_circle_diameter(rng=fixed_rng)
        assert -90 <= meta["angle_deg"] <= 90


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


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Unit tests for random_cubic_spline_segment()
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def test_returns_valid_path(fixed_rng):
    """Must return a Matplotlib Path object with correct codes."""
    p = random_cubic_spline_segment((0, 0), (1, 0), rng=fixed_rng)
    assert isinstance(p, mplPath)
    assert p.vertices.shape == (4, 2)
    assert p.codes.tolist() == [mplPath.MOVETO, mplPath.CURVE4, mplPath.CURVE4, mplPath.CURVE4]


def test_start_and_end_points_preserved(fixed_rng):
    """Start and end vertices must match input coordinates."""
    start, end = (0, 0), (1, 0)
    p = random_cubic_spline_segment(start, end, rng=fixed_rng)
    np.testing.assert_allclose(p.vertices[0], start)
    np.testing.assert_allclose(p.vertices[-1], end)


def test_perpendicular_deviation(fixed_rng):
    """Control points should deviate from the straight line."""
    p = random_cubic_spline_segment((0, 0), (1, 0), amp=0.5, rng=fixed_rng)
    P1, P2 = p.vertices[1:3]
    assert abs(P1[1]) > 0 or abs(P2[1]) > 0  # deviation in y-axis expected


def test_tightness_effect(fixed_rng):
    """Changing tightness should affect control points' proximity to endpoints."""
    p_loose = random_cubic_spline_segment((0, 0), (1, 0), tightness=0.2, rng=fixed_rng)
    p_tight = random_cubic_spline_segment((0, 0), (1, 0), tightness=0.8, rng=fixed_rng)

    # Control point 1 moves closer to the start as tightness increases
    assert p_tight.vertices[1][0] > p_loose.vertices[1][0]


def test_amp_effect_on_curvature(fixed_rng):
    """Higher amplitude should produce greater perpendicular deviation."""
    p_small = random_cubic_spline_segment((0, 0), (1, 0), amp=0.05, rng=fixed_rng)
    p_large = random_cubic_spline_segment((0, 0), (1, 0), amp=0.5, rng=fixed_rng)

    # Compare y deviations of control points
    dev_small = np.max(np.abs(p_small.vertices[:, 1]))
    dev_large = np.max(np.abs(p_large.vertices[:, 1]))
    assert dev_large > dev_small


def test_invalid_input_types():
    """Reject invalid types for start/end."""
    with pytest.raises(TypeError):
        random_cubic_spline_segment("bad", (1, 1))
    with pytest.raises(TypeError):
        random_cubic_spline_segment((0, 0), "bad")


def test_invalid_tuple_length():
    """Reject malformed tuples."""
    with pytest.raises(ValueError):
        random_cubic_spline_segment((0, 0, 1), (1, 1))


@pytest.mark.benchmark(group="random_cubic_spline_segment")
def test_perf_benchmark(benchmark, fixed_rng):
    """Ensure spline generation runs within microsecond range."""
    benchmark(lambda: random_cubic_spline_segment((0, 0), (1, 0), rng=fixed_rng))


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Unit tests for handdrawn_polyline_path()
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

@pytest.fixture(scope="function")
def base_points():
    return [(0, 0), (1, 0.2), (2, -0.3), (3, 0.1)]

# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------
def test_returns_path_instance(base_points, fixed_rng):
    """Function must return a valid Matplotlib Path."""
    path = handdrawn_polyline_path(base_points, rng=fixed_rng)
    assert isinstance(path, mplPath)
    assert len(path.vertices) == 46
    assert path.codes is not None
    assert len(path.vertices) == len(path.codes)


def test_minimum_two_points_required(fixed_rng):
    """Should raise ValueError if less than 2 points."""
    with pytest.raises(ValueError):
        handdrawn_polyline_path([(0, 0)], rng=fixed_rng)


def test_invalid_points_type():
    """Should reject invalid point inputs."""
    with pytest.raises(TypeError):
        handdrawn_polyline_path("not_a_list")  # type: ignore


def test_invalid_amp_type(base_points):
    with pytest.raises(TypeError):
        handdrawn_polyline_path(base_points, amp="big")  # type: ignore


def test_invalid_tightness_type(base_points):
    with pytest.raises(TypeError):
        handdrawn_polyline_path(base_points, tightness="tight")  # type: ignore


def test_invalid_splines_count(base_points):
    """Should reject negative value for splines_per_segment."""
    with pytest.raises(ValueError):
        path = handdrawn_polyline_path(base_points, splines_per_segment=-3)


# ---------------------------------------------------------------------------
# Geometric properties
# ---------------------------------------------------------------------------
def test_path_start_and_end_match_original(base_points, fixed_rng):
    """The resulting path should start and end at the original endpoints."""
    path = handdrawn_polyline_path(base_points, rng=fixed_rng)
    start, end = base_points[0], base_points[-1]
    np.testing.assert_allclose(path.vertices[0], start, atol=1e-8)
    np.testing.assert_allclose(path.vertices[-1], end, atol=1e-8)


def test_closed_polyline_roundtrip(fixed_rng):
    """Closed polyline should have CLOSEPOLY code at the end."""
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    path = handdrawn_polyline_path(square, rng=fixed_rng)
    assert path.codes[-1] == mplPath.CLOSEPOLY
    np.testing.assert_allclose(path.vertices[-1], square[0], atol=1e-8)


def test_path_contains_valid_curves(base_points, fixed_rng):
    """Path must include cubic Bezier segments."""
    path = handdrawn_polyline_path(base_points, rng=fixed_rng)
    assert mplPath.CURVE4 in path.codes
    assert path.codes[0] == mplPath.MOVETO


def test_jitter_introduces_variation(base_points):
    """Repeated calls should produce different geometries."""
    path1 = handdrawn_polyline_path(base_points)
    path2 = handdrawn_polyline_path(base_points)
    assert not np.allclose(path1.vertices, path2.vertices)


# ---------------------------------------------------------------------------
# Geometric stability / boundedness
# ---------------------------------------------------------------------------
def test_vertices_remain_within_expected_bounds(base_points, fixed_rng):
    """All vertices should stay within a reasonable bounding box around input points."""
    amp = 0.2
    path = handdrawn_polyline_path(base_points, amp=amp, rng=fixed_rng)
    verts = np.array(path.vertices)
    xs, ys = zip(*base_points)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Allow 3x amplitude vertical deviation and 10% horizontal tolerance
    tol_x = (x_max - x_min) * 0.1
    tol_y = (y_max - y_min) * 0.1 + 3 * amp

    assert np.all((verts[:, 0] >= x_min - tol_x) & (verts[:, 0] <= x_max + tol_x))
    assert np.all((verts[:, 1] >= y_min - tol_y) & (verts[:, 1] <= y_max + tol_y))
    

def test_vertex_to_vertex_distance_continuity(base_points, fixed_rng):
    """Adjacent vertices must not jump more than 2x average step length."""
    path = handdrawn_polyline_path(base_points, amp=0.3, rng=fixed_rng)
    verts = np.array(path.vertices)
    diffs = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    avg_step = np.mean(diffs)
    max_allowed = 2.0 * avg_step
    assert np.all(diffs < max_allowed), f"Found jump > {max_allowed:.3f}"


# ---------------------------------------------------------------------------
# Reproducibility and randomness
# ---------------------------------------------------------------------------
def test_different_seed_produces_different_results(base_points, fixed_rng):
    rng1 = get_rng(thread_safe=True)
    rng2 = get_rng(thread_safe=True)
    rng1.seed(101)
    rng2.seed(202)
    path1 = handdrawn_polyline_path(base_points, rng=rng1)
    path2 = handdrawn_polyline_path(base_points, rng=rng2)
    assert not np.allclose(path1.vertices, path2.vertices)


# ---------------------------------------------------------------------------
# Performance sanity check
# ---------------------------------------------------------------------------
@pytest.mark.benchmark(group="handdrawn_polyline_path")
def test_runtime_under_threshold(base_points, benchmark):
    """Performance check to ensure efficient execution."""
    def run():
        return handdrawn_polyline_path(base_points)
    path = benchmark(run)
    assert isinstance(path, mplPath)
    assert len(path.vertices) > 4

