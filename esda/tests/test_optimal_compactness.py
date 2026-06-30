"""Tests for the _optimal_compactness module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from shapely import Point, Polygon, box

from .. import _optimal_compactness as _oc
from .._optimal_compactness import CIRCLE_QUAD_SEGS, MIN_N_POLYGON, _regular_polygon
from ..shape import optimal_compactness as oc


@pytest.fixture(autouse=True)
def _single_worker(monkeypatch):
    monkeypatch.setattr(_oc, "cpus", 1)


# ---------------------------------------------------------------------------
# _regular_polygon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
def test_regular_polygon_vertex_count(n):
    poly = _regular_polygon(n, 0, 0, 1, 0)
    assert len(poly.exterior.coords) == n + 1  # Shapely closes the ring


def test_regular_polygon_circumradius():
    r = 5.0
    poly = _regular_polygon(6, 0, 0, r, 0)
    coords = np.array(poly.exterior.coords[:-1])
    dists = np.hypot(coords[:, 0], coords[:, 1])
    np.testing.assert_allclose(dists, r, atol=1e-12)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_regular_polygon_area(n):
    r = 2.0
    poly = _regular_polygon(n, 0, 0, r, 0)
    expected = n * r**2 * math.sin(2 * math.pi / n) / 2
    assert poly.area == pytest.approx(expected, rel=1e-10)


def test_regular_polygon_center():
    cx, cy = 3.0, -5.0
    poly = _regular_polygon(4, cx, cy, 2.0, 0)
    assert poly.centroid.x == pytest.approx(cx, abs=1e-10)
    assert poly.centroid.y == pytest.approx(cy, abs=1e-10)


def test_regular_polygon_rotation():
    r = 3.0
    # rotation 0: first vertex at (r, 0)
    x0, y0 = _regular_polygon(4, 0, 0, r, 0).exterior.coords[0]
    assert x0 == pytest.approx(r, abs=1e-10)
    assert y0 == pytest.approx(0.0, abs=1e-10)

    # rotation 90: first vertex at (0, r)
    x1, y1 = _regular_polygon(4, 0, 0, r, 90).exterior.coords[0]
    assert x1 == pytest.approx(0.0, abs=1e-10)
    assert y1 == pytest.approx(r, abs=1e-10)


def test_regular_polygon_invalid_n():
    with pytest.raises(ValueError, match="at least 3"):
        _regular_polygon(MIN_N_POLYGON - 1, 0, 0, 1, 0)


# ---------------------------------------------------------------------------
# _vertex_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 6])
def test_vertex_count_simple(n):
    poly = _regular_polygon(n, 0, 0, 1, 0)
    assert _oc._vertex_count(poly) == n


def test_vertex_count_with_holes():
    outer = box(0, 0, 4, 4)
    hole = box(1, 1, 3, 3)
    poly = Polygon(outer.exterior.coords, holes=[hole.exterior.coords])
    assert _oc._vertex_count(poly) == 8  # 4 exterior + 4 interior


# ---------------------------------------------------------------------------
# _xyr_bounds
# ---------------------------------------------------------------------------


def test_xyr_bounds_unit_square():
    poly = box(0, 0, 1, 1)
    (x_lo, x_hi), (y_lo, y_hi), (r_lo, r_hi) = _oc._xyr_bounds(poly)
    assert (x_lo, x_hi) == (0.0, 1.0)
    assert (y_lo, y_hi) == (0.0, 1.0)
    assert r_lo == pytest.approx(1e-6)
    assert r_hi == pytest.approx(math.hypot(1, 1) / 2)


def test_xyr_bounds_shifted_rectangle():
    poly = box(2, 3, 5, 7)
    (x_lo, x_hi), (y_lo, y_hi), (_, r_hi) = _oc._xyr_bounds(poly)
    assert (x_lo, x_hi) == (2.0, 5.0)
    assert (y_lo, y_hi) == (3.0, 7.0)
    assert r_hi == pytest.approx(math.hypot(3, 4) / 2)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def test_objective_f_circle_exact_match():
    """Non-fit is 0 when the candidate circle exactly matches the polygon."""
    poly = Point(0, 0).buffer(100, quad_segs=CIRCLE_QUAD_SEGS)
    score = _oc._objective_f_circle(np.array([0.0, 0.0, 100.0]), poly)
    assert score == pytest.approx(0.0, abs=1e-12)


def test_objective_f_circle_disjoint():
    """Non-fit is 1 when the candidate circle and polygon are entirely disjoint."""
    poly = box(0, 0, 1, 1)
    score = _oc._objective_f_circle(np.array([100.0, 100.0, 0.5]), poly)
    assert score == pytest.approx(1.0, abs=1e-10)


def test_objective_f_circle_in_range():
    """Non-fit is always in [0, 1]."""
    poly = box(0, 0, 2, 1)
    score = _oc._objective_f_circle(np.array([1.0, 0.5, 0.5]), poly)
    assert 0.0 <= score <= 1.0


def test_objective_f_ngon_exact_match():
    """Non-fit is 0 when the candidate n-gon exactly matches the polygon."""
    n, x, y, r, a = 3, 0.0, 0.0, 2.0, 15.0
    poly = _regular_polygon(n, x, y, r, a)
    score = _oc._objective_f_ngon(np.array([x, y, r, a]), poly, n)
    assert score == pytest.approx(0.0, abs=1e-12)


def test_objective_f_polygon_exact_match():
    """Non-fit is 0 when the variable-n candidate exactly matches the polygon."""
    n, x, y, r, a = 6, 0.0, 0.0, 2.0, 15.0
    # _objective_f_polygon uses rotation_frac where a = rotation_frac * 360 / n
    rotation_frac = a * n / 360  # = 15 * 6 / 360 = 0.25
    poly = _regular_polygon(n, x, y, r, a)
    score = _oc._objective_f_polygon(
        np.array([x, y, r, rotation_frac, float(n)]), poly
    )
    assert score == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# oc() — return type and value contracts
# ---------------------------------------------------------------------------


def test_return_result_false_gives_float():
    result = oc(box(0, 0, 1, 1), circle=True)
    assert isinstance(result, float)


def test_return_result_true_gives_tuple():
    result = oc(box(0, 0, 1, 1), circle=True, return_result=True)
    assert isinstance(result, tuple) and len(result) == 2
    c, opt = result
    assert isinstance(c, float)
    assert isinstance(opt, OptimizeResult)
    assert c == pytest.approx(1 - opt.fun)


# ---------------------------------------------------------------------------
# oc() — semantic / geometric correctness
# ---------------------------------------------------------------------------


def test_circle_compactness_near_one():
    """A circle has compactness ≈ 1 when compared with an optimal circle."""
    circle = Point(0, 0).buffer(100, quad_segs=CIRCLE_QUAD_SEGS)
    assert oc(circle, circle=True) == pytest.approx(1.0, abs=0.01)


def test_regular_polygon_self_compactness():
    """A regular n-gon has polygon compactness ≈ 1 when the comparison n is fixed to n."""
    square = _regular_polygon(4, 0, 0, 2, 15)
    assert oc(square, circle=False, n=4) == pytest.approx(1.0, abs=0.01)


def test_compactness_ordering_vs_circle():
    """More sides → closer to a circle → higher circle compactness."""
    c_tri = oc(_regular_polygon(3, 0, 0, 2, 0), circle=True)
    c_hex = oc(_regular_polygon(6, 0, 0, 2, 0), circle=True)
    c_cir = oc(Point(0, 0).buffer(100, quad_segs=CIRCLE_QUAD_SEGS), circle=True)
    assert c_tri < c_hex < c_cir


def test_elongated_rectangle_less_compact_than_square():
    """A very elongated rectangle is less compact vs a circle than a square of equal area."""
    square = box(0, 0, 4, 4)      # area 16, aspect ratio 1:1
    slab = box(0, 0, 32, 0.5)     # area 16, aspect ratio 64:1
    assert oc(slab, circle=True) < oc(square, circle=True)


def test_polygon_with_hole_lower_than_filled():
    """A rectangle with a hole has lower circle compactness than the filled rectangle."""
    filled = box(0, 0, 8, 4)
    holed = Polygon(filled.exterior.coords, holes=[box(1, 1, 7, 3).exterior.coords])
    assert oc(holed, circle=True) < oc(filled, circle=True)


def test_complex_shapes_in_range():
    """Compactness stays in [0, 1] for complex, non-convex, and multi-ring polygons."""
    shapes = [
        # diamond-approximated blobs joined at an edge
        Point(0, 0).buffer(1, quad_segs=1).union(Point(-0.5, 1).buffer(1, quad_segs=1)),
        # U-shape: three vertical bars with a connecting base
        box(0.5, 0, 1.5, 7)
        .union(box(2.5, 0, 3.5, 3))
        .union(box(4.5, 0, 5.5, 7))
        .union(box(0.5, 0, 5.5, 1)),
    ]
    for poly in shapes:
        c = oc(poly, circle=True)
        assert 0.0 <= c <= 1.0, f"compactness {c} out of range for {poly.geom_type}"
