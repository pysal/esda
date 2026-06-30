"""Tests for the _optimal_compactness module."""

from __future__ import annotations

import pytest
from shapely import Point, Polygon, box

from .. import _optimal_compactness as _oc
from .._optimal_compactness import CIRCLE_QUAD_SEGS, _regular_polygon
from ..shape import optimal_compactness as oc


@pytest.fixture(autouse=True)
def _single_worker(monkeypatch):
    monkeypatch.setattr(_oc, "cpus", 1)


def test_optimal_compactness():
    polys = []

    # circle
    poly = Point(0, 0).buffer(100, quad_segs=CIRCLE_QUAD_SEGS)
    polys.append(poly)

    # triangle
    poly = _regular_polygon(3, 0, 0, 3, 15)
    polys.append(poly)

    # hexagon
    poly = _regular_polygon(6, 0, 0, 2, 90)
    polys.append(poly)

    # star
    poly = _regular_polygon(3, 0, 0, 3, 20).union(_regular_polygon(3, 0, 0, 3, 200))
    polys.append(poly)

    # complex shape
    poly1 = Point(0, 0).buffer(1, quad_segs=1)
    poly2 = Point(-0.5, 1).buffer(1, quad_segs=1)
    poly = poly1.union(poly2)
    polys.append(poly)

    # rectangle with a hole
    outer = box(0, 0, 8, 4)
    hole = box(1, 1, 7, 3)
    poly = Polygon(outer.exterior.coords, holes=[hole.exterior.coords])
    polys.append(poly)

    # complex shape
    s1 = box(0.5, 0, 1.5, 7)
    s2 = box(2.5, 0, 3.5, 3)
    s3 = box(4.5, 0, 5.5, 7)
    s4 = box(0.5, 0, 5.5, 1)
    poly = s1.union(s2).union(s3).union(s4)
    polys.append(poly)

    for poly in polys:
        _, _ = oc(poly, circle=True, return_result=True)
        _ = oc(poly, circle=False, n=None)
        _ = oc(poly, circle=False, n=6)
