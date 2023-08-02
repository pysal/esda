import pytest
from numpy import array, testing
from shapely import geometry

import esda

pytest.importorskip("numba")


shape = array(
    [
        geometry.Polygon(
            [
                (0, 0),
                (0.25, 0.25),
                (0, 0.5),
                (0.25, 0.75),
                (0, 1),
                (1.25, 1),
                (0.75, 0.5),
                (1.25, 0),
            ]
        )
    ]
)

ATOL = 0.001


def test_boundary_amplitude():
    observed = esda.shape.boundary_amplitude(shape)
    testing.assert_allclose(observed, 0.844527, atol=ATOL)


def test_convex_hull_ratio():
    observed = esda.shape.convex_hull_ratio(shape)
    testing.assert_allclose(observed, 0.7, atol=ATOL)


def test_length_width_diff():
    observed = esda.shape.length_width_diff(shape)
    testing.assert_allclose(observed, 0.25, atol=ATOL)


def test_radii_ratio():
    observed = esda.shape.radii_ratio(shape)
    testing.assert_allclose(observed, 0.659366, atol=ATOL)


def test_diameter_ratio():
    observed = esda.shape.diameter_ratio(shape)
    testing.assert_allclose(observed, 0.8, atol=ATOL)


def test_iaq():
    observed = esda.shape.isoareal_quotient(shape)
    testing.assert_allclose(observed, 0.622314, atol=ATOL)


def test_ipq():
    observed = esda.shape.isoperimetric_quotient(shape)
    testing.assert_allclose(observed, 0.387275, atol=ATOL)


def test_moa():
    observed = esda.shape.moa_ratio(shape)
    testing.assert_allclose(observed, 3.249799, atol=ATOL)


def test_moment_of_interia():
    observed = esda.shape.moment_of_inertia(shape)
    testing.assert_allclose(observed, 0.315715, atol=ATOL)


def test_nmi():
    observed = esda.shape.nmi(shape)
    testing.assert_allclose(observed, 0.487412, atol=ATOL)


def test_mbc():
    observed = esda.shape.minimum_bounding_circle_ratio(shape)
    testing.assert_allclose(observed, 0.434765, atol=ATOL)


def test_reflexive_angle_ratio():
    observed = esda.shape.reflexive_angle_ratio(shape)
    testing.assert_allclose(observed, 3 / 8, atol=ATOL)


def test_fractal_dimension():
    r = [
        esda.shape.fractal_dimension(shape, support=support)[0]
        for support in ("hex", "square", "circle")
    ]
    testing.assert_allclose(r, [0.218144, -4.29504, 0.257882], atol=ATOL)


def test_squareness():
    observed = esda.shape.squareness(shape)
    testing.assert_allclose(observed, 0.493094, atol=ATOL)


def test_rectangularity():
    observed = esda.shape.rectangularity(shape)
    testing.assert_allclose(observed, 0.7, atol=ATOL)


def test_shape_index():
    observed = esda.shape.shape_index(shape)
    testing.assert_allclose(observed, 0.659366, atol=ATOL)


def test_equivalent_rectangular_index():
    observed = esda.shape.equivalent_rectangular_index(shape)
    testing.assert_allclose(observed, 0.706581, atol=ATOL)


def test_form_factor():
    observed = esda.shape.form_factor(shape, array([2]))
    testing.assert_allclose(observed, 0.602535, atol=ATOL)
