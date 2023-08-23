import pytest
from numpy import array, testing
import shapely
import esda
import geopandas, numpy

pytest.importorskip("numba")


shape = array(
    [
        shapely.geometry.Polygon(
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

## for a hole/multipart testbench:

test_geom_translated = shapely.from_wkt(
    [
        "POLYGON ((-3.1823503126754247 0.085191513232644, -3.2545854200972997 0.271135116748269, "
        "-3.2472001661910497 0.296769882373269, -3.2779008497847997 0.3333146821779565, "
        "-3.286461030448862 0.4668824922365502, -3.312919770683237 0.4887788545412377, "
        "-3.308738862480112 0.5528352510256127, -3.271751557792612 0.6005037080568627, "
        "-3.2749711622847997 0.6791780244631127, -3.301475678886362 0.7266938936037377, "
        "-3.3279496779097997 0.7266252290529565, -3.3439103712691747 0.8217256318849877, "
        "-3.385307465995737 0.9057634126467065, -3.385490571464487 1.0868776094240502, "
        "-3.4014665236129247 1.1563432466310815, -3.3219682325972997 1.1584108125490502, "
        "-3.3168107618941747 1.328538681201394, -3.2181779493941747 1.3360688936037377, "
        "-3.2150346388472997 1.4000871431154565, -3.1089555372847997 1.4057023774904565, "
        "-3.105888520683237 1.9232576143068627, -2.702140962089487 1.9063203584474877, "
        "-2.7046891798629247 0.8271501313967065, -2.749656831230112 0.7665956270021752, "
        "-2.671684419120737 0.5808656465334252, -2.8658219923629247 0.3313920747560815, "
        "-2.9108354200972997 0.1093156587404565, -3.1823503126754247 0.085191513232644))"
    ]
)[0]

test_simple = shapely.difference(
    shapely.box(-1, 0, -2, 1), shapely.box(-1, 0, -1.5, 0.5)
)

test_hole = shapely.difference(
    shapely.box(0, 0, 1.81, 1.81), shapely.box(0.8, 0.8, 1.6, 1.6)
)

test_mp = shapely.union(shapely.box(-1, -1, -1.5, -2), shapely.box(0, -1, 1.25, -2))

test_mp_hole = shapely.union(
    shapely.transform(
        test_hole, lambda x: numpy.column_stack((-x[:, 0] + 3, x[:, 1] * 0.5 + 3))
    ),
    shapely.transform(test_hole, lambda x: numpy.column_stack((x[:, 0] + 4, x[:, 1]))),
)

testbench = geopandas.GeoDataFrame(
    geometry=[test_geom_translated, test_simple, test_mp, test_hole, test_mp_hole]
).reset_index()
testbench["name"] = ["Hanock County", "Simple", "Multi", "Single Hole", "Multi Hole"]


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


def test_moment_of_interia():
    observed = esda.shape.moment_of_inertia(shape)
    testing.assert_allclose(observed, 0.315715, atol=ATOL)


def test_second_areal_moment():
    observed = esda.shape.second_areal_moment(testbench.geometry)
    testing.assert_allclose(
        observed,
        [0.23480628, 0.11458333, 1.57459077, 1.58210246, 14.18946959],
        atol=ATOL,
    )


def test_moa():
    observed = esda.shape.moa_ratio(shape)
    testing.assert_allclose(observed, 5.35261, atol=ATOL)


def test_nmi():
    observed = esda.shape.nmi(shape)
    testing.assert_allclose(observed, 0.802796, atol=ATOL)


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
