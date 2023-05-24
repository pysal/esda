import os

import geopandas
import numpy
import pytest

from .. import map_comparison as mc

shapely = pytest.importorskip("shapely")

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.zip")

r1 = geopandas.read_file(filepath, layer="regions1")
r1a = mc._cast(r1.geometry)
r2 = geopandas.read_file(filepath, layer="regions2")
r2a = mc._cast(r2.geometry)


def test_areal_entropy():
    ae1 = mc.areal_entropy(r1, base=2)
    ae1e = mc.areal_entropy(r1)
    numpy.testing.assert_equal(2, ae1)
    numpy.testing.assert_allclose(ae1e, 1.38629436111)
    ae1l = mc.areal_entropy(r1, base=2, local=True)
    numpy.testing.assert_allclose(ae1l, numpy.ones((4,)) * 0.5)
    ae1_from_areas = mc.areal_entropy(areas=shapely.area(r1a), base=2)
    numpy.testing.assert_equal(ae1, ae1_from_areas)


def test_overlay_entropy():
    oe = mc.overlay_entropy(r1, r2)
    oe2 = mc.overlay_entropy(r1, r2, base=2)
    # standardization removes base sensitivity
    numpy.testing.assert_equal(oe, oe2)
    oe2_raw = mc.overlay_entropy(r1, r2, base=2, standardize=False)
    oe_raw = mc.overlay_entropy(r1, r2, standardize=False)
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(oe2_raw, oe_raw)
        numpy.testing.assert_equal(oe2_raw, oe2)
    oe2_local = mc.overlay_entropy(r1, r2, base=2, local=True)
    oe2_local_raw = mc.overlay_entropy(r1, r2, base=2, local=True, standardize=False)
    numpy.testing.assert_equal(oe2_local.shape, (4,))
    numpy.testing.assert_equal(oe2_local.sum(), oe2)
    numpy.testing.assert_equal(oe2_local_raw.sum(), oe2_raw)


def test_completeness_and_homogeneity():
    for base in (numpy.e, 2):
        c1 = mc.completeness(r1, r2, base=base, local=True)
        c2 = mc.completeness(r2, r1, base=base, local=True)
        h1 = mc.homogeneity(r1, r2, base=base, local=True)
        h2 = mc.homogeneity(r2, r1, base=base, local=True)

        numpy.testing.assert_array_equal(c1, h2)
        numpy.testing.assert_array_equal(c2, h1)
        numpy.testing.assert_array_equal(c1.sum(), mc.completeness(r1, r2, base=base))

        numpy.testing.assert_equal(c1.shape, r1.geometry.shape)
        numpy.testing.assert_equal(c2.shape, r2.geometry.shape)
        numpy.testing.assert_allclose(c1.sum(), 0.42275150844)
        numpy.testing.assert_allclose(h1.sum(), 0.31534179219)


def test_external_entropy():
    for base in (2, numpy.e):
        v1 = mc.external_entropy(r1, r2, base=base)
        v2 = mc.external_entropy(r2, r1, base=base)
        numpy.testing.assert_allclose(v1, v2)
        numpy.testing.assert_allclose(v1, 0.3612313462)
        v1_ctipped = mc.external_entropy(r1, r2, base=base, balance=100)
        numpy.testing.assert_allclose(
            v1_ctipped, 0.42275150844
        )  # value from completeness
        v1_htipped = mc.external_entropy(r1, r2, base=base, balance=-100)
        numpy.testing.assert_allclose(
            v1_htipped, 0.31534179219
        )  # value from homogeneity
