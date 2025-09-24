from libpysal import examples
from esda import correlogram
import numpy as np
from numpy.testing import assert_array_almost_equal

import geopandas as gpd


sac = gpd.read_file(examples.load_example("Sacramento1").get_path("sacramentot2.shp"))
sac = sac.to_crs(sac.estimate_utm_crs())  # now in meters)

distances = [i + 500 for i in range(0, 2000, 500)]
kdists = list(range(1, 6))


def test_distance_correlogram():
    corr = correlogram(sac, "HH_INC", distances)

    test_data = np.array(
        [
            0.05822723177762817,
            0.49206877942505206,
            0.45494217612839183,
            0.5625914469490942,
        ]
    )

    assert_array_almost_equal(corr.I, test_data)


def test_k_distance_correlogram():
    corr = correlogram(sac, "HH_INC", kdists, distance_type="knn")

    test_data = np.array([0.62411734, 0.59734846, 0.56958116, 0.54252517, 0.54093269])
    assert_array_almost_equal(corr.I, test_data)


def test_unspecified_distances():
    corr = correlogram(sac, "HH_INC", distance_type="knn")

    assert len(corr) == 11  # not divisible by n_bins, so near 10
    assert_array_almost_equal(
        corr.I,
        [
            0.62411734,
            0.29078276,
            0.19922394,
            0.15656318,
            0.11640395,
            0.08748932,
            0.05202192,
            0.02162432,
            0.00577212,
            0.0,
            -0.0024899,
        ],
    )


def test_lowess_correlogram():
    corr = correlogram(sac, "HH_INC", distances, statistic="lowess")

    test_data = np.array([0.290893, 0.278535, 0.266325, 0.254256])

    assert_array_almost_equal(corr.lowess, test_data)


def test_lowess_precomputed():
    raise NotImplementedError()
