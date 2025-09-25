from libpysal import examples
from libpysal.weights.util import get_points_array
from esda import correlogram
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import spatial
import geopandas as gpd
import pytest


sac = gpd.read_file(examples.load_example("Sacramento1").get_path("sacramentot2.shp"))
sac = sac.to_crs(sac.estimate_utm_crs())  # now in meters)

dsupport = [i + 500 for i in range(0, 2000, 500)]
ksupport = list(range(1, 6))

try:
    import statsmodels  # noqa F401
except ImportError:
    statsmodels = None

def test_distance_correlogram():
    corr = correlogram(sac.geometry.centroid, sac.HH_INC, dsupport)

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
    corr = correlogram(sac.geometry.centroid, sac.HH_INC, ksupport, distance_type="knn")

    test_data = np.array([0.62411734, 0.59734846, 0.56958116, 0.54252517, 0.54093269])
    assert_array_almost_equal(corr.I, test_data)


def test_unspecified_distances():
    corr = correlogram(sac.geometry.centroid, sac.HH_INC, distance_type="knn")

    assert len(corr) == 50
    known = [
        0.62411734,
        0.48356077,
        0.40204787,
        0.36573375,
        0.32139756,
        -0.00165216,
        -0.00217858,
        -0.00237038,
        -0.00260488,
        -0.00248756,
    ]
    assert_array_almost_equal(  # check the first five and last five
        [*corr.I[:5], *corr.I[-5:]],
        known,
    )

@pytest.mark.skipif(statsmodels is None, reason="lowess requires statsmodels")
def test_lowess_correlogram():
    corr = correlogram(
        sac.geometry.centroid, sac.HH_INC, support=dsupport, statistic="lowess"
    )

    test_data = np.array([0.586032, 0.377228, 0.268441, 0.336877])

    assert_array_almost_equal(corr.lowess, test_data)

@pytest.mark.skipif(statsmodels is None, reason="lowess requires statsmodels")
def test_lowess_precomputed():
    coords = get_points_array(sac.geometry.centroid)
    n_samples = len(coords)
    D = np.zeros((n_samples, n_samples))
    upper_ix = np.triu_indices_from(D)
    lower_ix = np.tril_indices_from(D)

    # diff in x on top, diff in y on bottom
    D[upper_ix] = spatial.distance_matrix(coords[:, 0, None], coords[:, 0, None])[
        upper_ix
    ]
    D[lower_ix] = spatial.distance_matrix(coords[:, 1, None], coords[:, 1, None])[
        lower_ix
    ]

    corr = correlogram(
        sac.geometry.centroid,
        sac.HH_INC,
        statistic="lowess",
        stat_kwargs=dict(metric="precomputed", coordinates=D),
    )

    assert_array_almost_equal(
        corr.lowess.iloc[:5],
        [0.07460634, 0.05469855, 0.02606339, 0.02323869, 0.01068146],
    )
