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
