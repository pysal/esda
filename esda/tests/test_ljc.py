# based off: https://github.com/pysal/esda/blob/master/esda/tests/test_join_counts.py
import numpy as np
import pytest
from libpysal import graph
from libpysal.weights import KNN
from libpysal.weights.util import lat2W

from esda.join_counts_local import Join_Counts_Local

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        lat2W(4, 4),
        graph.Graph.from_W(lat2W(4, 4)),
    ],
    ids=["W", "Graph"],
)

# k=3 nearest neighbors on a 4 x 4 lattice of points are asymmetric, see
# test_join_counts.py (GH #198).
coords = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
parametrize_asymmetric_w = pytest.mark.parametrize(
    "w",
    [
        KNN.from_array(coords, k=3),
        graph.Graph.build_knn(coords, k=3),
    ],
    ids=["W", "Graph"],
)


class TestJoinCountsLocal:
    """Unit test for Local Join Counts (univariate)"""

    def setup_method(self):
        self.y = np.ones(16)
        self.y[0:8] = 0

    @parametrize_w
    def test_defaults(self, w):
        np.random.seed(12345)
        with pytest.WARN_ALT_HYPOTHESIS_DEPR:
            ljc = Join_Counts_Local(connectivity=w).fit(self.y)
        assert np.array_equal(ljc.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 2, 2, 3, 3, 2])

    @parametrize_asymmetric_w
    def test_asymmetric_weights(self, w):
        with pytest.raises(NotImplementedError, match="symmetric spatial weights"):
            Join_Counts_Local(connectivity=w).fit(self.y)
