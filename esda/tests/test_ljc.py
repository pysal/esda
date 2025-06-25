# based off: https://github.com/pysal/esda/blob/master/esda/tests/test_join_counts.py
import numpy as np
import pytest
from libpysal import graph
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


class TestJoinCountsLocal:
    """Unit test for Local Join Counts (univariate)"""

    def setup_method(self):
        self.y = np.ones(16)
        self.y[0:8] = 0

    @parametrize_w
    def test_defaults(self, w):
        np.random.seed(12345)
        ljc = Join_Counts_Local(connectivity=w).fit(self.y)
        assert np.array_equal(ljc.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 2, 2, 3, 3, 2])
