# based off: https://github.com/pysal/esda/blob/master/tests/test_join_counts.py
import numpy as np
import pytest
from libpysal import graph
from libpysal.weights.util import lat2W

from esda.join_counts_local_mv import Join_Counts_Local_MV

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        lat2W(4, 4),
        graph.Graph.from_W(lat2W(4, 4)),
    ],
    ids=["W", "Graph"],
)


class TestLocalJoinCountsMV:
    """Unit test for Local Join Counts (multivariate)"""

    def setup_method(self):
        self.x = np.ones(16)
        self.x[0:8] = 0
        self.y = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        self.z = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]

    @parametrize_w
    def test_defaults(self, w):
        np.random.seed(12345)
        ljc_mv = Join_Counts_Local_MV(connectivity=w).fit([self.x, self.y, self.z])
        assert np.array_equal(
            ljc_mv.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2]
        )
