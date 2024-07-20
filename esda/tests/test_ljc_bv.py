# based off: https://github.com/pysal/esda/blob/master/tests/test_join_counts.py
import numpy as np
from libpysal import graph
from libpysal.weights.util import lat2W
import pytest

from esda.join_counts_local_bv import Join_Counts_Local_BV

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        lat2W(4, 4),
        graph.Graph.from_W(
             lat2W(4, 4)
        ),
    ],
    ids=["W", "Graph"],
)



class TestLocalJoinCountsBV:
    """Unit test for Local Join Counts (bivariate)"""

    def setup_method(self):
        self.x = np.ones(16)
        self.x[0:8] = 0
        self.z = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    @parametrize_w
    def test_Local_Join_Counts_BV(self, w):
        """Test method"""
        np.random.seed(12345)
        ljc_bv_case1 = Join_Counts_Local_BV(connectivity=w).fit(
            self.x, self.z, case="BJC"
        )
        ljc_bv_case2 = Join_Counts_Local_BV(connectivity=w).fit(
            self.x, self.z, case="CLC"
        )
        assert np.array_equal(
            ljc_bv_case1.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        )
        assert np.array_equal(
            ljc_bv_case2.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2]
        )
