import numpy as np
import pytest
from libpysal import graph
from libpysal.weights.util import lat2W

from ..join_counts import Join_Counts

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        lat2W(4, 4),
        graph.Graph.from_W(lat2W(4, 4)),
    ],
    ids=["W", "Graph"],
)


class TestJoinCounts:
    """Unit test for Join Counts"""

    def setup_method(self):
        self.y = np.ones(16)
        self.y[0:8] = 0

    @parametrize_w
    def test_default(self, w):
        """Test method"""
        np.random.seed(12345)
        jc = Join_Counts(self.y, w)
        np.testing.assert_allclose(jc.bb, 10.0)
        np.testing.assert_allclose(jc.bw, 4.0)
        np.testing.assert_allclose(jc.ww, 10.0)
        np.testing.assert_allclose(jc.autocorr_neg, 4.0)  # jc.bw
        np.testing.assert_allclose(jc.autocorr_pos, 20.0)
        np.testing.assert_allclose(jc.J, 24.0)
        np.testing.assert_allclose(len(jc.sim_bb), 999)
        np.testing.assert_allclose(jc.p_sim_bb, 0.0030000000000000001)
        np.testing.assert_allclose(np.mean(jc.sim_bb), 5.5465465465465469)
        np.testing.assert_allclose(np.max(jc.sim_bb), 10.0)
        np.testing.assert_allclose(np.min(jc.sim_bb), 0.0)
        np.testing.assert_allclose(len(jc.sim_bw), 999)
        np.testing.assert_allclose(jc.p_sim_bw, 1.0)
        np.testing.assert_allclose(np.mean(jc.sim_bw), 12.811811811811811)
        np.testing.assert_allclose(np.max(jc.sim_bw), 24.0)
        np.testing.assert_allclose(np.min(jc.sim_bw), 7.0)
        np.testing.assert_allclose(8.166666666666666, jc.chi2)
        np.testing.assert_allclose(0.004266724822176128, jc.chi2_p)
        np.testing.assert_allclose(0.008, jc.p_sim_chi2)
        np.testing.assert_allclose(1.0, jc.p_sim_autocorr_neg)
        np.testing.assert_allclose(0.001, jc.p_sim_autocorr_pos)
        np.testing.assert_allclose(0.2653504320039377, jc.sim_autocorr_chi2)

    @parametrize_w
    def test_by_col(self, w):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["y"])
        np.random.seed(12345)
        r1 = Join_Counts.by_col(
            df, ["y"], w=w, permutations=999
        )  # outvals = ['bb', 'bw', 'ww', 'p_sim_bw', 'p_sim_bb']

        bb = np.unique(r1.y_bb.values)
        bw = np.unique(r1.y_bw.values)
        bb_p = np.unique(r1.y_p_sim_bb.values)
        bw_p = np.unique(r1.y_p_sim_bw.values)
        np.random.seed(12345)
        c = Join_Counts(self.y, w, permutations=999)
        np.testing.assert_allclose(bb, c.bb)
        np.testing.assert_allclose(bw, c.bw)
        np.testing.assert_allclose(bb_p, c.p_sim_bb)
        np.testing.assert_allclose(bw_p, c.p_sim_bw)
