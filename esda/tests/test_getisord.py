import libpysal
import numpy as np
import pytest
from libpysal.common import ATOL, RTOL
from libpysal.weights.distance import DistanceBand

from .. import getisord

POINTS = np.array([(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)])

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        DistanceBand(POINTS, threshold=15),
        libpysal.graph.Graph.from_W(DistanceBand(POINTS, threshold=15)),
    ],
    ids=["W", "Graph"],
)


class TestGetisG:
    def setup_method(self):
        self.y = np.array([2, 3, 3.2, 5, 8, 7])
        np.random.seed(10)

    @parametrize_w
    def test_default(self, w):
        g = getisord.G(self.y, w)
        np.testing.assert_allclose(g.G, 0.55709779, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g.p_norm, 0.172936, rtol=RTOL, atol=ATOL)

    @parametrize_w
    def test_by_col(self, w):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["y"])
        np.random.seed(12345)
        r1 = getisord.G.by_col(df, ["y"], w=w)
        this_getisord = np.unique(r1.y_g.values)
        this_pval = np.unique(r1.y_p_sim.values)
        np.random.seed(12345)
        stat = getisord.G(self.y, w)
        np.testing.assert_allclose(this_getisord, stat._statistic)
        np.testing.assert_allclose(this_pval, stat.p_sim)


class TestGLocal:
    def setup_method(self):
        self.y = np.array([2, 3, 3.2, 5, 8, 7])
        np.random.seed(10)

    @parametrize_w
    def test_binary(self, w):
        lg = getisord.G_Local(self.y, w, transform="B", seed=10)
        np.testing.assert_allclose(lg.Zs[0], -1.0136729, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lg.p_sim[0], 0.102, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lg.p_z_sim[0], 0.153923, rtol=RTOL, atol=ATOL)

    @parametrize_w
    def test_row_standardized(self, w):
        lg = getisord.G_Local(self.y, w, transform="R", seed=10)
        np.testing.assert_allclose(lg.Zs[0], -0.62074534, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lg.p_sim[0], 0.102, rtol=RTOL, atol=ATOL)

    @parametrize_w
    def test_star_binary(self, w):
        lg = getisord.G_Local(self.y, w, transform="B", star=True, seed=10)
        np.testing.assert_allclose(lg.Zs[0], -1.39727626, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lg.p_sim[0], 0.102, rtol=RTOL, atol=ATOL)

    @parametrize_w
    @pytest.mark.xfail(
        reason="Intermittently does not warn for W param; reason unknown; see gh#331"
    )
    def test_star_row_standardized_warning(self, w):
        with pytest.warns(UserWarning, match="Gi\\* requested, but"):
            getisord.G_Local(self.y, w, transform="R", star=True, seed=10)

    @parametrize_w
    def test_star_row_standardized_values(self, w):
        lg = getisord.G_Local(self.y, w, transform="R", star=True, seed=10)
        np.testing.assert_allclose(lg.Zs[0], -0.62488094, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lg.p_sim[0], 0.102, rtol=RTOL, atol=ATOL)

    @parametrize_w
    def test_by_col(self, w):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["y"])
        r1 = getisord.G_Local.by_col(df, ["y"], w=w, seed=12345)
        stat = getisord.G_Local(self.y, w, seed=12345)
        np.testing.assert_allclose(r1.y_g_local.values, stat.Gs)
        np.testing.assert_allclose(r1.y_p_sim, stat.p_sim)
