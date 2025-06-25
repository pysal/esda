import libpysal
import numpy as np
import pytest

from ..gamma import Gamma

parametrize_lat = pytest.mark.parametrize(
    "w",
    [
        libpysal.weights.util.lat2W(4, 4),
        libpysal.graph.Graph.from_W(libpysal.weights.util.lat2W(4, 4)),
    ],
    ids=["W", "Graph"],
)


class TestGamma:
    """Unit test for Gamma Index"""

    def setup_method(self):
        self.y = np.ones(16)
        self.y[0:8] = 0
        np.random.seed(12345)

    @parametrize_lat
    def test_default(self, w):
        g = Gamma(self.y, w)

        np.testing.assert_allclose(g.g, 20.0)
        np.testing.assert_allclose(g.g_z, 3.1879280354548638)
        np.testing.assert_allclose(g.p_sim_g, 0.0030000000000000001)
        np.testing.assert_allclose(g.min_g, 0.0)
        np.testing.assert_allclose(g.max_g, 20.0)
        np.testing.assert_allclose(g.mean_g, 11.093093093093094)

    @parametrize_lat
    def test_s(self, w):
        np.random.seed(12345)
        g1 = Gamma(self.y, w, operation="s")
        np.testing.assert_allclose(g1.g, 8.0)
        np.testing.assert_allclose(g1.g_z, -3.7057554345954791)
        np.testing.assert_allclose(g1.p_sim_g, 0.001)
        np.testing.assert_allclose(g1.min_g, 14.0)
        np.testing.assert_allclose(g1.max_g, 48.0)
        np.testing.assert_allclose(g1.mean_g, 25.623623623623622)

    @parametrize_lat
    def test_a(self, w):
        np.random.seed(12345)
        g2 = Gamma(self.y, w, operation="a")
        np.testing.assert_allclose(g2.g, 8.0)
        np.testing.assert_allclose(g2.g_z, -3.7057554345954791)
        np.testing.assert_allclose(g2.p_sim_g, 0.001)
        np.testing.assert_allclose(g2.min_g, 14.0)
        np.testing.assert_allclose(g2.max_g, 48.0)
        np.testing.assert_allclose(g2.mean_g, 25.623623623623622)

    @parametrize_lat
    def test_standardize(self, w):
        np.random.seed(12345)
        g3 = Gamma(self.y, w, standardize=True)
        np.testing.assert_allclose(g3.g, 32.0)
        np.testing.assert_allclose(g3.g_z, 3.7057554345954791)
        np.testing.assert_allclose(g3.p_sim_g, 0.001)
        np.testing.assert_allclose(g3.min_g, -48.0)
        np.testing.assert_allclose(g3.max_g, 20.0)
        np.testing.assert_allclose(g3.mean_g, -3.2472472472472473)

    @parametrize_lat
    def test_op(self, w):
        np.random.seed(12345)
        if isinstance(w, libpysal.graph.Graph):
            pytest.skip("Calleble not supported with Graph")

        def func(z, i, j):
            q = z[i] * z[j]
            return q

        g4 = Gamma(self.y, w, operation=func)
        np.testing.assert_allclose(g4.g, 20.0)
        np.testing.assert_allclose(g4.g_z, 3.1879280354548638)
        np.testing.assert_allclose(g4.p_sim_g, 0.0030000000000000001)

    @parametrize_lat
    def test_by_col(self, w):
        import pandas as pd

        g = Gamma(self.y, w)

        df = pd.DataFrame(self.y, columns=["y"])
        r1 = Gamma.by_col(df, ["y"], w=w)
        assert "y_gamma" in r1.columns
        assert "y_p_sim" in r1.columns
        this_gamma = np.unique(r1.y_gamma.values)
        this_pval = np.unique(r1.y_p_sim.values)

        np.testing.assert_allclose(this_gamma, g.g)
        np.testing.assert_allclose(this_pval, g.p_sim)
        Gamma.by_col(df, ["y"], inplace=True, operation="s", w=w)
        this_gamma = np.unique(df.y_gamma.values)
        this_pval = np.unique(df.y_p_sim.values)
        np.testing.assert_allclose(this_gamma, 8.0)
        np.testing.assert_allclose(this_pval, 0.001)
