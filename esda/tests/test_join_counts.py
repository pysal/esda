import numpy as np
import pytest
from libpysal import graph
from libpysal.weights import KNN, W
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

# k=3 nearest neighbors on a 4 x 4 lattice of points. Each corner takes the
# diagonal point as its third neighbor, but that point already has four
# neighbors at distance 1 and does not take the corner back (GH #198).
coords = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
parametrize_asymmetric_w = pytest.mark.parametrize(
    "w",
    [
        KNN.from_array(coords, k=3),
        graph.Graph.build_knn(coords, k=3),
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

    @parametrize_asymmetric_w
    def test_asymmetric_weights(self, w):
        with pytest.raises(NotImplementedError, match="symmetric spatial weights"):
            Join_Counts(self.y, w)

    @pytest.mark.parametrize("as_graph", [False, True], ids=["W", "Graph"])
    @pytest.mark.parametrize("transform", ["r", "d", "v"])
    def test_transformed_weights(self, as_graph, transform):
        """Symmetric weights pass the check whatever their transformation"""
        w = lat2W(4, 4)
        w.transform = transform
        if as_graph:
            w = graph.Graph.from_W(w)
        jc = Join_Counts(self.y, w, permutations=0)
        np.testing.assert_allclose(jc.bb, 10.0)
        np.testing.assert_allclose(jc.bw, 4.0)
        np.testing.assert_allclose(jc.ww, 10.0)

    @pytest.mark.parametrize("as_graph", [False, True], ids=["W", "Graph"])
    def test_one_way_zero_weight_link(self, as_graph):
        """A one-way link with zero weight is still counted as a join"""
        w = lat2W(4, 4)
        neighbors, weights = dict(w.neighbors), dict(w.weights)
        neighbors[0] = [*neighbors[0], 5]
        weights[0] = [*weights[0], 0.0]
        w = W(neighbors, weights)
        if as_graph:
            w = graph.Graph.from_W(w)
        with pytest.raises(NotImplementedError, match="symmetric spatial weights"):
            Join_Counts(self.y, w)
