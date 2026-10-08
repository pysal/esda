import warnings

import libpysal
import numpy as np
import pandas as pd
import pytest

from esda.geary_local_mv import Geary_Local_MV

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        libpysal.io.open(libpysal.examples.get_path("stl.gal")).read(),
        libpysal.graph.Graph.from_W(
            libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        ),
    ],
    ids=["W", "Graph"],
)


class TestGearyLocalMV:
    def setup_method(self):
        np.random.seed(100)
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y1 = np.array(f.by_col["HR8893"])
        self.y2 = np.array(f.by_col["HC8488"])

    @parametrize_w
    def test_defaults(self, w):
        lG_mv = Geary_Local_MV(connectivity=w).fit(np.column_stack([self.y1, self.y2]))
        np.testing.assert_allclose(lG_mv.localG[0], 0.4096931479581422)
        np.testing.assert_allclose(lG_mv.p_sim[0], 0.211)

    @parametrize_w
    def test_dataframe(self, w):
        df = pd.DataFrame({"y1": self.y1, "y2": self.y2})
        lG_mv = Geary_Local_MV(connectivity=w, permutations=0).fit(df)
        np.testing.assert_allclose(lG_mv.localG[0], 0.4096931479581422)
        assert lG_mv.variables.shape == (len(self.y1), 2)

    @parametrize_w
    def test_features_as_rows_deprecated(self, w):
        with pytest.warns(FutureWarning, match="n_features, n_samples"):
            lG_mv = Geary_Local_MV(connectivity=w).fit([self.y1, self.y2])
        np.testing.assert_allclose(lG_mv.localG[0], 0.4096931479581422)
        np.testing.assert_allclose(lG_mv.p_sim[0], 0.211)

    @parametrize_w
    def test_shape_mismatch(self, w):
        with pytest.raises(ValueError, match="does not match"):
            Geary_Local_MV(connectivity=w).fit(np.ones((10, 2)))

    @pytest.mark.parametrize("graph", [False, True], ids=["W", "Graph"])
    def test_square_input_is_row_first(self, graph):
        # With n_features == n_samples the layout cannot be told from the
        # shape, so the array is read as (n_samples, n_features) without
        # a warning.
        w = libpysal.weights.lat2W(1, 3)
        if graph:
            w = libpysal.graph.Graph.from_W(w)
        X = np.array([[0.0, 1.0, 4.0], [2.0, 5.0, 1.0], [7.0, 3.0, 9.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            lG_mv = Geary_Local_MV(connectivity=w, permutations=0).fit(X)
            lG_mv_t = Geary_Local_MV(connectivity=w, permutations=0).fit(X.T)
        np.testing.assert_allclose(lG_mv.localG, [2.42935636, 2.92503925, 3.42072214])
        np.testing.assert_allclose(lG_mv_t.localG, [2.01098901, 3.41208791, 4.81318681])
