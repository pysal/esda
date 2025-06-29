import libpysal
import numpy as np
import pytest

from esda.geary_local import Geary_Local

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


class TestGearyLocal:
    def setup_method(self):
        np.random.seed(10)
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    @parametrize_w
    def test_defaults(self, w):
        lG = Geary_Local(connectivity=w).fit(self.y)
        np.testing.assert_allclose(lG.localG[0], 0.696703432)
        np.testing.assert_allclose(lG.p_sim[0], 0.19)
