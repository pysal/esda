import libpysal
import numpy as np
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
        lG_mv = Geary_Local_MV(connectivity=w).fit([self.y1, self.y2])
        print(lG_mv.p_sim[0])
        np.testing.assert_allclose(lG_mv.localG[0], 0.4096931479581422)
        np.testing.assert_allclose(lG_mv.p_sim[0], 0.211)
