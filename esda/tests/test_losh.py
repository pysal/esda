# based off: https://github.com/pysal/esda/blob/master/tests/test_moran.py#L96
import libpysal
import numpy as np
import pytest

from esda.losh import LOSH

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


class TestLosh:
    def setup_method(self):
        np.random.seed(10)
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    @parametrize_w
    def test_defaults(self, w):
        ls = LOSH(connectivity=w, inference="chi-square").fit(self.y)
        np.testing.assert_allclose(ls.Hi[0], 0.77613471)
        np.testing.assert_allclose(ls.pval[0], 0.22802201)
