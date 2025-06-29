"""Geary Unittest."""

import numpy as np
import pytest
from libpysal import examples, graph
from libpysal.io import open as popen
from libpysal.weights import Rook

from .. import geary

w = Rook.from_shapefile(examples.get_path("columbus.shp"))
w.transform = "r"

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        w,
        graph.Graph.from_W(w),
    ],
    ids=["W", "Graph"],
)


class TestGeary:
    """Geary class for unit tests."""

    def setup_method(self):
        f = popen(examples.get_path("columbus.dbf"))
        self.y = np.array(f.by_col["CRIME"])

    @parametrize_w
    def test_default(self, w):
        c = geary.Geary(self.y, w, permutations=0)
        np.testing.assert_allclose(c.C, 0.5154408058652411)
        np.testing.assert_allclose(c.EC, 1.0)
        np.testing.assert_allclose(c.VC_norm, 0.011403109626468939)
        np.testing.assert_allclose(c.seC_norm, 0.10678534368755357)
        np.testing.assert_allclose(c.p_norm, 2.8436375936967054e-06)
        np.testing.assert_allclose(c.z_norm, -4.5376938201608)

        np.testing.assert_allclose(c.VC_rand, 0.010848882767131886)
        np.testing.assert_allclose(c.p_rand, 1.6424069196305004e-06)
        np.testing.assert_allclose(c.z_rand, -4.652156651668989)

        np.random.seed(12345)
        c = geary.Geary(self.y, w, permutations=999)
        np.testing.assert_allclose(c.C, 0.5154408058652411)
        np.testing.assert_allclose(c.EC, 1.0)
        np.testing.assert_allclose(c.VC_norm, 0.011403109626468939)
        np.testing.assert_allclose(c.seC_norm, 0.10678534368755357)
        np.testing.assert_allclose(c.p_norm, 2.8436375936967054e-06)
        np.testing.assert_allclose(c.z_norm, -4.5376938201608)

        np.testing.assert_allclose(c.VC_rand, 0.010848882767131886)
        np.testing.assert_allclose(c.p_rand, 1.6424069196305004e-06)
        np.testing.assert_allclose(c.z_rand, -4.652156651668989)

        np.testing.assert_allclose(c.EC_sim, 0.9981883958193233)
        np.testing.assert_allclose(c.VC_sim, 0.010631247074115058)
        np.testing.assert_allclose(c.p_sim, 0.001)
        np.testing.assert_allclose(c.p_z_sim, 1.4207015378575605e-06)

    @parametrize_w
    def test_by_col(self, w):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["y"])
        np.random.seed(12345)
        r1 = geary.Geary.by_col(df, ["y"], w=w, permutations=999)
        this_geary = np.unique(r1.y_geary.values)
        this_pval = np.unique(r1.y_p_sim.values)
        c = geary.Geary(self.y, w, permutations=999)
        np.testing.assert_allclose(this_geary, c.C)
        np.testing.assert_allclose(this_pval, c.p_sim)
