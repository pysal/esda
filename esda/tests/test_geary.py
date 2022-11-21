"""Geary Unittest."""
import unittest

import numpy as np
from libpysal import examples
from libpysal.common import pandas
from libpysal.io import open as popen
from libpysal.weights import Rook

from .. import geary

PANDAS_EXTINCT = pandas is None


class Geary_Tester(unittest.TestCase):
    """Geary class for unit tests."""

    def setUp(self):
        w = Rook.from_shapefile(examples.get_path("columbus.shp"))
        f = popen(examples.get_path("columbus.dbf"))
        w.transform = "r"
        self.w = w
        self.y = np.array(f.by_col["CRIME"])

    def test_Geary(self):
        c = geary.Geary(self.y, self.w, permutations=0)
        self.assertAlmostEqual(c.C, 0.5154408058652411)
        self.assertAlmostEqual(c.EC, 1.0)
        self.assertAlmostEqual(c.VC_norm, 0.011403109626468939)
        self.assertAlmostEqual(c.seC_norm, 0.10678534368755357)
        self.assertAlmostEqual(c.p_norm, 2.8436375936967054e-06)
        self.assertAlmostEqual(c.z_norm, -4.5376938201608)

        self.assertAlmostEqual(c.VC_rand, 0.010848882767131886)
        self.assertAlmostEqual(c.p_rand, 1.6424069196305004e-06)
        self.assertAlmostEqual(c.z_rand, -4.652156651668989)

        np.random.seed(12345)
        c = geary.Geary(self.y, self.w, permutations=999)
        self.assertAlmostEqual(c.C, 0.5154408058652411)
        self.assertAlmostEqual(c.EC, 1.0)
        self.assertAlmostEqual(c.VC_norm, 0.011403109626468939)
        self.assertAlmostEqual(c.seC_norm, 0.10678534368755357)
        self.assertAlmostEqual(c.p_norm, 2.8436375936967054e-06)
        self.assertAlmostEqual(c.z_norm, -4.5376938201608)

        self.assertAlmostEqual(c.VC_rand, 0.010848882767131886)
        self.assertAlmostEqual(c.p_rand, 1.6424069196305004e-06)
        self.assertAlmostEqual(c.z_rand, -4.652156651668989)

        self.assertAlmostEqual(c.EC_sim, 0.9981883958193233)
        self.assertAlmostEqual(c.VC_sim, 0.010631247074115058)
        self.assertAlmostEqual(c.p_sim, 0.001)
        self.assertAlmostEqual(c.p_z_sim, 1.4207015378575605e-06)

    @unittest.skipIf(PANDAS_EXTINCT, "missing pandas")
    def test_by_col(self):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["y"])
        r1 = geary.Geary.by_col(df, ["y"], w=self.w, permutations=999)
        this_geary = np.unique(r1.y_geary.values)
        this_pval = np.unique(r1.y_p_sim.values)
        np.random.seed(12345)
        c = geary.Geary(self.y, self.w, permutations=999)
        self.assertAlmostEqual(this_geary, c.C)
        self.assertAlmostEqual(this_pval, c.p_sim)


suite = unittest.TestSuite()
test_classes = [Geary_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
