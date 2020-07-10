import unittest
import libpysal
from .. import util
from .. import moran
import numpy as np


class Fdr_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    def test_fdr(self):
        lm = moran.Moran_Local(
            self.y, self.w, transformation="r", permutations=999, seed=10
        )
        self.assertAlmostEqual(util.fdr(lm.p_sim, 0.1), 0.002564102564102564)
        self.assertAlmostEqual(util.fdr(lm.p_sim, 0.05), 0.001282051282051282)


suite = unittest.TestSuite()
test_classes = [Fdr_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
