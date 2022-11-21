# based off: https://github.com/pysal/esda/blob/master/tests/test_moran.py#L96
import unittest

import libpysal
import numpy as np

from esda.losh import LOSH


class Losh_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    def test_losh(self):
        ls = LOSH(connectivity=self.w, inference="chi-square").fit(self.y)
        self.assertAlmostEqual(ls.Hi[0], 0.77613471)
        self.assertAlmostEqual(ls.pval[0], 0.22802201)


suite = unittest.TestSuite()
test_classes = [Losh_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
