import unittest
import libpysal
from libpysal.common import pandas, RTOL, ATOL
from esda import local_geary
import numpy as np

PANDAS_EXTINCT = pandas is None

from ..local_geary import Local_Geary

class Local_Geary_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col['HR8893'])

    def test_local_geary(self):
        lG = Local_Geary(connectivity=self.w).fit(self.y)
        self.assertAlmostEqual(lG.localG[0], 0.696703432)
        self.assertAlmostEqual(lG.p_sim[0], 0.19)
        
suite = unittest.TestSuite()
test_classes = [
    Local_Geary_Tester
]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
