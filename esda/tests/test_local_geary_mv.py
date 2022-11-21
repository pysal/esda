import unittest

import libpysal
import numpy as np

from esda.geary_local_mv import Geary_Local_MV


class Geary_Local_MV_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y1 = np.array(f.by_col["HR8893"])
        self.y2 = np.array(f.by_col["HC8488"])

    def test_local_geary_mv(self):
        lG_mv = Geary_Local_MV(connectivity=self.w).fit([self.y1, self.y2])
        print(lG_mv.p_sim[0])
        self.assertAlmostEqual(lG_mv.localG[0], 0.4096931479581422)
        self.assertAlmostEqual(lG_mv.p_sim[0], 0.211)


suite = unittest.TestSuite()
test_classes = [Geary_Local_MV_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
