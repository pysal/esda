# based off: https://github.com/pysal/esda/blob/master/tests/test_join_counts.py
import unittest

import numpy as np
from libpysal.common import pandas
from libpysal.weights.util import lat2W

from esda.join_counts_local_bv import Join_Counts_Local_BV

PANDAS_EXTINCT = pandas is None


class Local_Join_Counts_BV_Tester(unittest.TestCase):
    """Unit test for Local Join Counts (univariate)"""

    def setUp(self):
        self.w = lat2W(4, 4)
        self.x = np.ones(16)
        self.x[0:8] = 0
        self.z = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    def test_Local_Join_Counts_BV(self):
        """Test method"""
        np.random.seed(12345)
        ljc_bv_case1 = Join_Counts_Local_BV(connectivity=self.w).fit(
            self.x, self.z, case="BJC"
        )
        ljc_bv_case2 = Join_Counts_Local_BV(connectivity=self.w).fit(
            self.x, self.z, case="CLC"
        )
        assert np.array_equal(
            ljc_bv_case1.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        )
        assert np.array_equal(
            ljc_bv_case2.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2]
        )


suite = unittest.TestSuite()
test_classes = [Local_Join_Counts_BV_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
