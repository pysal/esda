# based off: https://github.com/pysal/esda/blob/master/esda/tests/test_join_counts.py
import unittest

import numpy as np
from libpysal.common import pandas
from libpysal.weights.util import lat2W

from esda.join_counts_local import Join_Counts_Local

PANDAS_EXTINCT = pandas is None


class Join_Counts_Locals_Tester(unittest.TestCase):
    """Unit test for Local Join Counts (univariate)"""

    def setUp(self):
        self.w = lat2W(4, 4)
        self.y = np.ones(16)
        self.y[0:8] = 0

    def test_Join_Counts_Locals(self):
        """Test method"""
        np.random.seed(12345)
        ljc = Join_Counts_Local(connectivity=self.w).fit(self.y)
        assert np.array_equal(ljc.LJC, [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 2, 2, 3, 3, 2])


suite = unittest.TestSuite()
test_classes = [Join_Counts_Locals_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
