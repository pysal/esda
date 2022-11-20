import unittest

import libpysal
import numpy as np
from libpysal.common import ATOL, RTOL, pandas

from ..moran import Moran
from ..smaup import Smaup

PANDAS_EXTINCT = pandas is None


class Smaup_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])
        self.rho = Moran(self.y, self.w).I
        self.n = len(self.y)
        self.k = int(self.n / 2)

    def test_smaup(self):
        sm = Smaup(self.n, self.k, self.rho)
        self.assertAlmostEqual(sm.n, 78)
        self.assertAlmostEqual(sm.k, 39)
        self.assertAlmostEqual(sm.rho, 0.24365582621771695)
        np.testing.assert_allclose(sm.smaup, 0.15221341690376405, rtol=RTOL, atol=ATOL)
        self.assertAlmostEqual(sm.critical_01, 0.38970613333333337)
        self.assertAlmostEqual(sm.critical_05, 0.3557221333333333)
        self.assertAlmostEqual(sm.critical_1, 0.3157950666666666)
        self.assertEqual(sm.summary, "Pseudo p-value > 0.10 (H0 is not rejected)")

    def test_sids(self):
        w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        SIDR = np.array(f.by_col("SIDR74"))
        rho = Moran(SIDR, w, two_tailed=False).I
        n = len(SIDR)
        k = int(n / 2)
        sm = Smaup(n, k, rho)
        np.testing.assert_allclose(sm.smaup, 0.15176796553181948, rtol=RTOL, atol=ATOL)
        self.assertAlmostEqual(sm.critical_01, 0.23404000000000003)
        self.assertAlmostEqual(sm.critical_05, 0.21088)
        self.assertAlmostEqual(sm.critical_1, 0.18239)
        self.assertEqual(sm.summary, "Pseudo p-value > 0.10 (H0 is not rejected)")

    @unittest.skipIf(PANDAS_EXTINCT, "missing pandas")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        np.random.seed(11213)
        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        k = int(w.n / 2)
        mi = Moran.by_col(df, ["SIDR74"], w=w, two_tailed=False)
        rho = np.unique(mi.SIDR74_moran.values).item()
        sm = Smaup(w.n, k, rho)
        np.testing.assert_allclose(sm.smaup, 0.15176796553181948, atol=ATOL, rtol=RTOL)
        self.assertAlmostEqual(sm.critical_01, 0.23404000000000003)
        self.assertAlmostEqual(sm.critical_05, 0.21088)
        self.assertAlmostEqual(sm.critical_1, 0.18239)
        self.assertEqual(sm.summary, "Pseudo p-value > 0.10 (H0 is not rejected)")
