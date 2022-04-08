import unittest
import libpysal
from libpysal.common import pandas, RTOL, ATOL
from .. import moran
import numpy as np


PANDAS_EXTINCT = pandas is None
SEED = 12345


class Moran_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    def test_moran(self):
        mi = moran.Moran(self.y, self.w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.24365582621771659, rtol=RTOL, atol=ATOL)
        self.assertAlmostEqual(mi.p_norm, 0.00013573931385468807)

    def test_sids(self):
        w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        SIDR = np.array(f.by_col("SIDR74"))
        mi = moran.Moran(SIDR, w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        self.assertAlmostEqual(mi.p_norm, 5.7916539074498452e-05)

    def test_variance(self):
        y = np.arange(1, 10)
        w = libpysal.weights.util.lat2W(3, 3)
        mi = moran.Moran(y, w, transformation="B")
        np.testing.assert_allclose(
            mi.VI_rand, 0.059687500000000004, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            mi.VI_norm, 0.053125000000000006, atol=ATOL, rtol=RTOL
        )

    def test_z_consistency(self):
        m1 = moran.Moran(self.y, self.w)
        # m2 = moran.Moran_BV(self.x, self.y, self.w) TODO testing for other.z values
        m3 = moran.Moran_Local(self.y, self.w, keep_simulations=True, seed=SEED)
        # m4 = moran.Moran_Local_BV(self.x, self.y, self.w)
        np.testing.assert_allclose(m1.z, m3.z, atol=ATOL, rtol=RTOL)

    @unittest.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        mi = moran.Moran.by_col(df, ["SIDR74"], w=w, two_tailed=False)
        sidr = np.unique(mi.SIDR74_moran.values).item()
        pval = np.unique(mi.SIDR74_p_sim.values).item()
        np.testing.assert_allclose(sidr, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        self.assertAlmostEqual(pval, 0.001)


class Moran_Rate_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col["SID79"])
        self.b = np.array(f.by_col["BIR79"])

    def test_moran_rate(self):
        mi = moran.Moran_Rate(self.e, self.b, self.w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.16622343552567395, rtol=RTOL, atol=ATOL)
        self.assertAlmostEqual(mi.p_norm, 0.004191499504892171)

    @unittest.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        mi = moran.Moran_Rate.by_col(
            df, ["SID79"], ["BIR79"], w=self.w, two_tailed=False
        )
        sidr = np.unique(mi["SID79-BIR79_moran_rate"].values).item()
        pval = np.unique(mi["SID79-BIR79_p_sim"].values).item()
        np.testing.assert_allclose(sidr, 0.16622343552567395, rtol=RTOL, atol=ATOL)
        self.assertAlmostEqual(pval, 0.008)


class Moran_BV_matrix_Tester(unittest.TestCase):
    def setUp(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        varnames = ["SIDR74", "SIDR79", "NWR74", "NWR79"]
        self.names = varnames
        vars = [np.array(f.by_col[var]) for var in varnames]
        self.vars = vars
        self.w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()

    def test_Moran_BV_matrix(self):
        res = moran.Moran_BV_matrix(self.vars, self.w, varnames=self.names)
        self.assertAlmostEqual(res[(0, 1)].I, 0.19362610652874668)
        self.assertAlmostEqual(res[(3, 0)].I, 0.37701382542927858)


class Moran_Local_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("desmith.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("desmith.txt"))
        self.y = np.array(f.by_col["z"])

    def test_Moran_Local(self):
        lm = moran.Moran_Local(
            self.y,
            self.w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        self.assertAlmostEqual(lm.z_sim[0], -0.6990291160835514)
        self.assertAlmostEqual(lm.p_z_sim[0], 0.24226691753791396)

    def test_Moran_Local_parallel(self):
        lm = moran.Moran_Local(
            self.y,
            self.w,
            transformation="r",
            n_jobs=-1,
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        self.assertAlmostEqual(lm.z_sim[0], -0.6990291160835514)
        self.assertAlmostEqual(lm.p_z_sim[0], 0.24226691753791396)


    @unittest.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        import pandas as pd

        df = pd.DataFrame(self.y, columns=["z"])
        lm = moran.Moran_Local.by_col(
            df,
            ["z"],
            w=self.w,
            transformation="r",
            permutations=99,
            outvals=["z_sim", "p_z_sim"],
            keep_simulations=True,
            seed=SEED,
        )
        self.assertAlmostEqual(lm.z_z_sim[0], -0.6990291160835514)
        self.assertAlmostEqual(lm.z_p_z_sim[0], 0.24226691753791396)

    def test_local_moments(self):
        lm = moran.Moran_Local(
            self.y,
            self.w,
            transformation="r",
            permutations=0,
            seed=SEED,
        )
        
        wikh_fast = moran._wikh_fast(lm.w.sparse)
        wikh_slow = moran._wikh_slow(lm.w.sparse)
        wikh_fast_c = moran._wikh_fast(lm.w.sparse, sokal_correction=True)
        wikh_slow_c = moran._wikh_slow(lm.w.sparse, sokal_correction=True)

        np.testing.assert_allclose(wikh_fast, wikh_slow, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(wikh_fast, wikh_slow, rtol=RTOL, atol=ATOL)
        EIc = np.array([-0.00838113, -0.0243949 , -0.07031778, 
                           -0.21520869, -0.16547163, -0.00178435, 
                           -0.11531888, -0.36138555, -0.05471258, -0.09413562])
        VIc = np.array([0.03636013, 0.10412408, 0.28600769, 
                           0.26389674, 0.21576683, 0.00779261, 
                           0.44633942, 0.57696508, 0.12929777, 0.3730742 ])
        
        EI = -np.ones((10,))/9
        VI = np.array([0.47374172, 0.47356458, 0.47209663, 
                       0.15866023, 0.15972526, 0.47376436, 
                       0.46927721, 0.24584217, 0.26498308, 0.47077467])

        


        np.testing.assert_allclose(lm.EIc, EIc, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.VIc, VIc, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.EI, EI, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.VI, VI, rtol=RTOL, atol=ATOL)


class Moran_Local_BV_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.x = np.array(f.by_col["SIDR79"])
        self.y = np.array(f.by_col["SIDR74"])

    def test_Moran_Local_BV(self):
        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            self.w,
            keep_simulations=True,
            transformation="r",
            permutations=99,
            seed=SEED,
        )
        self.assertAlmostEqual(lm.Is[0], 1.4649221250620736)
        self.assertAlmostEqual(lm.z_sim[0], 1.330673752886702)
        self.assertAlmostEqual(lm.p_z_sim[0], 0.09164819151535242)

    @unittest.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        moran.Moran_Local_BV.by_col(
            df,
            ["SIDR74", "SIDR79"],
            w=self.w,
            inplace=True,
            outvals=["z_sim", "p_z_sim"],
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        bvstats = df["SIDR79-SIDR74_moran_local_bv"].values
        bvz = df["SIDR79-SIDR74_z_sim"].values
        bvzp = df["SIDR79-SIDR74_p_z_sim"].values
        self.assertAlmostEqual(bvstats[0], 1.4649221250620736)
        self.assertAlmostEqual(bvz[0], 1.7900932313425777, 5)
        self.assertAlmostEqual(bvzp[0], 0.036719462378528744, 5)


class Moran_Local_Rate_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col["SID79"])
        self.b = np.array(f.by_col["BIR79"])

    def test_moran_rate(self):
        lm = moran.Moran_Local_Rate(
            self.e, self.b, self.w, transformation="r", permutations=99, seed=SEED
        )
        self.assertAlmostEqual(lm.z_sim[0], 0.02702781851384379, 7)
        self.assertAlmostEqual(lm.p_z_sim[0], 0.4892187730835096)

    @unittest.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        lm = moran.Moran_Local_Rate.by_col(
            df,
            ["SID79"],
            ["BIR79"],
            w=self.w,
            outvals=["p_z_sim", "z_sim"],
            transformation="r",
            permutations=99,
            seed=SEED,
        )
        self.assertAlmostEqual(lm["SID79-BIR79_z_sim"][0], 0.02702781851384379, 7)
        self.assertAlmostEqual(lm["SID79-BIR79_p_z_sim"][0], 0.4892187730835096)


suite = unittest.TestSuite()
test_classes = [
    Moran_Tester,
    Moran_Rate_Tester,
    Moran_BV_matrix_Tester,
    Moran_Local_Tester,
    Moran_Local_BV_Tester,
    Moran_Local_Rate_Tester,
]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
