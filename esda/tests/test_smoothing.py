import libpysal
import numpy as np
import pytest
from libpysal.common import ATOL, RTOL, pandas
from libpysal.weights.distance import Kernel

from .. import smoothing as sm

PANDAS_EXTINCT = pandas is None


class TestFlatten:
    def setup_method(self):
        self.input = [[1, 2], [3, 3, 4], [5, 6]]

    def test_flatten(self):
        out1 = sm.flatten(self.input)
        out2 = sm.flatten(self.input, unique=False)
        assert out1 == [1, 2, 3, 4, 5, 6]
        assert out2 == [1, 2, 3, 3, 4, 5, 6]


class TestWMean:
    def setup_method(self):
        self.d = np.array([5, 4, 3, 1, 2])
        self.w1 = np.array([10, 22, 9, 2, 5])
        self.w2 = np.array([10, 14, 17, 2, 5])

    def test_weighted_median(self):
        out1 = sm.weighted_median(self.d, self.w1)
        out2 = sm.weighted_median(self.d, self.w2)
        assert out1 == 4
        assert out2 == 3.5


class TestAgeStd:
    def setup_method(self):
        self.e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        self.b = np.array([1000, 1000, 1100, 900, 1000, 900, 1100, 900])
        self.s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
        self.s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])
        self.n = 2

    def test_crude_age_standardization(self):
        crude = sm.crude_age_standardization(self.e, self.b, self.n).round(8)
        crude_exp = np.array([0.02375000, 0.02666667])
        assert list(crude) == list(crude_exp)

    def test_direct_age_standardization(self):
        direct = np.array(
            sm.direct_age_standardization(self.e, self.b, self.s_b, self.n)
        ).round(8)
        direct_exp = np.array(
            [[0.02374402, 0.01920491, 0.02904848], [0.02665072, 0.02177143, 0.03230508]]
        )
        assert list(direct.flatten()) == list(direct_exp.flatten())

    def test_indirect_age_standardization(self):
        indirect = np.array(
            sm.indirect_age_standardization(self.e, self.b, self.s_e, self.s_b, self.n)
        ).round(8)
        indirect_exp = np.array(
            [[0.02372382, 0.01940230, 0.02900789], [0.02610803, 0.02154304, 0.03164035]]
        )
        assert list(indirect.flatten()) == list(indirect_exp.flatten())


class TestSRate:
    def setup_method(self):
        sids = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"), "r")
        self.w = libpysal.io.open(libpysal.examples.get_path("sids2.gal"), "r").read()
        self.b, self.e = np.array(sids[:, 8]), np.array(sids[:, 9])
        self.er = [0.453433, 0.000000, 0.775871, 0.973810, 3.133190]
        self.eb = [0.0016973, 0.0017054, 0.0017731, 0.0020129, 0.0035349]
        self.sr = [0.0009922, 0.0012639, 0.0009740, 0.0007605, 0.0050154]
        self.smr = [0.00083622, 0.00109402, 0.00081567, 0.0, 0.0048209]
        self.smr_w = [0.00127146, 0.00127146, 0.0008433, 0.0, 0.0049889]
        self.smr2 = [0.00091659, 0.00087641, 0.00091073, 0.0, 0.00467633]
        self.s_ebr10 = np.array(
            [
                4.01485749e-05,
                3.62437513e-05,
                4.93034844e-05,
                5.09387329e-05,
                3.72735210e-05,
                3.69333797e-05,
                5.40245456e-05,
                2.99806055e-05,
                3.73034109e-05,
                3.47270722e-05,
            ]
        ).reshape(-1, 1)

        stl_ex = libpysal.examples.load_example("stl")
        self.stl = libpysal.io.open(stl_ex.get_path("stl_hom.csv"), "r")
        self.stl_e, self.stl_b = np.array(self.stl[:, 10]), np.array(self.stl[:, 13])
        self.stl_w = libpysal.io.open(stl_ex.get_path("stl.gal"), "r").read()
        if not self.stl_w.id_order_set:
            self.stl_w.id_order = list(range(1, len(self.stl) + 1))

        if not PANDAS_EXTINCT:
            self.df = libpysal.io.open(libpysal.examples.get_path("sids2.dbf")).to_df()
            self.ename = "SID74"
            self.bname = "BIR74"
            self.stl_df = libpysal.io.open(
                libpysal.examples.get_path("stl_hom.csv")
            ).to_df()
            self.stl_ename = "HC7984"
            self.stl_bname = "PO7984"

    def test_excess_risk(self):
        out_er = sm.Excess_Risk(self.e, self.b).r
        np.testing.assert_allclose(out_er[:5].flatten(), self.er, rtol=RTOL, atol=ATOL)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_excess_risk_tabular(self):
        out_er = sm.Excess_Risk(self.df[self.ename], self.df[self.bname]).r
        np.testing.assert_allclose(out_er[:5].flatten(), self.er, rtol=RTOL, atol=ATOL)
        assert isinstance(out_er, np.ndarray)

    def test_empirical_bayes(self):
        out_eb = sm.Empirical_Bayes(self.e, self.b).r
        np.testing.assert_allclose(out_eb[:5].flatten(), self.eb, rtol=RTOL, atol=ATOL)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_empirical_bayes_tabular(self):
        out_eb = sm.Empirical_Bayes(self.df[self.ename], self.df[self.bname]).r
        np.testing.assert_allclose(out_eb[:5].flatten(), self.eb, rtol=RTOL, atol=ATOL)
        assert isinstance(out_eb, np.ndarray)

    def test_spatial_empirical_bayes(self):
        s_eb = sm.Spatial_Empirical_Bayes(self.stl_e, self.stl_b, self.stl_w)
        np.testing.assert_allclose(self.s_ebr10, s_eb.r[:10], rtol=RTOL, atol=ATOL)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_spatial_empirical_bayes_tabular(self):
        s_eb = sm.Spatial_Empirical_Bayes(
            self.stl_df[self.stl_ename], self.stl_df[self.stl_bname], self.stl_w
        ).r
        assert isinstance(s_eb, np.ndarray)
        np.testing.assert_allclose(self.s_ebr10, s_eb[:10])

    def test_spatial_rate(self):
        out_sr = sm.Spatial_Rate(self.e, self.b, self.w).r
        np.testing.assert_allclose(out_sr[:5].flatten(), self.sr, rtol=RTOL, atol=ATOL)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_spatial_rate_tabular(self):
        out_sr = sm.Spatial_Rate(self.df[self.ename], self.df[self.bname], self.w).r
        np.testing.assert_allclose(out_sr[:5].flatten(), self.sr, rtol=RTOL, atol=ATOL)
        assert isinstance(out_sr, np.ndarray)

    def test_spatial_median_rate(self):
        out_smr = sm.Spatial_Median_Rate(self.e, self.b, self.w).r
        out_smr_w = sm.Spatial_Median_Rate(self.e, self.b, self.w, aw=self.b).r
        out_smr2 = sm.Spatial_Median_Rate(self.e, self.b, self.w, iteration=2).r
        np.testing.assert_allclose(
            out_smr[:5].flatten(), self.smr, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            out_smr_w[:5].flatten(), self.smr_w, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            out_smr2[:5].flatten(), self.smr2, atol=ATOL, rtol=RTOL
        )

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_spatial_median_rate_tabular(self):
        out_smr = sm.Spatial_Median_Rate(
            self.df[self.ename], self.df[self.bname], self.w
        ).r
        out_smr_w = sm.Spatial_Median_Rate(
            self.df[self.ename], self.df[self.bname], self.w, aw=self.df[self.bname]
        ).r
        out_smr2 = sm.Spatial_Median_Rate(
            self.df[self.ename], self.df[self.bname], self.w, iteration=2
        ).r

        assert isinstance(out_smr, np.ndarray)
        assert isinstance(out_smr_w, np.ndarray)
        assert isinstance(out_smr2, np.ndarray)

        np.testing.assert_allclose(
            out_smr[:5].flatten(), self.smr, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            out_smr_w[:5].flatten(), self.smr_w, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            out_smr2[:5].flatten(), self.smr2, atol=ATOL, rtol=RTOL
        )


class TestKernelAgeAdjSM:
    def setup_method(self):
        self.e = np.array([10, 1, 3, 4, 2, 5])
        self.b = np.array([100, 15, 20, 20, 80, 90])
        self.e1 = np.array([10, 8, 1, 4, 3, 5, 4, 3, 2, 1, 5, 3])
        self.b1 = np.array([100, 90, 15, 30, 25, 20, 30, 20, 80, 80, 90, 60])
        self.s = np.array([98, 88, 15, 29, 20, 23, 33, 25, 76, 80, 89, 66])
        self.points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        self.kw = Kernel(self.points)
        self.points1 = np.array(self.points) + 5
        self.points1 = np.vstack((self.points1, self.points))
        self.kw1 = Kernel(self.points1)
        if not self.kw.id_order_set:
            self.kw.id_order = list(range(0, len(self.points)))
        if not PANDAS_EXTINCT:
            import pandas as pd

            dfa = np.array([self.e, self.b]).T
            dfb = np.array([self.e1, self.b1, self.s]).T
            self.dfa = pd.DataFrame(dfa, columns=["e", "b"])
            self.dfb = pd.DataFrame(dfb, columns=["e", "b", "s"])

        # answers
        self.kernel_exp = [
            0.10543301,
            0.0858573,
            0.08256196,
            0.09884584,
            0.04756872,
            0.04845298,
        ]
        self.ageadj_exp = [
            0.10519625,
            0.08494318,
            0.06440072,
            0.06898604,
            0.06952076,
            0.05020968,
        ]
        self.disk_exp = [
            0.12222222000000001,
            0.10833333,
            0.08055556,
            0.08944444,
            0.09944444,
            0.09351852,
        ]
        self.sf_exp = np.array([0.111111, 0.111111, 0.085106, 0.076923])

    def test_smoother(self):
        kr = sm.Kernel_Smoother(self.e, self.b, self.kw)
        np.testing.assert_allclose(kr.r.flatten(), self.kernel_exp)

    def test_age_adjusted_smoother(self):
        ar = sm.Age_Adjusted_Smoother(self.e1, self.b1, self.kw, self.s)
        np.testing.assert_allclose(ar.r, self.ageadj_exp)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_age_adjusted_mmoother_tabular(self):
        dfb = self.dfb
        kr = sm.Age_Adjusted_Smoother(dfb.e, dfb.b, s=dfb.s, w=self.kw)
        assert isinstance(kr.r, np.ndarray)
        np.testing.assert_allclose(kr.r, self.ageadj_exp)

    def test_disk_smoother(self):
        self.kw.transform = "b"
        disk = sm.Disk_Smoother(self.e, self.b, self.kw)
        np.testing.assert_allclose(disk.r.flatten(), self.disk_exp)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_disk_smoother_tabular(self):
        self.kw.transform = "b"
        dfa = self.dfa
        disk = sm.Disk_Smoother(dfa.e, dfa.b, self.kw).r
        np.testing.assert_allclose(disk.flatten(), self.disk_exp)

    def test_spatial_filtering(self):
        points = np.array(self.points)
        bbox = [[0, 0], [45, 45]]
        sf = sm.Spatial_Filtering(bbox, points, self.e, self.b, 2, 2, r=30)
        np.testing.assert_allclose(sf.r, self.sf_exp, rtol=RTOL, atol=ATOL)

    @pytest.mark.skipif(PANDAS_EXTINCT, reason="missing pandas")
    def test_spatial_filtering_tabular(self):
        point_array = np.array(self.points)
        bbox = [[0, 0], [45, 45]]
        dfa = self.dfa
        sf = sm.Spatial_Filtering(bbox, point_array, dfa.e, dfa.b, 2, 2, r=30)

        np.testing.assert_allclose(sf.r, self.sf_exp, rtol=RTOL, atol=ATOL)


class TestUtils:
    def test_sum_by_n(self):
        d = np.array([10, 9, 20, 30])
        w = np.array([0.5, 0.1, 0.3, 0.8])
        n = 2
        exp_sum = np.array([5.9, 30.0])
        np.testing.assert_array_almost_equal(exp_sum, sm.sum_by_n(d, w, n))

    def test_standardized_mortality_ratio(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
        s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])
        n = 2
        exp_smr = np.array([2.48691099, 2.73684211])
        np.testing.assert_array_almost_equal(
            exp_smr, sm.standardized_mortality_ratio(e, b, s_e, s_b, n)
        )

    def test_choynowski(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        n = 2
        exp_choy = np.array([0.30437751, 0.29367033])
        np.testing.assert_array_almost_equal(exp_choy, sm.choynowski(e, b, n))

    def test_assuncao_rate(self):
        e = np.array([30, 25, 25, 15, 33, 21, 30, 20])
        b = np.array([100, 100, 110, 90, 100, 90, 110, 90])
        exp_assuncao = np.array([1.03843594, -0.04099089, -0.56250375, -1.73061861])
        np.testing.assert_array_almost_equal(exp_assuncao, sm.assuncao_rate(e, b)[:4])
