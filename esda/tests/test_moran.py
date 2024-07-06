import unittest

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pytest
from libpysal.common import ATOL, RTOL
from numpy.testing import assert_array_equal

from .. import moran

SEED = 12345

parametrize_stl = pytest.mark.parametrize(
    "w",
    [
        libpysal.io.open(libpysal.examples.get_path("stl.gal")).read(),
        libpysal.graph.Graph.from_W(
            libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        ),
    ],
    ids=["W", "Graph"],
)
parametrize_sids = pytest.mark.parametrize(
    "w",
    [
        libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read(),
        libpysal.graph.Graph.from_W(
            libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        ),
    ],
    ids=["W", "Graph"],
)

parametrize_lat3x3 = pytest.mark.parametrize(
    "w",
    [
        libpysal.weights.util.lat2W(3, 3),
        libpysal.graph.Graph.from_W(libpysal.weights.util.lat2W(3, 3)),
    ],
    ids=["W", "Graph"],
)
parametrize_desmith = pytest.mark.parametrize(
    "w",
    [
        libpysal.io.open(libpysal.examples.get_path("desmith.gal")).read(),
        libpysal.graph.Graph.from_W(
            libpysal.io.open(libpysal.examples.get_path("desmith.gal")).read()
        ),
    ],
    ids=["W", "Graph"],
)

sac1 = libpysal.examples.load_example("Sacramento1")
sac1 = gpd.read_file(sac1.get_path("sacramentot2.shp"))

parametrize_sac = pytest.mark.parametrize(
    "w",
    [
        libpysal.weights.Queen.from_dataframe(sac1),
        libpysal.graph.Graph.build_contiguity(sac1, rook=False),
    ],
    ids=["W", "Graph"],
)


class TestMoran:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    @parametrize_stl
    def test_moran(self, w):
        mi = moran.Moran(self.y, w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.24365582621771659, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(mi.p_norm, 0.00013573931385468807)

    @parametrize_sids
    def test_sids(self, w):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        SIDR = np.array(f.by_col("SIDR74"))
        mi = moran.Moran(SIDR, w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(mi.p_norm, 5.7916539074498452e-05)

    @parametrize_lat3x3
    def test_variance(self, w):
        y = np.arange(1, 10)
        mi = moran.Moran(y, w, transformation="B")
        np.testing.assert_allclose(
            mi.VI_rand, 0.059687500000000004, atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            mi.VI_norm, 0.053125000000000006, atol=ATOL, rtol=RTOL
        )

    @parametrize_stl
    def test_z_consistency(self, w):
        m1 = moran.Moran(self.y, w)
        # m2 = moran.Moran_BV(self.x, self.y, self.w) TODO testing for other.z values
        m3 = moran.Moran_Local(self.y, w, keep_simulations=True, seed=SEED)
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
        np.testing.assert_allclose(pval, 0.001)


class TestMoranRate:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col["SID79"])
        self.b = np.array(f.by_col["BIR79"])

    @parametrize_sids
    def test_moran_rate(self, w):
        mi = moran.Moran_Rate(self.e, self.b, w, two_tailed=False)
        np.testing.assert_allclose(mi.I, 0.16622343552567395, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(mi.p_norm, 0.004191499504892171)

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
        np.testing.assert_allclose(pval, 0.008)


class TestMoranBVmatrix:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        varnames = ["SIDR74", "SIDR79", "NWR74", "NWR79"]
        self.names = varnames
        vars = [np.array(f.by_col[var]) for var in varnames]
        self.vars = vars

    @parametrize_sids
    def test_Moran_BV_matrix(self, w):
        res = moran.Moran_BV_matrix(self.vars, w, varnames=self.names)
        np.testing.assert_allclose(res[(0, 1)].I, 0.19362610652874668)
        np.testing.assert_allclose(res[(3, 0)].I, 0.37701382542927858)


class TestMoranLocal:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("desmith.txt"))
        self.y = np.array(f.by_col["z"])

    @parametrize_desmith
    def test_Moran_Local(self, w):
        lm = moran.Moran_Local(
            self.y,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        np.testing.assert_allclose(lm.z_sim[0], -0.6990291160835514)
        np.testing.assert_allclose(lm.p_z_sim[0], 0.24226691753791396)

    @parametrize_sac
    def test_Moran_Local_labels(self, w):
        lm = moran.Moran_Local(
            sac1.HSG_VAL.values,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        expected_labels = np.array(
            [
                "High-High",
                "High-High",
                "Insignificant",
                "High-High",
                "Insignificant",
                "High-High",
                "High-High",
                "High-High",
                "Insignificant",
                "Insignificant",
            ]
        )
        assert_array_equal(lm.get_cluster_labels()[:10], expected_labels)
        assert_array_equal(
            pd.Series(lm.get_cluster_labels(0.05)).value_counts().values,
            np.array([277, 82, 38, 3, 3]),
        )

    @parametrize_sac
    def test_Moran_Local_explore(self, w):
        lm = moran.Moran_Local(
            sac1.HSG_VAL.values,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        m = lm.explore(sac1)
        np.testing.assert_array_equal(
            m.get_bounds(),
            [[38.018422, -122.422049], [39.316476, -119.877249]],
        )
        assert len(m.to_dict()["children"]) == 3

        out_str = _fetch_map_string(m)

        assert '"High-High","__folium_color":"#d7191c"' in out_str
        assert '"Low-High","__folium_color":"#89cff0"' in out_str
        assert '"Low-Low","__folium_color":"#2c7bb6"' in out_str
        assert '"High-Low","__folium_color":"#fdae61"' in out_str
        assert '"Insignificant","__folium_color":"#d3d3d3"' in out_str

        assert out_str.count("#d7191c") == 41
        assert out_str.count("#89cff0") == 6
        assert out_str.count("#2c7bb6") == 85
        assert out_str.count("#fdae61") == 6
        assert out_str.count("#d3d3d3") == 280

    @parametrize_desmith
    def test_Moran_Local_parallel(self, w):
        lm = moran.Moran_Local(
            self.y,
            w,
            transformation="r",
            n_jobs=-1,
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        np.testing.assert_allclose(lm.z_sim[0], -0.6990291160835514)
        np.testing.assert_allclose(lm.p_z_sim[0], 0.24226691753791396)

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
        np.testing.assert_allclose(lm.z_z_sim[0], -0.6990291160835514)
        np.testing.assert_allclose(lm.z_p_z_sim[0], 0.24226691753791396)

    @parametrize_desmith
    def test_local_moments(self, w):
        lm = moran.Moran_Local(
            self.y,
            w,
            transformation="r",
            permutations=0,
            seed=SEED,
        )

        wikh_fast = moran._wikh_fast(lm.w.sparse)
        wikh_slow = moran._wikh_slow(lm.w.sparse)
        # wikh_fast_c = moran._wikh_fast(lm.w.sparse, sokal_correction=True)
        # wikh_slow_c = moran._wikh_slow(lm.w.sparse, sokal_correction=True)

        np.testing.assert_allclose(wikh_fast, wikh_slow, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(wikh_fast, wikh_slow, rtol=RTOL, atol=ATOL)
        EIc = np.array(
            [
                -0.00838113,
                -0.0243949,
                -0.07031778,
                -0.21520869,
                -0.16547163,
                -0.00178435,
                -0.11531888,
                -0.36138555,
                -0.05471258,
                -0.09413562,
            ]
        )
        VIc = np.array(
            [
                0.03636013,
                0.10412408,
                0.28600769,
                0.26389674,
                0.21576683,
                0.00779261,
                0.44633942,
                0.57696508,
                0.12929777,
                0.3730742,
            ]
        )

        EI = -np.ones((10,)) / 9
        VI = np.array(
            [
                0.47374172,
                0.47356458,
                0.47209663,
                0.15866023,
                0.15972526,
                0.47376436,
                0.46927721,
                0.24584217,
                0.26498308,
                0.47077467,
            ]
        )

        np.testing.assert_allclose(lm.EIc, EIc, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.VIc, VIc, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.EI, EI, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lm.VI, VI, rtol=RTOL, atol=ATOL)


class TestMoranLocalBV:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.x = np.array(f.by_col["SIDR79"])
        self.y = np.array(f.by_col["SIDR74"])

    @parametrize_sids
    def test_Moran_Local_BV(self, w):
        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            w,
            keep_simulations=True,
            transformation="r",
            permutations=99,
            seed=SEED,
        )
        np.testing.assert_allclose(lm.Is[0], 1.4649221250620736)
        np.testing.assert_allclose(lm.z_sim[0], 1.330673752886702)
        np.testing.assert_allclose(lm.p_z_sim[0], 0.09164819151535242)

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
        np.testing.assert_allclose(bvstats[0], 1.4649221250620736)
        np.testing.assert_allclose(bvz[0], 1.7900932313425777, 5)
        np.testing.assert_allclose(bvzp[0], 0.036719462378528744, 5)


class TestMoranLocalRate:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.e = np.array(f.by_col["SID79"])
        self.b = np.array(f.by_col["BIR79"])

    @parametrize_sids
    def test_moran_rate(self, w):
        lm = moran.Moran_Local_Rate(
            self.e, self.b, w, transformation="r", permutations=99, seed=SEED
        )
        np.testing.assert_allclose(lm.z_sim[0], 0.02702781851384379, 7)
        np.testing.assert_allclose(lm.p_z_sim[0], 0.4892187730835096)

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
        np.testing.assert_allclose(lm["SID79-BIR79_z_sim"][0], 0.02702781851384379, 7)
        np.testing.assert_allclose(lm["SID79-BIR79_p_z_sim"][0], 0.4892187730835096)


def _fetch_map_string(m):
    out = m._parent.render()
    out_str = "".join(out.split())
    return out_str
