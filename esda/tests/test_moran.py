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

    @pytest.mark.skip("This function is being deprecated in the next release.")
    def test_by_col(self):
        from libpysal.io import geotable as pdio

        df = pdio.read_files(libpysal.examples.get_path("sids2.dbf"))
        w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
        mi = moran.Moran.by_col(df, ["SIDR74"], w=w, two_tailed=False)
        sidr = np.unique(mi.SIDR74_moran.values).item()
        pval = np.unique(mi.SIDR74_p_sim.values).item()
        np.testing.assert_allclose(sidr, 0.24772519320480135, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(pval, 0.001)

    @parametrize_sac
    def test_plot_simulation(self, w):
        pytest.importorskip("seaborn")

        m = moran.Moran(sac1.WHITE, w=w)
        ax = m.plot_simulation()

        assert len(ax.collections) == 3

        kde = ax.collections[0]
        np.testing.assert_array_almost_equal(
            kde.get_facecolor(),
            [[0.7294117647058823, 0.7294117647058823, 0.7294117647058823, 0.25]],
        )
        assert kde.get_fill()
        assert len(kde.get_paths()[0]) == 403

        i_vline = ax.collections[1]
        np.testing.assert_array_almost_equal(
            i_vline.get_color(),
            [[0.8392156862745098, 0.3764705882352941, 0.30196078431372547, 1.0]],
        )
        assert i_vline.get_label() == "Moran's I"
        np.testing.assert_array_almost_equal(
            i_vline.get_paths()[0].vertices,
            np.array([[m.I, 0.0], [m.I, 1.0]]),
        )

        ei_vline = ax.collections[2]
        np.testing.assert_array_almost_equal(
            ei_vline.get_color(),
            [[0.12156863, 0.46666667, 0.70588235, 1.0]],
        )
        assert ei_vline.get_label() == "Expected I"
        np.testing.assert_array_almost_equal(
            ei_vline.get_paths()[0].vertices,
            np.array([[m.EI, 0.0], [m.EI, 1.0]]),
        )

    @parametrize_sac
    def test_plot_simulation_custom(self, w):
        pytest.importorskip("seaborn")
        plt = pytest.importorskip("matplotlib.pyplot")

        m = moran.Moran(sac1.WHITE, w=w)

        _, ax = plt.subplots(figsize=(12, 12))
        ax = m.plot_simulation(
            ax=ax, fitline_kwds={"color": "red"}, color="pink", shade=False, legend=True
        )

        assert len(ax.collections) == 2
        assert len(ax.lines) == 1

        kde = ax.lines[0]
        np.testing.assert_array_almost_equal(
            kde.get_color(),
            [1.0, 0.75294118, 0.79607843, 1],
        )
        assert len(kde.get_path()) == 200

        i_vline = ax.collections[0]
        np.testing.assert_array_almost_equal(
            i_vline.get_color(),
            [[1.0, 0.0, 0.0, 1.0]],
        )
        assert i_vline.get_label() == "Moran's I"
        np.testing.assert_array_almost_equal(
            i_vline.get_paths()[0].vertices,
            np.array([[m.I, 0.0], [m.I, 1.0]]),
        )

        ei_vline = ax.collections[1]
        np.testing.assert_array_almost_equal(
            ei_vline.get_color(),
            [[0.12156863, 0.46666667, 0.70588235, 1.0]],
        )
        assert ei_vline.get_label() == "Expected I"
        np.testing.assert_array_almost_equal(
            ei_vline.get_paths()[0].vertices,
            np.array([[m.EI, 0.0], [m.EI, 1.0]]),
        )

        assert ax.get_legend_handles_labels()[1] == [
            "Distribution of simulated Is",
            "Moran's I",
            "Expected I",
        ]

    @parametrize_sac
    def test_plot_scatter(self, w):
        import matplotlib

        matplotlib.use("Agg")

        m = moran.Moran(
            sac1.WHITE,
            w,
        )

        ax = m.plot_scatter()

        # test scatter
        np.testing.assert_array_almost_equal(
            ax.collections[0].get_facecolors(),
            np.array([[0.729412, 0.729412, 0.729412, 0.6]]),
        )

        # test fitline
        l_ = ax.lines[2]
        x, y = l_.get_data()
        np.testing.assert_almost_equal(x.min(), -1.8236414387225368)
        np.testing.assert_almost_equal(x.max(), 3.893056527659032)
        np.testing.assert_almost_equal(y.min(), -0.7371749399524187)
        np.testing.assert_almost_equal(y.max(), 1.634939204358587)
        assert l_.get_color() == "#d6604d"

    @parametrize_sac
    def test_plot_scatter_args(self, w):
        import matplotlib

        matplotlib.use("Agg")

        m = moran.Moran(
            sac1.WHITE,
            w,
        )

        ax = m.plot_scatter(
            scatter_kwds=dict(color="blue"), fitline_kwds=dict(color="pink")
        )

        # test scatter
        np.testing.assert_array_almost_equal(
            ax.collections[0].get_facecolors(),
            np.array([[0, 0, 1, 0.6]]),
        )

        # test fitline
        l_ = ax.lines[2]
        assert l_.get_color() == "pink"


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

    @pytest.mark.skip("This function is being deprecated in the next release.")
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
        vars_ = [np.array(f.by_col[var]) for var in varnames]
        self.vars_ = vars_

    @parametrize_sids
    def test_defaults(self, w):
        res = moran.Moran_BV_matrix(self.vars_, w, varnames=self.names)
        np.testing.assert_allclose(res[(0, 1)].I, 0.19362610652874668)
        np.testing.assert_allclose(res[(3, 0)].I, 0.37701382542927858)

    @parametrize_sids
    def test_plot_moran_facet(self, w):
        matrix = moran.Moran_BV_matrix(self.vars_, w, varnames=self.names)
        axes = moran.plot_moran_facet(matrix)
        assert axes.shape == (4, 4)

        assert axes[0][0].spines["left"].get_visible()
        assert not axes[0][0].spines["bottom"].get_visible()
        assert axes[3][0].spines["left"].get_visible()
        assert axes[3][0].spines["bottom"].get_visible()
        assert not axes[3][1].spines["left"].get_visible()
        assert axes[3][1].spines["bottom"].get_visible()
        assert not axes[1][1].spines["left"].get_visible()
        assert not axes[1][1].spines["bottom"].get_visible()

        np.testing.assert_array_almost_equal(
            axes[1][1].get_facecolor(),
            (0.8509803921568627, 0.8509803921568627, 0.8509803921568627, 1.0),
        )


class TestMoranLocal:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("desmith.txt"))
        self.y = np.array(f.by_col["z"])

    @parametrize_desmith
    def test_defaults(self, w):
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
    def test_labels(self, w):
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
    def test_explore(self, w):
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

    @parametrize_sac
    def test_plot(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local(
            sac1.HSG_VAL.values,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        ax = lm.plot(sac1)
        unique, counts = np.unique(
            ax.collections[0].get_facecolors(), axis=0, return_counts=True
        )
        np.testing.assert_array_almost_equal(
            unique,
            np.array(
                [
                    [0.17254902, 0.48235294, 0.71372549, 1.0],
                    [0.5372549, 0.81176471, 0.94117647, 1.0],
                    [0.82745098, 0.82745098, 0.82745098, 1.0],
                    [0.84313725, 0.09803922, 0.10980392, 1.0],
                    [0.99215686, 0.68235294, 0.38039216, 1.0],
                ]
            ),
        )
        np.testing.assert_array_equal(counts, np.array([86, 3, 298, 38, 3]))

    @parametrize_sac
    def test_plot_scatter(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local(
            sac1.WHITE,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )

        ax = lm.plot_scatter()

        # test scatter
        unique, counts = np.unique(
            ax.collections[0].get_facecolors(), axis=0, return_counts=True
        )
        np.testing.assert_array_almost_equal(
            unique,
            np.array(
                [
                    [0.17254902, 0.48235294, 0.71372549, 0.6],
                    [0.5372549, 0.81176471, 0.94117647, 0.6],
                    [0.82745098, 0.82745098, 0.82745098, 0.6],
                    [0.84313725, 0.09803922, 0.10980392, 0.6],
                    [0.99215686, 0.68235294, 0.38039216, 0.6],
                ]
            ),
        )
        np.testing.assert_array_equal(counts, np.array([73, 12, 261, 52, 5]))

        # test fitline
        l_ = ax.lines[2]
        x, y = l_.get_data()
        np.testing.assert_almost_equal(x.min(), -1.8236414387225368)
        np.testing.assert_almost_equal(x.max(), 3.893056527659032)
        np.testing.assert_almost_equal(y.min(), -0.7371749399524187)
        np.testing.assert_almost_equal(y.max(), 1.634939204358587)
        assert l_.get_color() == "k"

    @parametrize_sac
    def test_plot_scatter_args(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local(
            sac1.WHITE,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )

        ax = lm.plot_scatter(
            crit_value=None,
            scatter_kwds={"s": 10},
            fitline_kwds={"linewidth": 4},
        )
        # test scatter
        np.testing.assert_array_almost_equal(
            ax.collections[0].get_facecolors(),
            np.array([[0.729412, 0.729412, 0.729412, 0.6]]),
        )
        assert ax.collections[0].get_sizes()[0] == 10

        # test fitline
        l_ = ax.lines[2]
        assert l_.get_color() == "#d6604d"
        assert l_.get_linewidth() == 4.0

    @parametrize_desmith
    def test_parallel(self, w):
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

    @pytest.mark.skip("This function is being deprecated in the next release.")
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

    @parametrize_sac
    def test_plot_combination(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local(
            sac1.WHITE,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        axs = lm.plot_combination(
            sac1,
            "WHITE",
            legend_kwds=dict(loc="lower right"),
            region_column="FIPS",
            mask=["06067009504", "06067009503"],
            quadrant=1,
        )

        assert len(axs) == 3
        assert len(axs[0].patches) == 1
        assert len(axs[1].collections) == 4
        assert len(axs[2].collections) == 4

        axs2 = lm.plot_combination(
            sac1,
            "WHITE",
            legend_kwds=dict(loc="lower right"),
        )

        assert len(axs2) == 3
        assert len(axs2[0].patches) == 0
        assert len(axs2[1].collections) == 1
        assert len(axs2[2].collections) == 1


class TestMoranLocalBV:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
        self.gdf = gpd.read_file(libpysal.examples.get_path("sids2.shp"))
        self.x = np.array(f.by_col["SIDR79"])
        self.y = np.array(f.by_col["SIDR74"])

    @parametrize_sids
    def test_defaults(self, w):
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

    @pytest.mark.skip("This function is being deprecated in the next release.")
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

    @parametrize_sids
    def test_labels(self, w):
        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        expected_labels = np.array(
            [
                "Insignificant",
                "Insignificant",
                "Low-Low",
                "High-Low",
                "Low-High",
                "Insignificant",
                "Insignificant",
                "Insignificant",
                "Insignificant",
                "Insignificant",
            ]
        )
        assert_array_equal(lm.get_cluster_labels()[:10], expected_labels)
        assert_array_equal(
            pd.Series(lm.get_cluster_labels(0.05)).value_counts().values,
            np.array([80, 7, 6, 5, 2]),
        )

    @parametrize_sids
    def test_explore(self, w):
        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        m = lm.explore(self.gdf)
        np.testing.assert_array_equal(
            m.get_bounds(),
            [
                [33.88199234008789, -84.3238525390625],
                [36.58964920043945, -75.45697784423828],
            ],
        )
        assert len(m.to_dict()["children"]) == 2

        out_str = _fetch_map_string(m)

        assert '"High-High","__folium_color":"#d7191c"' in out_str
        assert '"Low-High","__folium_color":"#89cff0"' in out_str
        assert '"Low-Low","__folium_color":"#2c7bb6"' in out_str
        assert '"High-Low","__folium_color":"#fdae61"' in out_str
        assert '"Insignificant","__folium_color":"#d3d3d3"' in out_str

        assert out_str.count("#d7191c") == 10
        assert out_str.count("#89cff0") == 5
        assert out_str.count("#2c7bb6") == 9
        assert out_str.count("#fdae61") == 8
        assert out_str.count("#d3d3d3") == 83

    @parametrize_sids
    def test_plot(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        ax = lm.plot(self.gdf)
        unique, counts = np.unique(
            ax.collections[0].get_facecolors(), axis=0, return_counts=True
        )
        np.testing.assert_array_almost_equal(
            unique,
            np.array(
                [
                    [0.17254902, 0.48235294, 0.71372549, 1.0],
                    [0.5372549, 0.81176471, 0.94117647, 1.0],
                    [0.82745098, 0.82745098, 0.82745098, 1.0],
                    [0.84313725, 0.09803922, 0.10980392, 1.0],
                    [0.99215686, 0.68235294, 0.38039216, 1.0],
                ]
            ),
        )
        np.testing.assert_array_equal(counts, np.array([6, 2, 86, 7, 7]))

    @parametrize_sids
    def test_plot_combination(self, w):
        import matplotlib

        matplotlib.use("Agg")

        lm = moran.Moran_Local_BV(
            self.x,
            self.y,
            w,
            transformation="r",
            permutations=99,
            keep_simulations=True,
            seed=SEED,
        )
        axs = lm.plot_combination(
            self.gdf,
            "SIDR79",
            legend_kwds=dict(loc="lower right"),
            quadrant=1,
        )

        assert len(axs) == 3
        assert len(axs[0].patches) == 1
        assert len(axs[1].collections) == 3
        assert len(axs[2].collections) == 3


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

    @pytest.mark.skip("This function is being deprecated in the next release.")
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
