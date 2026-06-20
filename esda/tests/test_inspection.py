import libpysal
import numpy as np
import pytest

from esda.getisord import G_Local
from esda.inspection import LocalCrossPlot
from esda.losh import LOSH
from esda.moran import Moran_Local

parametrize_w = pytest.mark.parametrize(
    "w",
    [
        libpysal.weights.lat2W(3, 3),
        libpysal.graph.Graph.from_W(libpysal.weights.lat2W(3, 3)),
    ],
    ids=["W", "Graph"],
)


def _y():
    return np.array([0.2, 1.5, 0.7, 2.1, 3.3, 1.2, 4.4, 2.8, 3.7])


@parametrize_w
def test_local_cross_plot_plot(w):
    import matplotlib

    matplotlib.use("Agg")

    y = _y()
    cross_plot = LocalCrossPlot(
        connectivity=w,
        permuations=9,
        n_jobs=1,
        seed=12345,
    ).fit(y)

    ax = cross_plot.plot(losh_scaling_factor=5)

    assert ax.get_xlabel() == r"Getis-Ord $G_i$"
    assert ax.get_ylabel() == "Moran's $I_i$"
    assert len(ax.collections) == 2

    plotted_sizes = np.concatenate([col.get_sizes() for col in ax.collections])
    expected_sizes = np.exp(cross_plot.losh_.Hi) * 5
    np.testing.assert_allclose(np.sort(plotted_sizes), np.sort(expected_sizes))


@parametrize_w
def test_local_cross_plot_from_estimators(w):
    y = _y()

    losh = LOSH(w).fit(y)
    moran_local = Moran_Local(
        y,
        w,
        permutations=9,
        n_jobs=1,
        seed=12345,
        keep_simulations=True,
    )
    g_local = G_Local(
        y,
        w,
        permutations=9,
        n_jobs=1,
        seed=12345,
    )

    cross_plot = LocalCrossPlot.from_estimators(g_local, moran_local, losh)

    ax = cross_plot.plot(losh_scaling_factor=5)

    assert ax.get_xlabel() == r"Getis-Ord $G_i$"
    assert ax.get_ylabel() == "Moran's $I_i$"
    assert len(ax.collections) == 2

    plotted_sizes = np.concatenate([col.get_sizes() for col in ax.collections])
    expected_sizes = np.exp(cross_plot.losh_.Hi) * 5
    np.testing.assert_allclose(np.sort(plotted_sizes), np.sort(expected_sizes))
