# Spatial Correlograms

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from libpysal.cg.kdtree import KDTree
from libpysal.weights import KNN, DistanceBand
from libpysal.weights.util import get_points_array

from .geary import Geary
from .getisord import G
from .moran import Moran


def _get_autocorrelation_stat(inputs):
    """helper function for computing parallel autocorrelation statistics

    Parameters
    ----------
    inputs : tuple
        tuple of (y, tree, W, statistic, STATISTIC, dist, weights_kwargs, stat_kwargs)

    Returns
    -------
    pandas.Series
        a pandas series with the computed autocorrelation statistic and its simulated p-value
    """
    (
        y,  # y variable
        tree,  # kd tree
        W,  # weights class DistanceBand or KNN
        STATISTIC,  # class of statistic (Moran, Geary, etc)
        dist,  # threshold/k parameter for the weights
        weights_kwargs,  # additional args
        stat_kwargs,  # additional args
    ) = inputs

    w = W(tree, dist, silence_warnings=True, **weights_kwargs)
    autocorr = STATISTIC(y, w, **stat_kwargs)
    attrs = []
    all_attrs = list(dict(vars(autocorr)).keys())
    for attribute in all_attrs:
        attrs.append(getattr(autocorr, str(attribute)))
    return pd.Series(attrs, index=all_attrs, name=dist)


def correlogram(
    gdf: gpd.GeoDataFrame,
    variable: str,
    distances: list,
    distance_type: str = "band",
    statistic: str = "I",
    weights_kwargs: dict = None,
    stat_kwargs: dict = None,
    n_jobs: int = -1,
    backend: str = "loky",
):
    """Generate a spatial correlogram

    A spatial correlogram is a set of spatial autocorrelation statistics calculated for
    a set of increasing distances. It is a useful exploratory tool for examining
    how the relationship between spatial units changes over different notions of scale.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geodataframe holding spatial and attribute data
    variable: str
        column on the geodataframe used to compute autocorrelation statistic
    distances : list
        list of distances to compute the autocorrelation statistic
    distance_type : str, optional
        which concept of distance to increment. Options are {`band`, `knn`}.
        by default 'band' (for `libpysal.weights.DistanceBand` weights)
    statistic : str, by default 'I'
        which spatial autocorrelation statistic to compute. Options in {`I`, `G`, `C`}
    weights_kwargs : dict
        additional keyword arguments passed to the libpysal.weights.W class
    stat_kwargs : dict
        additional keyword arguments passed to the `esda` autocorrelation statistic class.
        For example for faster results with no statistical inference, set the number
        of permutations to zero with {permutations: 0}
    n_jobs : int
        number of jobs to pass to joblib. If -1 (default), all cores will be used
    backend : str
        backend parameter passed to joblib

    Returns
    -------
    outputs : pandas.DataFrame
        table of autocorrelation statistics at increasing distance bandwidths
    """
    if stat_kwargs is None:
        stat_kwargs = dict()
    if weights_kwargs is None:
        weights_kwargs = dict()
    if statistic == "I":
        STATISTIC = Moran
    elif statistic == "G":
        STATISTIC = G
    elif statistic == "C":
        STATISTIC = Geary
    else:
        with NotImplementedError as e:
            raise e("Only I, G, and C statistics are currently implemented")

    if distance_type == "band":
        W = DistanceBand
    elif distance_type == "knn":
        if max(distances) > gdf.shape[0] - 1:
            with ValueError as e:
                raise e("max number of neighbors must be less than or equal to n-1")
        W = KNN
    else:
        with NotImplementedError as e:
            raise e("distance_type must be either `band` or `knn` ")

    #  should be able to build the tree once and reuse it?
    #  but in practice, im not seeing any real difference from starting a new W from scratch each time
    pts = get_points_array(gdf[gdf.geometry.name])
    tree = KDTree(pts)
    y = gdf[variable].values

    inputs = [
        tuple(
            [
                y,
                tree,
                W,
                STATISTIC,
                dist,
                weights_kwargs,
                stat_kwargs,
            ]
        )
        for dist in distances
    ]

    outputs = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_get_autocorrelation_stat)(i) for i in inputs
    )

    return (
        pd.DataFrame(outputs)
        .select_dtypes(["number"])
        .drop(columns=["permutations", "n"])
    )


## Note: To be implemented:

## non-parametric version used in geoda https://geodacenter.github.io/workbook/5a_global_auto/lab5a.html#spatial-correlogram
## as given in https://link.springer.com/article/10.1023/A:1009601932481'
