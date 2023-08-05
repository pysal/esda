# Spatial Correlograms

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from libpysal.cg.kdtree import KDTree
from libpysal.weights import KNN, DistanceBand
from libpysal.weights.util import get_points_array
from esda.moran import Moran


def _get_stat(inputs):
    """helper function for computing parallel statistics at multiple Graph specifications

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
        tree,  # kdreee,
        W,  # weights class
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
    statistic: callable = Moran,
    distance_type: str = "band",
    weights_kwargs: dict = None,
    stat_kwargs: dict = None,
    select_numeric: bool = False,
    n_jobs: int = -1,
):
    """Generate a spatial correlogram

    A spatial profile is a set of spatial autocorrelation statistics calculated for
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
    statistic : callable
        statistic to be computed for a range of libpysal.Graph specifications.
        This should be a class with a signature like `Statistic(y,w, **kwargs)`
        where y is a numpy array and w is a libpysal.Graph.
        Generally, this is a class from pysal's `esda` package
        defaults to esda.Moran, which computes the Moran's I statistic
    distance_type : str, optional
        which concept of distance to increment. Options are {`band`, `knn`}.
        by default 'band' (for `libpysal.weights.DistanceBand` weights)
    weights_kwargs : dict
        additional keyword arguments passed to the libpysal.weights.W class
    stat_kwargs : dict
        additional keyword arguments passed to the `esda` autocorrelation statistic class.
        For example for faster results with no statistical inference, set the number
        of permutations to zero with {permutations: 0}
    select_numeric : bool
        if True, only return numeric attributes from the original class. This is useful
        e.g. to prevent lists inside a "cell" of a dataframe
    n_jobs : int
        number of jobs to pass to joblib. If -1 (default), all cores will be used


    Returns
    -------
    outputs : pandas.DataFrame
        table of autocorrelation statistics at increasing distance bandwidths
    """
    if stat_kwargs is None:
        stat_kwargs = dict()
    if weights_kwargs is None:
        weights_kwargs = dict()

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

    # there's a faster way to do this by building the largest tree first, then subsetting...
    pts = get_points_array(gdf[gdf.geometry.name])
    tree = KDTree(pts)
    y = gdf[variable].values

    inputs = [
        (
                y,
                tree,
                W,
                statistic,
                dist,
                weights_kwargs,
                stat_kwargs,
            )
        for dist in distances
    ]

    outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_get_stat)(i) for i in inputs
    )

    df = pd.DataFrame(outputs)
    if select_numeric:
        df = df.select_dtypes(["number"])
    return df


## Note: To be implemented:

## non-parametric version used in geoda https://geodacenter.github.io/workbook/5a_global_auto/lab5a.html#spatial-correlogram
## as given in https://link.springer.com/article/10.1023/A:1009601932481'
