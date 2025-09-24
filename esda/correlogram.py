# Spatial Correlograms
from collections.abc import Callable

import geopandas as gpd
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from libpysal.cg.kdtree import KDTree
from libpysal.weights import KNN, DistanceBand, min_threshold_distance
from libpysal.weights.util import get_points_array
from sklearn.metrics import pairwise_distances
from scipy import spatial, linalg

from .moran import Moran


def _get_stat(inputs: tuple) -> pd.Series:
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
    variable: str | list | pd.Series | None,
    distances: list | None = None,
    statistic: Callable | str = Moran,
    distance_type: str = "band",
    weights_kwargs: dict = None,
    stat_kwargs: dict = None,
    select_numeric: bool = False,
    n_jobs: int = -1,
    n_bins: int | None = 10,
) -> pd.DataFrame:
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
    distances : list or None
        list of distances at which to compute the autocorrelation statistic
    statistic : callable | str
        statistic to be computed for a range of libpysal.Graph specifications.
        This should be a class with a signature like `Statistic(y,w, **kwargs)`
        where y is a  array and w is a libpysal.Graph.
        Generally, this is a class from pysal's `esda` package
        defaults to esda.Moran, which computes the Moran's I statistic. If
        'lowess' is provided, a non-parametric correlogram is computed using
        lowess regression on the spatial-covariation model, see Notes.
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

    Notes
    -----
    The nonparametric correlogram uses a lowess regression
    to estimate the spatial-covariation model:

        zi*zj = f(d_{ij}) + e_ij

    where f is a smooth function of distance d_{ij} between points i and j.
    This function requires the statsmodels package to be installed.

    For the nonparametric correlogram, a precomputed distance matrix can
    be used. To do this, set
    stat_kwargs={'metric':'precomputed', 'coordinates':distance_matrix}
    where `distance_matrix` is a square matrix of pairwise distances that
    aligns with the `gdf` rows.
    """
    if stat_kwargs is None:
        stat_kwargs = dict()
    if weights_kwargs is None:
        weights_kwargs = dict()

    # there's a faster way to do this by building the largest tree first, then subsetting...
    pts = get_points_array(gdf[gdf.geometry.name])
    tree = KDTree(pts)

    if distances is None:
        if distance_type == 'band':
            stop = spatial.distance.cdist(
                tree.maxes.reshape(1, 2), tree.mins.reshape(1, 2), "euclidean"
            ).item()
            start = min_threshold_distance(tree.data)
            distances = np.linspace(start, stop, n_bins).squeeze().tolist()
        else:
            n_samples = gdf.shape[0]
            step = n_samples // n_bins
            # not guaranteed to be n_bins if n_samples not divisible by n_bins
            distances = np.arange(1, n_samples, step).astype(int)
    if distance_type == "band":
        W = DistanceBand
    elif distance_type == "knn":
        if max(distances) > (gdf.shape[0] - 1):
            raise ValueError("max number of neighbors must be less than or equal to n-1")
        W = KNN
    else:
        raise ValueError("distance_type must be either `band` or `knn`")

    if isinstance(variable, str):
        y = gdf[variable].values
    else:
        y = np.asarray(variable).squeeze()

    if y.shape[0] != gdf.shape[0]:
        raise ValueError(f"variable is length {len(y)} but gdf has {gdf.shape[0]} rows")

    if statistic != "lowess":
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
    elif statistic == "lowess":
        # lowess correlogram
        outputs = _lowess_correlogram(y, pts, xvals=distances, **stat_kwargs)
    else:
        raise ValueError(
            f"statistic must be a callable or 'lowess', recieved {statistic}"
        )

    df = pd.DataFrame(outputs)
    if select_numeric:
        df = df.select_dtypes(["number"])
    return df


def _lowess_correlogram(
    y: np.ndarray,
    coordinates: np.ndarray,
    xvals: np.ndarray,
    metric: str = "euclidean",
    **lowess_args,
) -> pd.DataFrame:
    """
    Compute a nonparametric correlogram using a kernel regression
    on the spatial-covariation model:

        zi*zj = f(d_{ij}) + e_ij

    where f is a smooth function of distance d_{ij} between points i and j.

    Arguments
    ---------
    y : array-like
        1D array of values to compute the correlogram on
    coordinates : array-like
        2D array of point coordinates or a precomputed distance matrix
    xvals : array-like
        1D array of distance values to evaluate the correlogram at
    metric : str
        distance metric to use. Any metric from sklearn.metrics.pairwise_distances
        is allowed. If 'precomputed', then coordinates is assumed to be a distance matrix
    lowess_args : keyword arguments
        additional keyword arguments passed to statsmodels.nonparametric.smoothers_lowess.lowess

    Returns
    -------
    pandas.DataFrame
        dataframe with index of xvals and a single column 'lowess' with the smoothed
        correlogram values

    Notes
    -----
    This function requires the statsmodels package to be installed. Further, no
    validation is done on the input parameters.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as e:
        raise ImportError("Nonparametric correlograms require statsmodels") from e

    if metric == "precomputed":
        d = coordinates  # assume this is a distance matrix
    else:
        d = pairwise_distances(coordinates, metric=metric)

    z = (y - y.mean()) / y.std()
    cov = np.multiply.outer(z, z)

    lowess_args.setdefault("frac", 0.33)
    if metric != "precomputed":
        row, col = np.triu_indices_from(cov)
        smooth = lowess(cov[row, col], d[row, col], xvals=xvals, **lowess_args)
    else:  # can't use upper triangle if d is not symmetric
        if linalg.issymetric(d):
            row, col = np.triu_indices_from(cov)
            smooth = lowess(cov[row, col], d[row, col], xvals=xvals, **lowess_args)
        else:
            smooth = lowess(cov, d, xvals=xvals, **lowess_args)

    return pd.DataFrame(smooth, index=xvals, columns=["lowess"])
