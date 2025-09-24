# Spatial Correlograms

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from libpysal.cg.kdtree import KDTree
from libpysal.weights import KNN, DistanceBand
from libpysal.graph import Graph
from libpysal.weights.util import get_points_array
from sklearn.metrics import pairwise_distances
from scipy import spatial

from .moran import Moran


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
    distances: list | None = None,
    statistic: callable = Moran,
    distance_type: str = "band",
    weights_kwargs: dict = None,
    stat_kwargs: dict = None,
    select_numeric: bool = False,
    n_jobs: int = -1,
    n_bins : int | None = 10,
    parametric = True
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
    distances : list or None
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
            raise e("distance_type must be either `band` or `knn`")

    # there's a faster way to do this by building the largest tree first, then subsetting...
    pts = get_points_array(gdf[gdf.geometry.name])
    tree = KDTree(pts)

    if distances is None:
        stop = spatial.distance.cdist(t.maxes.reshape(1,2), t.mins.reshape(1,2), 'euclidean')
        start = libpysal.weights.min_threshold_distances(tree) 
        distances = numpy.linspace(start, stop, n_bins).tolist()

    if isinstance(variable, str):
        y = gdf[variable].values
    else:
        y = numpy.asarray(variable).squeeze()
        
    if len(y) != gdf.shape[0]:
        raise ValueError(f"variable is length {len(y)} but gdf has {gdf.shape[0]} rows")

    if parametric:
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
    else:
        # non-parametric correlogram
        outputs = _nonparametric_correlogram(y, pts, metric=distance_type, xvals=distances, **stat_kwargs)

    df = pd.DataFrame(outputs)
    if select_numeric:
        df = df.select_dtypes(["number"])
    return df

def _nonparametric_correlogram(y,coordinates, metric, xvals, **lowess_args):
    """
    Compute a nonparametric correlogram using a kernel regression 
    on the spatial-covariation model:
    
        zi*zj = f(d_{ij}) + e_ij
    
    where f is a smooth function of distance d_{ij} between points i and j.
    """
    try:
        from statsmodels.nonparametric import lowess
    except ImportError as e:
        raise ImportError("Nonparametric correlograms require statsmodels") from e
    
    # TODO: if metric='precomputed' then coordiantes is a distance matrix
    # we have this somewhere else I know it... I wrote it! Where is it...

    if metric=='precomputed':
        d = coordinates # assume this is a distance matrix  
    else:
        d = pairwise_distances(coordinates, metric=metric)

    z = (y - y.mean())/y.std()
    cov = numpy.multiply.outer(z,z)

    smooth = lowess(cov.flatten(), d.flatten(), xvals=xvals, **lowess_args)

    return smooth
