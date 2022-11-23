import numpy
import pandas
from libpysal import weights
from scipy.spatial import distance
from scipy.stats import mode as most_common_value
from sklearn.utils import check_array

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _passthrough(sequence):
    return sequence


def _resolve_metric(X, coordinates, metric):
    """
    Provide a distance function that you can use
    to find the distance betwen arbitrary points.
    """
    if callable(metric):
        distance_func = metric
    elif metric.lower() == "haversine":
        try:
            from numba import autojit
        except ImportError:

            def autojit(func):
                return func

        @autojit
        def harcdist(p1, p2):
            """Compute the kernel of haversine"""
            x = numpy.sin(p2[1] - p1[1] / 2) ** 2
            cosp1, cosp2 = numpy.cos(p1[1]), numpy.cos(p2[1])
            y = cosp2 * cosp1 * numpy.sin((p2[0] - p1[0]) / 2) ** 2
            return 2 * numpy.arcsin(numpy.sqrt(x + y))

        distance_func = harcdist
    elif metric.lower() == "precomputed":
        # so, in this case, coordinates is actually distance matrix of some kind
        # and is assumed aligned to X, such that the distance from X[a] to X[b] is
        # coordinates[a,b], confusingly... So, we'll re-write them as "distances"
        distances = check_array(coordinates, accept_sparse=True)
        n, k = distances.shape
        assert k == n, (
            "With metric='precomputed', coordinates must be an (n,n)"
            " matrix representing distances between coordinates."
        )

        def lookup_distance(a, b):
            """Find location of points a,b in X and return precomputed distances"""
            (aloc,) = (X == a).all(axis=1).nonzero()
            (bloc,) = (X == b).all(axis=1).nonzero()
            if (len(aloc) > 1) or (len(bloc) > 1):
                raise NotImplementedError(
                    "Precomputed distances cannot disambiguate coincident points."
                    " Add a slight bit of noise to the input to force them"
                    " into non-coincidence and re-compute the distance matrix."
                )
            elif (len(aloc) == 0) or (len(bloc) == 0):
                raise NotImplementedError(
                    "Precomputed distances cannot compute distances to new points."
                )
            return distances[aloc, bloc]

        distance_func = lookup_distance
    else:
        try:
            distance_func = getattr(distance, metric)
        except AttributeError:
            raise KeyError(
                f"Metric '{metric}' not understood. Choose "
                "something available in scipy.spatial.distance."
            )
    return distance_func


def isolation(
    X,
    coordinates,
    metric="euclidean",
    middle="mean",
    return_all=False,
    progressbar=False,
):
    """
    Compute the isolation of each value of X by constructing the distance
    to the nearest higher value in the data.

    Parameters
    ----------
    X : numpy.ndarray
        (N, p) array of data to use as input. If p > 1, the "elevation" is computed
        using the topo.to_elevation function.
    coordinates : numpy.ndarray
        (N,k) array of locations for X to compute distances. If
        metric='precomputed', this should contain the distances from
        each point to every other point, and k == N.
    metric : string or callable (default: 'euclidean')
        name of distance metric in scipy.spatial.distance, or function, that can be
        used to compute distances between locations. If 'precomputed', ad-hoc function
        will be defined to look up distances between points instead.
    middle : string or callable (default: 'mean')
        method to define the elevation of points. See to_elevation for more details.
    return_all : bool (default: False)
        if False, only return the isolation (distance to nearest higher value).
    progressbar: bool (default: False)
        if True, show a progressbar for the computation.
    Returns
    -------
    either (N,) array of isolation values, or a pandas dataframe containing the full
    tree of precedence for the isolation tree.
    """
    X = check_array(X, ensure_2d=False)
    X = to_elevation(X, middle=middle).squeeze()
    try:
        from rtree.index import Index as SpatialIndex
    except ImportError:
        raise ImportError(
            "rtree library must be installed to use the prominence measure"
        )
    distance_func = _resolve_metric(X, coordinates, metric)
    sort_order = numpy.argsort(-X)
    tree = SpatialIndex()
    ix = sort_order[0]
    tree.insert(0, tuple(coordinates[ix]), obj=X[ix])
    precedence_tree = [[ix, numpy.nan, 0, numpy.nan, numpy.nan, numpy.nan]]

    if progressbar and HAS_TQDM:
        pbar = tqdm
    elif progressbar and (not HAS_TQDM):
        raise ImportError("The `tqdm` module is required for progressbars.")
    else:
        pbar = _passthrough

    for iter_ix, ix in pbar(enumerate(sort_order[1:])):
        rank = iter_ix + 1
        value = X[ix]
        location = coordinates[
            ix,
        ]
        (match,) = tree.nearest(tuple(location), objects=True)
        higher_rank = match.id
        higher_value = match.object
        higher_location = match.bbox[:2]
        higher_ix = sort_order[higher_rank]
        distance = distance_func(location, higher_location)
        gap = higher_value - value
        precedence_tree.append([ix, higher_ix, rank, higher_rank, distance, gap])
        tree.insert(rank, tuple(location), obj=value)
    # return precedence_tree
    precedence_tree = numpy.asarray(precedence_tree)
    # print(precedence_tree.shape)
    out = numpy.empty_like(precedence_tree)
    out[sort_order] = precedence_tree
    result = pandas.DataFrame(
        out,
        columns=["index", "parent_index", "rank", "parent_rank", "isolation", "gap"],
    ).sort_values(["index", "parent_index"])
    if return_all:
        return result
    else:
        return result.isolation.values


def prominence(
    X,
    connectivity,
    return_all=False,
    gdf=None,
    verbose=False,
    middle="mean",
    progressbar=False,
):
    """
    Return the prominence of peaks in input, given a connectivity matrix.

    Parameters
    ----------
    X : numpy.ndarray
        an array of shape N,p containing data to use for computing prominence. When
        p > 1, X will be converted to an "elevation" using to_elevation.
    connectivity : scipy.sparse matrix
        a sparse matrix encoding the connectivity graph pertaining to rows of X. If
        coordinates are provided, they must be (N,2), and the delaunay triangulation
        will be computed.
    return_class : bool (default: False)
        whether or not to return additional information about the result, such as
        the set of dominating peaks or the set of classifications for each observation.
    verbose : bool (default: None)
        whether or not to print extra information about the progress of the algorithm.
    middle : string or callable (default: "mean")
        how to compute the center of mass from X, when the dimension of X > 2.

    Returns
    -------
    the prominence of each observation in X, possibly along with the
    set of saddle points, peaks, and/or dominating peak tree.

    Notes
    -----
    An observation has 0 prominence when it is a saddle point.
    An observation has positive prominence when it is a peak, and this
    is computed as the elevation of the peak minus the elevation of the saddle point.

    Observations have "NA" prominence when they are neither a saddle point nor a peak.

    """
    X = check_array(X, ensure_2d=False).squeeze()
    X = to_elevation(X, middle=middle).squeeze()
    (n,) = X.shape

    if not isinstance(verbose, (bool, int)):
        gdf = verbose
        verbose = True
    else:
        gdf = None

    connectivity = _check_connectivity(connectivity)

    # sort the variable in descending order
    sort_order = numpy.argsort(-X)

    peaks = []
    assessed_peaks = set()
    prominence = numpy.empty_like(X) * numpy.nan
    dominating_peak = numpy.ones_like(X) * -1
    predecessors = numpy.ones_like(X) * -1
    keycols = numpy.ones_like(X) * -1
    ids = numpy.arange(n)
    classifications = [None] * n
    key_cols = dict()

    if progressbar and HAS_TQDM:
        pbar = tqdm
    elif progressbar and (not HAS_TQDM):
        raise ImportError("The `tqdm` module is required for progressbars.")
    else:
        pbar = _passthrough

    for rank, value in pbar(enumerate(X[sort_order])):
        # This is needed to break ties in the same way that argsort does. A more
        # natural way to do this is to use X >= value, but if value is tied, then
        # that would generate a mask where too many elements are selected!
        # e.g. mask.sum() > rank
        mask = numpy.isin(numpy.arange(n), sort_order[: rank + 1])
        (full_indices,) = mask.nonzero()
        this_full_ix = (ids[sort_order])[rank]
        msg = f"assessing {this_full_ix} (rank: {rank}, value: {value})"

        # use the dominating_peak vector. A new obs either has:
        #   Neighbors whose dominating_peak...
        #       1. ... are all -1 (new peak)
        #       2. ... are all -1 or an integer (slope of current peak)
        #       3. ... include at least two integers and any -1 (key col)
        _, neighbs = connectivity[this_full_ix].toarray().nonzero()
        this_preds = predecessors[neighbs]

        # need to keep ordering in this sublist to preserve hierarchy
        this_unique_preds = [p for p in peaks if ((p in this_preds) & (p >= 0))]
        joins_new_subgraph = not set(this_unique_preds).issubset(assessed_peaks)
        if tuple(this_unique_preds) in key_cols.keys():
            classification = "slope"
        elif len(this_unique_preds) == 0:
            classification = "peak"
        elif (len(this_unique_preds) >= 2) & joins_new_subgraph:
            classification = "keycol"
        else:
            classification = "slope"

        classifications[this_full_ix] = classification

        if (
            classification == "keycol"
        ):  # this_ix merges two or more subgraphs, so is a key_col
            # find the peaks it joins
            now_joined_peaks = this_unique_preds
            # add them as keys for the key_col lut
            key_cols.update({tuple(now_joined_peaks): this_full_ix})
            msg += f"\n{this_full_ix} is a key col between {now_joined_peaks}!"
            dominating_peak[this_full_ix] = now_joined_peaks[
                -1
            ]  # lowest now-joined peak
            predecessors[this_full_ix] = now_joined_peaks[-1]
            prominence[this_full_ix] = 0
            # given we now know the key col, get the prominence for
            # unassayed peaks in the subgraph
            for peak_ix in now_joined_peaks:
                if peak_ix in assessed_peaks:
                    continue
                # prominence is peak - key col
                keycols[peak_ix] = this_full_ix
                prominence[peak_ix] -= value
                assessed_peaks.update((peak_ix,))
        elif classification == "peak":  # this_ix is a new peak since it's disconnected
            msg += f"\n{this_full_ix} is a peak!"
            # its parent is the last visited peak (for precedence purposes)
            try:
                previous_peak = peaks[-1]
            except IndexError:
                previous_peak = this_full_ix
            if not (this_full_ix in peaks):
                peaks.append(this_full_ix)
            dominating_peak[this_full_ix] = previous_peak
            predecessors[this_full_ix] = this_full_ix
            # we initialize prominence here, rather than compute it solely in
            # the `key_col` branch because a graph `island` disconnected observation
            # should have prominence "value - 0", since it has no key cols
            prominence[this_full_ix] = X[this_full_ix]
        else:  # this_ix is connected to an existing peak, but doesn't bridge peaks.
            msg += f"\n{this_full_ix} is a slope!"
            # get all the peaks that are linked to this slope
            this_peak = this_unique_preds
            if len(this_peak) == 1:  # if there's only one peak the slope is related to
                # then use it
                best_peak = this_peak[0]
            else:  # otherwise, if there are multiple peaks
                # pick the one that most of its neighbors are assigned to
                best_peak = most_common_value(
                    this_unique_preds, keepdims=False
                ).mode.item()
            all_on_slope = numpy.arange(n)[dominating_peak == best_peak]
            msg += f"\n{all_on_slope} are on the slopes of {best_peak}."
            dominating_peak[this_full_ix] = best_peak
            predecessors[this_full_ix] = best_peak

        if verbose:
            print(
                (
                    "--------------------------------------------\n"
                    f"at the {rank} iteration:\n{msg}\n\tpeaks\t{peaks}\n"
                    f"\tprominence\t{prominence}\n\tkey_cols\t{key_cols}\n"
                )
            )
        if gdf is not None:
            peakframe = gdf.iloc[peaks]
            keycolframe = gdf.iloc[list(key_cols.values())]
            _isin_peak = gdf.index.isin(peakframe.index)
            _isin_keycol = gdf.index.isin(keycolframe.index)
            slopeframe = gdf[~(_isin_peak | _isin_keycol) & mask]
            rest = gdf[~mask]
            this_geom = gdf.iloc[[this_full_ix]]
            ax = rest.plot(edgecolor="k", lw=0.1, fc="lightblue")
            ax = slopeframe.plot(ec="k", lw=0.1, fc="linen", ax=ax)
            ax = keycolframe.plot(ec="k", lw=0.1, fc="red", ax=ax)
            ax = peakframe.plot(ec="k", lw=0.1, fc="yellow", ax=ax)
            ax = this_geom.centroid.plot(ax=ax, color="orange", marker="*")
            plt.show()
            command = input()
            if command.strip().lower() == "stop":
                break
    result = pandas.DataFrame.from_dict(
        dict(
            index=ids,
            prominence=prominence,
            classification=classifications,
            predecessor=predecessors,
            keycol=keycols,
            dominating_peak=dominating_peak,
        )
    )
    if not return_all:
        return result.prominence.values
    return result


def to_elevation(X, middle="mean", metric="euclidean"):
    """
    Compute the "elevation" of coordinates in p-dimensional space.

    For 1 dimensional X, this simply sets the zero point at the minimum value
    for the data. As an analogue to physical elevation, this means that the
    lowest value in 1-dimensional X is considered "sea level."

    For X in higher dimension, we treat X as defining a location on a (hyper)sphere.
    The "elevation," then, is the distance from the center of mass.
    So, this computes the distance of each point to the overall the center of mass
    and uses this as the "elevation," setting sea level (zero) to the lowest elevation.

    Parameters
    ----------
    X : numpy.ndarray
        Array of values for which to compute elevation.
    middle : callable or string
        name of function in numpy (or function itself)
        used to compute the center point of X
    metric : string
        metric to use in `scipy.spatial.distance.cdist` to
        compute the distance from the center of mass to the point.

    Returns
    --------
    (N,1)-shaped numpy array containing the "elevation" of
    each point relative to sea level (zero).

    """
    if X.ndim == 1:
        return X - X.min()
    else:
        if callable(middle):
            middle_point = middle(X, axis=0)
        else:
            try:
                middle = getattr(numpy, middle)
                return to_elevation(X, middle=middle)
            except AttributeError:
                raise KeyError(
                    f"numpy has no '{middle}' function to "
                    "compute the middle of a point cloud."
                )
        distance_from_center = distance.cdist(
            X, middle_point.reshape(1, -1), metric=metric
        )
        return to_elevation(distance_from_center.squeeze())


def _check_connectivity(connectivity_or_coordinates):
    """
    Check that connectivity provided is either:
    1. a sparse graph from scipy.sparse
    2. a weights object from libpysal that we need to cast to a scipy.sparse matrix
    3. a set of coordinates that we need to build the delaunay triangulation
       and then return the graph.
    """
    from scipy.sparse import issparse

    if issparse(connectivity_or_coordinates):
        shape = connectivity_or_coordinates.shape
        assert (
            shape[0] == shape[1]
        ), f"Connectivity matrix must be square, but is {shape}."
        return connectivity_or_coordinates
    if issubclass(type(connectivity_or_coordinates), weights.W):
        return connectivity_or_coordinates.sparse
    else:
        from libpysal.weights import Voronoi

        return _check_connectivity(Voronoi(connectivity_or_coordinates))


if __name__ == "__main__":
    import geopandas
    import matplotlib.pyplot as plt
    import pandas  # noqa F811
    from libpysal import examples, weights  # noqa F811
    from matplotlib import cm

    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color="lightgrey")

    data = (
        geopandas.read_file(examples.get_path("NAT.shp"))
        .query('STATE_NAME == "Illinois"')
        .reset_index()
    )
    coordinates = numpy.column_stack((data.centroid.x, data.centroid.y))
    gini = data[["GI89"]].values.flatten()
    contig_graph = weights.Rook.from_dataframe(data)
    iso = isolation(gini, coordinates, return_all=True)

    f, ax = plt.subplots(1, 3)
    for i in range(1, 3):
        data.plot(color="lightgrey", ax=ax[i])
    data.plot(iso.gap, ax=ax[1], cmap=current_cmap)
    data.plot(iso.distance, ax=ax[2], cmap=current_cmap)
    data.plot("GI89", ax=ax[0])
    ax[0].set_title("Variable")
    ax[1].set_title("Releif")
    ax[2].set_title("Isolation")
    plt.show()

    prom = prominence(gini, contig_graph)
    ax = data.plot(prom)
    data.plot(color="lightgrey", ax=ax, zorder=-1)
    plt.show()
