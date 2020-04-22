import numpy
import warnings
from scipy import spatial, interpolate
from functools import singledispatch
from collections import namedtuple
from libpysal.cg import alpha_shape_auto
from libpysal.cg.kdtree import Arc_KDTree

### Utilities and dispatching

TREE_TYPES = (spatial.KDTree, spatial.cKDTree, Arc_KDTree)
try:
    from sklearn.neighbors import KDTree, BallTree

    TREE_TYPES = (*TREE_TYPES, KDTree, BallTree)
except ModuleNotFoundError:
    pass

## Define default dispatches and special dispatches without GEOS
@singledispatch
def _area(shape):
    """
    If a shape has an area attribute, return it. 
    Works for: 
        shapely.geometry.Polygon
        scipy.spatial.ConvexHull
    """
    return shape.area


@_area.register
def _(shape: numpy.ndarray):
    """
    If a shape describes a bounding box, compute length times width
    """
    assert len(shape) == 4, "shape is not a bounding box!"
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return numpy.abs(width * height)


@singledispatch
def _bbox(shape):
    """
    If a shape has bounds, use those.
    Works for:
        shapely.geometry.Polygon
    """
    return shape.bounds


@_bbox.register
def _(shape: numpy.ndarray):
    """
    If a shape is an array of points, compute the minima/maxima
    """
    return numpy.array([*shape.min(axis=0), *shape.max(axis=0)])


@_bbox.register
def _(shape: spatial.ConvexHull):
    """
    For scipy.spatial.ConvexHulls, compute the bounding box from
    their boundary points.
    """
    return _bbox(shape.points[shape.vertices])


@singledispatch
def _within(x: float, y: float, shape: spatial.Delaunay):
    """
    For points and a delaunay triangulation, use the find_simplex
    method to identify whether a point is inside the triangulation.

    If the returned simplex index is -1, then the point is not
    within a simplex of the triangulation. 
    """
    return delaunay.find_simplex((x, y)) > 0


@_within.register
def _(x: float, y: float, shape: spatial.ConvexHull):
    """
    For convex hulls, convert them first. 
    """
    exterior = hull.points[hull.vertices]
    delaunay = spatial.Delaunay(exterior)
    return _within(x, y, delaunay)


try:
    import shapely

    HAS_SHAPELY = True
except ModuleNotFoundError:
    HAS_SHAPELY = False

    @_.register
    def _within(x: float, y: float, shape: shapely.geometry.Polygon):
        """
        If we know we're working with a shapely polygon, 
        then use the contains method on the input shape. 
        """
        return shape.contains(shapely.geometry.Point((x, y)))


try:
    import pygeos

    HAS_PYGEOS = True
except ModuleNotFoundError:
    HAS_PYGEOS = False

    @_area.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.area
        """
        return pygeos.area(shape)

    @_within.register
    def _(x: float, y: float, shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.within
        """
        return pygeos.within(pygeos.points((x, y)), shape)

    @_bbox.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.bounds
        """
        return pygeos.bounds(shape)


def _build_best_tree(coordinates, metric):
    tree = scipy.spatial.cKDTree
    try:
        from sklearn.neighbors import KDTree, BallTree

        if metric in KDTree.valid_metrics:
            tree = lambda coordinates: KDTree(coordinates, metric=metric)
        elif metric in BallTree.valid_metrics:
            tree = lambda coordinates: BallTree(coordinates, metric=metric)
        elif callable(metric):
            warnings.Warn(
                "Distance metrics defined in pure Python may "
                " have unacceptable performance!"
            )
            tree = lambda coordinates: BallTree(coordinates, metric=metric)
    except ModuleNotFoundError:
        pass
    return tree(coordinates)


def _prepare_hull(coordinates, hull):
    """
    Construct a hull from the coordinates given a hull type
    Will either return:
        - a bounding box of [xmin, ymin, xmax, ymax]
        - a scipy.spatial.ConvexHull object from the Qhull library
        - a shapely shape using alpha_shape_auto
    """
    fail = True
    if (hull is None) or (hull == "bbox"):
        hull = numpy.asarray([*coordinates.min(axis=0), *coordinates.max(axis=0)])
        fail = False
    elif hull.startswith("convex"):
        hull = spatial.ConvexHull(coordinates)
        fail = False
    elif hull.startswith("alpha") or hull.startswith("α"):
        hull = alpha_shape_auto(coordinates)
        fail = False
    elif HAS_SHAPELY:  # protect the isinstance check if import has failed
        if isinstance(hull, shapely.geometry.Polygon):
            hull = hull
            fail = False
    elif HAS_PYGEOS:
        if isinstance(hull, pygeos.Geometry):
            hull = hull
            fail = False
    if fail:
        raise ValueError(
            "Hull type {hull} not in the set of valid options:"
            '{None, "bbox", "convex", "alpha", "α", '
            " shapely.geometry.Polygon, pygeos.Geometry}"
        )
    return hull


def _prepare(coordinates, support, distances, metric, hull, edge_correction):

    # Throw early if edge correction is requested
    if edge_correction is not None:
        raise NotImplementedError("Edge correction is not currently implemented.")

    # cast to coordinate array
    coordinates = numpy.asarray(coordinates)
    hull = _prepare_hull(hull)

    # evaluate distances
    if (distances is None) and metric == "precomputed":
        raise ValueError(
            "If metric =`precomputed` then distances must"
            " be provided as a (n,n) numpy array."
        )
    elif not (isinstance(metric, str) or callable(metric)):
        raise TypeError(
            f"`metric` argument must be callable or a string. Recieved: {metric}"
        )
    elif distances is not None and metric != "euclidean":
        warnings.Warn(
            "Distances were provided. The specified metric will be ignored."
            " To use precomputed distances with a custom distance metric,"
            " do not specify a `metric` argument."
        )
    elif isinstance(distances, numpy.ndarray):
        assert (
            distances.shape[0] == distances.shape[1] == coordinates.shape[0]
        ), "Distances are not (n,n), aligned with coordinate matrix"

    n = int(coordinates.shape[0])
    upper_tri_n = int(n * (n - 1) * 0.5)

    if support is None:
        support = 20

    if isinstance(support, int):
        support = numpy.linspace(0, distances.max(), num=support)
    elif isinstance(support, tuple):
        if len(support) == 1:
            support = numpy.linspace(0, support[0], num=20)  # default support n bins
        elif len(support) == 2:
            support = numpy.linspace(*support, num=20)  # default support n bins
        elif len(support == 3):
            support = numpy.linspace(support[0], support[1], num=support[2])
    else:
        try:
            support = numpy.asarray(support)
        except:
            raise TypeError(
                "`support` must be a tuple (either (start, stop, step), (start, stop) or (stop,)),"
                " an int describing the number of breaks to use to evalute the function,"
                " or an iterable containing the breaks to use to evaluate the function."
                " Recieved object of type {}: {}".format(type(support), support)
            )

    return coordinates, support, distances, metric, hull, edge_correction


### simulators


def simulate(hull, intensity=None, size=None):
    if (intensity is None) and (size is None):
        intensity = 100 / _area(hull)  # default to intensity at 100 points per area
        size = 1  # default to one replication

    if isinstance(size, tuple):
        if len(size) == 2 and intensity is None:
            n_observations, n_replications = size
            intensity = n_observations / _area(hull)
        elif len(size) == 2 and intensity is not None:
            raise ValueError(
                "Either intensity or size as (n observations, n replications)"
                " can be provided. Providing both creates statistical conflicts."
                " between the requested intensity and implied intensity by"
                " the number of observations and the area of the hull. If"
                " you want to specify the intensity, use the intensity argument"
                " and set size equal to the number of replications."
            )
        else:
            raise ValueError(
                f"Intensity and size not understood. Provide size as a tuple"
                " containing (number of observations, number of replications)"
                " with no specified intensity, or an intensity and size equal"
                " to the number of replications."
                " Recieved: `intensity={intensity}, size={size}`"
            )

    elif intensity is not None and isinstance(size, int):  # catches default, too!
        n_observations = intensity * _area(hull)
        n_replications = size
    else:
        raise ValueError(
            f"Intensity and size not understood. Provide size as a tuple"
            " containing (number of observations, number of replications)"
            " with no specified intensity, or an intensity and size equal"
            " to the number of replications."
            " Recieved: `intensity={intensity}, size={size}`"
        )
    result = numpy.empty((n_replications, n_observations, 2))

    bbox = _bbox(hull)

    for i_replication in range(n_replications):
        generating = True
        i_observation = 0
        while i_observation < n_observations:
            x, y = (
                numpy.random.random(bbox[0], bbox[2]),
                numpy.random.random(bbox[1], bbox[3]),
            )
            if _within(x, y, hull):
                result[i_observation, i_replication] = (x, y)
            i_observation += 1
    return result


def simulate_from(coordinates, hull=None, size=None):
    """
    Simulate a pattern from the coordinates provided using a given assumption
    about the hull of the process. 

    Note: will always assume the implicit intensity of the process. 
    """
    n_observations = vertices.shape[0]
    hull = _prepare_hull(coordinates, hull)
    return simulate(hull, intensity=None, size=(n_observations, size))


### Ripley's functions


def f_function(
    coordinates, support=None, distances=None, metric="euclidean", hull=None
):
    if isinstance(coordinates, TREE_TYPES):
        tree = coordinates
        coordinates = tree.data
    coordinates, support, distances, metric, hull, _ = _prepare(
        coordinates, support, distances, metric, hull, None
    )
    if distances is not None:
        if distances.shape[0] == distances.shape[1] == coordinates.shape[0]:
            warnings.Warn(
                "The full distance matrix is not required for this function,"
                " only the distance to the nearest neighbor within the pattern."
                " Computing this, assuming provided distance matrix has rows"
                " pertaining to the coordinates and columns pertaining to the"
                " empty space sample points."
            )
            distances = distances.min(axis=1)
        elif distance.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "Distances are not aligned with coordinates! Distance"
                " matrix must be (n_coordinates, p_random_points), but recieved"
                " {distance.shape}"
            )
    else:
        # if we only have a few, do 1000 empties.
        # Otherwise, grow empties logarythmically
        n_empty_points = numpy.minimum(
            numpy.log10(coordinates.shape[0]).astype(int), 1000
        )
        intensity = n_empty_points / _area(hull)
        randoms = simulate(hull, intensity=intensity)
        try:
            distances, _ = tree.query(randoms, k=1)
        except NameError:
            if metric != "euclidean":
                raise ValueError(
                    "KDTree can only use euclidean distances."
                    " If you would like to use alternative distance"
                    " metrics, compute them using scipy.spatial.distance.cdist"
                    " "
                )
            tree = _build_best_tree(coordinates, metric)
            distances, _ = tree.query(randoms, k=1)

    counts, bins = numpy.histogram(distances, bins=support)
    return bins, numpy.cumsum(counts) / counts.sum()


def g_function(coordinates, support=None, distances=None, metric="euclidean"):
    if isinstance(coordinates, (spatial.distance.KDTree, spatial.distance.cKDTree)):
        tree = coordinates
        coordinates = tree.data
    coordinates, support, distances, metric = _prepare(
        coordinates, support, distances, metric, None, None
    )
    if distances is not None:
        if distances.shape[0] == distances.shape[1] == coordinates.shape[0]:
            warnings.Warn(
                "The full distance matrix is not required for this function,"
                " only the distance to the nearest neighbor within the pattern."
                " Computing this and discarding the rest."
            )
            distances = distances.min(axis=1)
        elif distance.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "Distances are not aligned with coordinates! Distance"
                " matrix must be (n_coordinates, n_coordinates), but recieved"
                " {distance.shape}"
            )
    else:
        try:
            distances, indices = tree.query(coordinates, k=2)
        except NameError:
            tree = _build_best_tree(coordinates, metric)
            distances, indices = tree.query(coordinates, k=2)
        finally:
            # in case the tree returns self-neighbors in slot 2
            # when there are coincident points
            arange = numpy.arange(distances.shape[0])
            distances = distances[arange, indices[:, 1] != arange]

    counts, bins = numpy.histogram(distances, bins=support)
    return bins, numpy.cumsum(counts) / counts.sum()


def j_function(
    coordinates, support=None, distances=None, metric="euclidean", hull=None
):
    gsupport, gstats = g_function(
        coordinates, support=support, distances=distances, metric=metric
    )
    fsupport, fstats = f_function(
        coordinates, support=support, distances=distances, metric=metric, hull=hull
    )
    if not numpy.allclose(gsupport, fsupport):
        ffunction = interpolate.interp1d(fsupport, fstats)
        fstats = ffunction(gsupport)
    return gsupport, (1 - gstats) / (1 - fstats)


def k_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    coordinates, support, distances, metric, hull, edge_correction = _prepare(
        coordinates, support, distances, metric, hull, edge_correction
    )
    n = coordinates.shape[0]
    upper_tri_n = n * (n - 1) * 0.5
    if distances is not None:
        if distances.ndim == 1:
            if distances.shape[0] != upper_tri_n:
                raise ValueError(
                    f"Shape of inputted distances is not square, nor is the upper triangular"
                    " matrix matching the number of input points. The shape of the input matrix"
                    " is {distance.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
                )
            upper_tri_distances = distances
        elif distances.shape[0] == distances.shape[1] == n:
            upper_tri_distances = distances[numpy.triu_indices_from(distances, k=1)]
        else:
            raise ValueError(
                f"Shape of inputted distances is not square, nor is the upper triangular"
                " matrix matching the number of input points. The shape of the input matrix"
                " is {distance.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
            )
    n_pairs_less_than_d = (upper_tri_distances < support.reshape(-1, 1)).sum(axis=1)
    intensity = n / _area(hull)
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    return support, kestimate


def l_function(
    coordinates,
    support=None,
    permutations=9999,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    # This is not equivalent to the estimator in pointpats, but that
    # is likely flawed according to https://github.com/pysal/pointpats/issues/44
    support, kestimate = k_function(
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
    )
    return support, numpy.sqrt(kestimate / numpy.pi)


### Ripley tests

FtestResult = namedtuple(
    "FtestResult", ("support", "statistic", "pvalue", "reference_distribution")
)
GtestResult = namedtuple(
    "GtestResult", ("support", "statistic", "pvalue", "reference_distribution")
)
JtestResult = namedtuple(
    "JtestResult", ("support", "statistic", "pvalue", "reference_distribution")
)
KtestResult = namedtuple(
    "KtestResult", ("support", "statistic", "pvalue", "reference_distribution")
)
LtestResult = namedtuple(
    "LtestResult", ("support", "statistic", "pvalue", "reference_distribution")
)

_ripley_dispatch = {
    "F": (f_function, FtestResult),
    "G": (g_function, GtestResult),
    "J": (j_function, JtestResult),
    "K": (k_function, KtestResult),
    "L": (l_function, LtestResult),
}


def _ripley_test(
    calltype,
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_replications=False,
    n_replications=9999,
):
    stat_function, result_container = dispatch.get(calltype)
    core_kwargs = dict(
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
    )
    tree = _build_best_tree(coordinates, metric=metric)  # amortize this
    hull = _prepare_hull(tree.data)  # and this over replications
    core_kwargs["hull"] = hull

    observed_support, observed_statistic = stat_function(tree, **core_kwargs)
    core_kwargs["support"] = observed_support

    if keep_replications:
        replications = numpy.empty((len(observed_support), n_replications))
    for i_replication in range(n_replications):
        replications_i = stat_function(tree, **core_kwargs)[1]
        if keep_replications:
            replications[i] = replications_i
    return result_container(
        observed_support, statistic, pvalue, replications if keep_replications else None
    )


def f_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_replications=False,
    n_replications=9999,
):

    return _ripley_test(
        "F",
        coordinates,
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
        keep_replications=False,
        n_replications=9999,
    )


def g_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_replications=False,
    n_replications=9999,
):
    return _ripley_test(
        "G",
        coordinates,
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
        keep_replications=False,
        n_replications=9999,
    )


def j_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_replications=False,
    n_replications=9999,
):
    return _ripley_test(
        "J",
        coordinates,
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
        keep_replications=False,
        n_replications=9999,
    )


def k_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    n_replications=9999,
):
    return _ripley_test(
        "K",
        coordinates,
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
        keep_replications=False,
        n_replications=9999,
    )


def l_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    n_replications=9999,
):
    return _ripley_test(
        "L",
        coordinates,
        support=None,
        distances=None,
        metric="euclidean",
        hull=None,
        edge_correction=None,
        keep_replications=False,
        n_replications=9999,
    )
