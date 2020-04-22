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
    width, height = shape[2] - shape[0], shape[3] - shape[1]
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
    or let it pass through if it's 1 dimensional & length 4
    """
    if (shape.ndim == 1) & (len(shape) == 4):
        return shape
    return numpy.array([*shape.min(axis=0), *shape.max(axis=0)])


@_bbox.register
def _(shape: spatial.ConvexHull):
    """
    For scipy.spatial.ConvexHulls, compute the bounding box from
    their boundary points.
    """
    return _bbox(shape.points[shape.vertices])


@singledispatch
def _contains(shape, x, y):
    """
    Try to use the shape's contains method directly on XY.
    Does not currently work on anything. 
    """
    return shape.contains((x, y))


@_contains.register
def _(shape: numpy.ndarray, x: float, y: float):
    """
    If provided an ndarray, assume it's a bbox
    and return whether the point falls inside
    """
    xmin, xmax = shape[0], shape[2]
    ymin, ymax = shape[1], shape[3]
    in_x = (xmin <= x) and (x <= xmax)
    in_y = (ymin <= y) and (y <= ymax)
    return in_x & in_y


@_contains.register
def _(shape: spatial.Delaunay, x: float, y: float):
    """
    For points and a delaunay triangulation, use the find_simplex
    method to identify whether a point is inside the triangulation.

    If the returned simplex index is -1, then the point is not
    within a simplex of the triangulation. 
    """
    return delaunay.find_simplex((x, y)) > 0


@_contains.register
def _(shape: spatial.ConvexHull, x: float, y: float):
    """
    For convex hulls, convert their exterior first into a Delaunay triangulation
    and then use the delaunay dispatcher.
    """
    exterior = shape.points[shape.vertices]
    delaunay = spatial.Delaunay(exterior)
    return _contains(x, y, delaunay)


try:
    import shapely

    HAS_SHAPELY = True

    @_contains.register
    def _(shape: shapely.geometry.Polygon, x: float, y: float):
        """
        If we know we're working with a shapely polygon, 
        then use the contains method & cast input coords to a shapely point
        """
        return shape.contains(shapely.geometry.Point((x, y)))


except ModuleNotFoundError:
    HAS_SHAPELY = False


try:
    import pygeos

    HAS_PYGEOS = True

    @_area.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.area
        """
        return pygeos.area(shape)

    @_contains.register
    def _(shape: pygeos.Geometry, x: float, y: float):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.within casting the points to a pygeos object too
        """
        return pygeos.within(pygeos.points((x, y)), shape)

    @_bbox.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.bounds
        """
        return pygeos.bounds(shape)


except ModuleNotFoundError:
    HAS_PYGEOS = False


def _build_best_tree(coordinates, metric):
    """
    Build the best query tree that can support the application.
    Chooses from:
    1. sklearn.KDTree if available and metric is simple
    2. sklearn.BallTree if available and metric is complicated
    3. scipy.spatial.cKDTree if nothing else
    """
    tree = spatial.cKDTree
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
        - a bounding box array of [xmin, ymin, xmax, ymax]
        - a scipy.spatial.ConvexHull object from the Qhull library
        - a shapely shape using alpha_shape_auto
    """
    if (hull is None) or (hull == "bbox"):
        return numpy.array([*coordinates.min(axis=0), *coordinates.max(axis=0)])
    if isinstance(hull, numpy.ndarray):
        assert len(hull) == 4, f"bounding box provided is not shaped correctly! {hull}"
        assert hull.ndim == 1, f"bounding box provided is not shaped correctly! {hull}"
        return hull
    if HAS_SHAPELY:  # protect the isinstance check if import has failed
        if isinstance(hull, shapely.geometry.Polygon):
            return hull
    if HAS_PYGEOS:
        if isinstance(hull, pygeos.Geometry):
            return hull
    if isinstance(hull, str):
        if hull.startswith("convex"):
            return spatial.ConvexHull(coordinates)
        elif hull.startswith("alpha") or hull.startswith("α"):
            return alpha_shape_auto(coordinates)

    raise ValueError(
        f"Hull type {hull} not in the set of valid options:"
        f" (None, 'bbox', 'convex', 'alpha', 'α', "
        f" shapely.geometry.Polygon, pygeos.Geometry)"
    )


def _prepare(coordinates, support, distances, metric, hull, edge_correction):
    """
    prepare the arguments to convert into a standard format
    1. cast the coordinates to a numpy array
    2. precomputed metrics must have distances provided
    3. metrics must be callable or string
    4. warn if distances are specified and metric is not default
    5. make distances a numpy.ndarray
    6. construct the support, accepting:
        - num_steps -> a linspace with len(support) == num_steps
                       from zero to a quarter of the bounding box's smallest side
        - (stop, ) -> a linspace with len(support) == 20
                 from zero to stop
        - (start, stop) -> a linspace with len(support) == 20
                           from start to stop
        - (start, stop, num_steps) -> a linspace with len(support) == num_steps
                                      from start to stop
        - numpy.ndarray -> passed through
    """
    # Throw early if edge correction is requested
    if edge_correction is not None:
        raise NotImplementedError("Edge correction is not currently implemented.")

    # cast to coordinate array
    coordinates = numpy.asarray(coordinates)
    hull = _prepare_hull(coordinates, hull)

    # evaluate distances
    if (distances is None) and metric == "precomputed":
        raise ValueError(
            "If metric =`precomputed` then distances must"
            " be provided as a (n,n) numpy array."
        )
    if not (isinstance(metric, str) or callable(metric)):
        raise TypeError(
            f"`metric` argument must be callable or a string. Recieved: {metric}"
        )
    if distances is not None and metric != "euclidean":
        warnings.Warn(
            "Distances were provided. The specified metric will be ignored."
            " To use precomputed distances with a custom distance metric,"
            " do not specify a `metric` argument."
        )
        metric = "euclidean"

    if support is None:
        support = 20

    if isinstance(support, int):
        bbox = _bbox(coordinates)
        height, width = bbox[3] - bbox[1], bbox[2] - bbox[0]
        ripley_rule = 0.25 * numpy.minimum(height, width)
        support = numpy.linspace(0, ripley_rule, num=support)
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
    """
    Simulate from the given hull with a specified intensity or size.
    
    Hulls can be:
    - bounding boxes (numpy.ndarray with dim==1 and len == 4)
    - scipy.spatial.ConvexHull
    - shapely.geometry.Polygon
    - pygeos.Geometry

    If intensity is specified, size must be an integer reflecting
    the number of realizations. 
    If the size is specified as a tuple, then the intensity is 
    determined by the area of the hull. 
    """
    if size is None:
        if intensity is not None:
            # if intensity is provided, assume
            # n_observations
            n_observations = int(_area(hull) * intensity)
        else:
            # default to 100 points
            n_observations = 100
        n_replications = 1
        size = (n_observations, n_replications)

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
                f" containing (number of observations, number of replications)"
                f" with no specified intensity, or an intensity and size equal"
                f" to the number of replications."
                f" Recieved: `intensity={intensity}, size={size}`"
            )

    elif intensity is not None and isinstance(size, int):
        # assume int size with specified intensity means n_replications at x intensity
        n_observations = intensity * _area(hull)
        n_replications = size
    else:
        raise ValueError(
            f"Intensity and size not understood. Provide size as a tuple"
            f" containing (number of observations, number of replications)"
            f" with no specified intensity, or an intensity and size equal"
            f" to the number of replications."
            f" Recieved: `intensity={intensity}, size={size}`"
        )
    result = numpy.empty((n_replications, n_observations, 2))

    bbox = _bbox(hull)

    for i_replication in range(n_replications):
        generating = True
        i_observation = 0
        while i_observation < n_observations:
            x, y = (
                numpy.random.uniform(bbox[0], bbox[2]),
                numpy.random.uniform(bbox[1], bbox[3]),
            )
            if _contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
            i_observation += 1
    return result.squeeze()


def simulate_from(coordinates, hull=None, size=None):
    """
    Simulate a pattern from the coordinates provided using a given assumption
    about the hull of the process. 

    Note: will always assume the implicit intensity of the process. 
    """
    n_observations = coordinates.shape[0]
    hull = _prepare_hull(coordinates, hull)
    return simulate(hull, intensity=None, size=(n_observations, size))


### Ripley's functions


def f_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """
    if isinstance(coordinates, TREE_TYPES):
        tree = coordinates
        coordinates = tree.data
    coordinates, support, distances, metric, hull, _ = _prepare(
        coordinates, support, distances, metric, hull, edge_correction
    )
    if distances is not None:
        n_observations = coordinates.shape[0]
        if distances.ndim == 2:
            k, p = distances.shape
            if k == p == n:
                warnings.Warn(
                    f"A full distance matrix is not required for this function, and"
                    f" the intput matrix is a square {n},{n} matrix. Only the"
                    f" distances from p random points to their nearest neighbor within"
                    f" the pattern is required, as an {n},p matrix. Assuming the"
                    f" provided distance matrix has rows pertaining to input"
                    f" pattern and columns pertaining to the output points."
                )
                distances = distances.min(axis=0)
            elif k == n:
                distances = distances.min(axis=0)
            else:
                raise ValueError(
                    f"Distance matrix should have the same rows as the input"
                    f" coordinates with p columns, where n may be equal to p."
                    f" Recieved an {k},{p} distance matrix for {n} coordinates"
                )
        elif distances.ndim == 1:
            p = len(distances)
    else:
        # if we only have a few, do 1000 empties.
        # Otherwise, grow empties slowly
        n = coordinates.shape[0]
        order_of_magnitude = numpy.log10(n).astype(int)
        if order_of_magnitude < 4:
            n_empty_points = 1000
        else:
            n_empty_points = 10 ** (int(order_of_magnitude ** 0.5) + 1)

        randoms = simulate(hull=hull, size=(n_empty_points, 1))
        try:
            tree
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, _ = tree.query(randoms, k=1)

    counts, bins = numpy.histogram(distances, bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()

    return bins, numpy.asarray([*fracs, 1])


def g_function(
    coordinates, support=None, distances=None, metric="euclidean", edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """

    if isinstance(coordinates, (spatial.KDTree, spatial.cKDTree)):
        tree = coordinates
        coordinates = tree.data
    coordinates, support, distances, metric, *_ = _prepare(
        coordinates, support, distances, metric, None, edge_correction
    )
    if distances is not None:
        if distances.ndim == 2:
            if distances.shape[0] == distances.shape[1] == coordinates.shape[0]:
                warnings.Warn(
                    "The full distance matrix is not required for this function,"
                    " only the distance to the nearest neighbor within the pattern."
                    " Computing this and discarding the rest."
                )
                distances = distances.min(axis=1)
            else:
                k, p = distances.shape
                n = coordinates.shape[0]
                raise ValueError(
                    " Input distance matrix has an invalid shape: {k},{p}."
                    " Distances supplied can either be 2 dimensional"
                    " square matrices with the same number of rows"
                    " as `coordinates` ({n}) or 1 dimensional and contain"
                    " the shortest distance from each point in "
                    " `coordinates` to some other point in coordinates."
                )
        elif distances.ndim == 1:
            if distance.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    f"Distances are not aligned with coordinates! Distance"
                    f" matrix must be (n_coordinates, n_coordinates), but recieved"
                    f" {distance.shape} instead of ({coordinates.shape[0]},)"
                )
        else:
            raise ValueError(
                "Distances supplied can either be 2 dimensional"
                " square matrices with the same number of rows"
                " as `coordinates` or 1 dimensional and contain"
                " the shortest distance from each point in "
                " `coordinates` to some other point in coordinates."
                " Input matrix was {distances.ndim} dimensioanl"
            )
    else:
        try:
            tree
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, indices = tree.query(coordinates, k=2)

    counts, bins = numpy.histogram(distances, bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()

    return bins, numpy.asarray([*fracs, 1])


def j_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: tuple of numpy.ndarray
        precomputed distances to use to evaluate the j function. 
        The first must be of shape (n,n) or (n,) and is used in the g function.
        the second must be of shape (n,p) or (p,) (with p possibly equal to n)
        used in the f function.
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern for the f function.
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """
    if distances is not None:
        g_distances, f_distances = distances
    else:
        g_distances = f_distances = None
    gsupport, gstats = g_function(
        coordinates,
        support=support,
        distances=g_distances,
        metric=metric,
        edge_correction=edge_correction,
    )
    fsupport, fstats = f_function(
        coordinates,
        support=support,
        distances=f_distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
    )
    if not numpy.allclose(gsupport, fsupport):
        ffunction = interpolate.interp1d(fsupport, fstats)
        fstats = ffunction(gsupport)
    both_zero = (gstats == 1) & (fstats == 1)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        hazard_ratio = (1 - gstats) / (1 - fstats)

    hazard_ratio[both_zero] = 1

    return gsupport, hazard_ratio


def k_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """
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
                    f" matrix matching the number of input points. The shape of the input matrix"
                    f" is {distance.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
                )
            upper_tri_distances = distances
        elif distances.shape[0] == distances.shape[1] == n:
            upper_tri_distances = distances[numpy.triu_indices_from(distances, k=1)]
        else:
            raise ValueError(
                f"Shape of inputted distances is not square, nor is the upper triangular"
                f" matrix matching the number of input points. The shape of the input matrix"
                f" is {distance.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
            )
    else:
        upper_tri_distances = spatial.distance.pdist(coordinates, metric=metric)
    n_pairs_less_than_d = (upper_tri_distances < support.reshape(-1, 1)).sum(axis=1)
    intensity = n / _area(hull)
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    return support, k_estimate


def l_function(
    coordinates,
    support=None,
    permutations=9999,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    linearized=False,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    linearized : bool
        whether or not to subtract the expected value from l at each 
        distance bin. This centers the l function on zero for all distances.
        Proposed by Besag (1977) #TODO: fix besag ref
    """

    support, k_estimate = k_function(
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
    )

    return support, numpy.sqrt(k_estimate / numpy.pi) - linearized * support


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
    **kwargs,
):
    stat_function, result_container = dispatch.get(calltype)
    core_kwargs = dict(
        support=None, metric="euclidean", hull=None, edge_correction=None,
    )
    tree = _build_best_tree(coordinates, metric=metric)  # amortize this
    hull = _prepare_hull(tree.data, hull)  # and this over replications
    core_kwargs["hull"] = hull

    if calltype in ("F", "J"):
        random = simulate_from(coordinates)
        distances, _ = tree.query(random)
        random_tree = _build_best_tree(random)

    observed_support, observed_statistic = stat_function(
        tree, distances=distances, **core_kwargs
    )
    core_kwargs["support"] = observed_support

    if keep_replications:
        replications = numpy.empty((len(observed_support), n_replications))
    for i_replication in range(n_replications):
        random_i = simulate_from(tree.coordinates)
        if calltype in ("F", "J"):
            distances, _ = random_tree(random_i)
            core_kwargs["distance"] = distances
        replications_i = stat_function(random_i, **core_kwargs)[1]
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
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_replications: bool
        whether or not to keep the simulation envelopes. If so, 
        will be returned as the result's reference_distribution attribute
    n_replications: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness. 
    """

    return _ripley_test(
        "F",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_replications=keep_replications,
        n_replications=n_replications,
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
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_replications: bool
        whether or not to keep the simulation envelopes. If so, 
        will be returned as the result's reference_distribution attribute
    n_replications: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness. 
    """
    return _ripley_test(
        "G",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_replications=keep_replications,
        n_replications=n_replications,
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
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_replications: bool
        whether or not to keep the simulation envelopes. If so, 
        will be returned as the result's reference_distribution attribute
    n_replications: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness. 
    """
    return _ripley_test(
        "J",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_replications=keep_replications,
        n_replications=n_replications,
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
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_replications: bool
        whether or not to keep the simulation envelopes. If so, 
        will be returned as the result's reference_distribution attribute
    n_replications: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness. 
    """
    return _ripley_test(
        "K",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_replications=keep_replications,
        n_replications=n_replications,
    )


def l_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    linearized=False,
    n_replications=9999,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_replications: bool
        whether or not to keep the simulation envelopes. If so, 
        will be returned as the result's reference_distribution attribute
    n_replications: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness. 
    """
    return _ripley_test(
        "L",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_replications=keep_replications,
        n_replications=n_replications,
    )
