import contextlib
import warnings

import geopandas
import numpy
import pandas
import shapely
from packaging.version import Version

from .crand import njit, prange

__author__ = (
    "Martin Fleischmann <martin@fleischmann.net>",
    "Levi John Wolf <levi.john.wolf@gmail.com>",
    "Alan Murray <amurray@ucsb.edu>",
    "Jiwan Baik <jiwon.baik@geog.ucsb.edu>",
)


# -------------------- UTILITIES --------------------#
# shapely.orient_polygons not available in shapely<2.1. Check version, and if
# not available, use shapely.normalize (to make exterior rings clockwise) +
# shapely.reverse. This is about 30% slower than shapely.orient_polygons.
# Remove and use shapely.orient_polygons when shapely 2.1+ is required.

if Version(shapely.__version__) >= Version("2.1.0"):
    orient_polygons = shapely.orient_polygons
else:

    def orient_polygons(geometry, exterior_cw=False):
        g = shapely.normalize(geometry)
        if not exterior_cw:
            g = shapely.reverse(g)
        return g


def _cast(collection):
    """
    Cast a collection to a shapely geometry array.
    """

    if isinstance(collection, geopandas.GeoSeries | geopandas.GeoDataFrame):
        return numpy.asarray(collection.geometry.array)
    elif isinstance(collection, numpy.ndarray | list):
        return numpy.asarray(collection)
    else:
        return numpy.array([collection])


def _cast_pts_as_array(x):
    """
    Accepts:
      - array-like of shape (2,) or (n, 2) of real-valued numerics
      - shapely.geometry.Point or iterable of Points
      - geopandas.GeoSeries of Points

    Returns:
      - ndarray of shape (2,) or (n, 2), dtype float

    Elements must be real-valued numerics (no complex).
    Geometry inputs must be Points only; non-Point geometries are rejected.
    """

    # Handle GeoSeries
    if isinstance(x, geopandas.GeoSeries):
        if not all(x.geom_type == "Point"):
            raise TypeError("All geometries in GeoSeries must be Points")
        coords = shapely.get_coordinates(x)
        if coords.shape == (1, 2):
            coords = coords[0]
        return coords

    # Handle Single Point
    if isinstance(x, shapely.Point):
        return numpy.array([x.x, x.y], dtype=float)

    # Handle Iterable of Points
    if hasattr(x, "__iter__") and all(isinstance(p, shapely.Point) for p in x):
        coords = shapely.get_coordinates(x)
        return coords

    # Handle Array-Like
    try:
        arr = numpy.asarray(x, dtype=float)
    except Exception as e:
        raise TypeError("Input must be array-like") from e

    # Shape validation
    if arr.ndim == 1:
        if arr.shape != (2,):
            raise ValueError(f"Expected shape (2,), got {arr.shape}")
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"Expected shape (n, 2), got {arr.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D input, got {arr.ndim}D")

    # Type validation: real numbers only
    if not numpy.issubdtype(arr.dtype, numpy.floating) and not numpy.issubdtype(
        arr.dtype, numpy.integer
    ):
        raise TypeError("Elements must be floats or ints")

    return arr


def get_angles(collection, return_indices=False):
    """
    Get the angles pertaining to each vertex of a set of polygons.
    This assumes the input are polygons.

    Parameters
    ----------
    ga  :   shapely geometry array
        array of polygons/multipolygons
    return_indices  :   bool (Default: False)
        whether to return the indices relating each geometry to a polygon

    Returns
    -------
    angles between triples of points on each geometry, as well as the indices
    relating angles to input geometries (if requested).

    See the Notes for information on the shape of angles and indices.

    Notes
    -------
    If a geometry has n coordinates and k parts, the array will be n - k.
    If each geometry has n_i coordinates, then let N be a vector storing
    those counts (computed, for example, using shapely.get_num_coordinates(ga)).
    Likewise, let K be a vector storing the number of parts each geometry has, k_i
    (computed, for example, using shapely.get_num_geometries(ga))

    Then, the output is of shape (N - K).sum()

    """
    ga = _cast(collection)
    exploded = shapely.get_parts(ga)
    coords = shapely.get_coordinates(exploded)
    n_coords_per_geom = shapely.get_num_coordinates(exploded)
    angles = numpy.asarray(_get_angles(coords, n_coords_per_geom))
    if return_indices:
        return angles, numpy.repeat(
            numpy.arange(len(ga)),
            shapely.get_num_coordinates(ga) - shapely.get_num_geometries(ga),
        )
    else:
        return angles


@njit
def _get_angles(points, n_coords_per_geom):
    """
    Iterate over points in a set of geometries.
    This assumes that the input geometries are simple, not multi!
    """
    # Start at the first point of the first geometry
    offset = 0
    on_geom = 0
    on_coord = 0
    result = []
    while True:
        # if we're on the last point before the closure point,
        if on_coord == (n_coords_per_geom[on_geom] - 1):
            # set the offset to start on the first point of the next geometry
            offset += on_coord + 1
            on_geom += 1
            on_coord = 0
            # if we're now done with all geometries, exit
            if on_geom == len(n_coords_per_geom):
                break
            else:
                # and continue to the next iteration.
                continue
        # construct the triple so that we wrap around and avoid the closure point
        left_ix = offset + on_coord % (n_coords_per_geom[on_geom] - 1)
        center_ix = offset + (on_coord + 1) % (n_coords_per_geom[on_geom] - 1)
        right_ix = offset + (on_coord + 2) % (n_coords_per_geom[on_geom] - 1)
        # grab the actual coordinates corresponding to the triple
        left = points[left_ix]
        center = points[center_ix]
        right = points[right_ix]
        # build the line segments originating at center
        a = left - center
        b = right - center
        # compute the angle between the segments
        angle = numpy.arctan2(a[0] * b[1] - a[1] * b[0], numpy.dot(a, b))
        result.append(angle)
        on_coord += 1
    return result


# -------------------- IDEAL SHAPE MEASURES -------------------- #


def isoperimetric_quotient(collection):
    """
    The Isoperimetric quotient, defined as the ratio of a polygon's area to the
    area of the equi-perimeter circle.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, numpy.ndarray, list
        Input collection of polygons.

    Returns
    -------

    numpy.ndarray
        An array of the same length as the input collection, containing the
        Isoperimetric quotient for each polygon in the collection.

    Notes
    -----
    Altman's :math:`PA_1` measure :cite:`altman1998Districting`.

    The formula is given by:

    .. math::
        IPQ = \\frac{4 \\pi A}{P^2}

    Where :math:`A` is the area of the polygon and
    :math:`P` is the perimeter of the polygon.

    The :math:`IPQ` is scale invariant and due to the inclusion
    of :math:`\\pi` in the formula, it is bounded between 0 and 1, with 1
    representing a perfect circle, the most compact shape by this measure.
    """

    ga = _cast(collection)
    return (4 * numpy.pi * shapely.area(ga)) / (shapely.measurement.length(ga) ** 2)


def isoareal_quotient(collection):
    """
    The Isoareal quotient, defined as the ratio of a polygon's perimeter to the
    perimeter of the equi-areal circle.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, numpy.ndarray, list
        Input collection of polygons.

    Returns
    -------

    numpy.ndarray
        An array of the same length as the input collection, containing the
        Isoareal quotient for each polygon in the collection.

    Notes
    -----
    Altman's :math:`PA_3` measure :cite:`altman1998Districting`.

    The formula is given by:

    .. math::
        IAQ = \\frac{2 \\sqrt{\\pi A}}{P}

    Where :math:`A` is the area of the polygon and :math:`P`
    is the perimeter of the polygon.

    With some manipulation, :math:`IAQ` can also be expressed as the square root
    of the Isoperimetric quotient, given by

    .. math::
        IAQ = \\frac{2 \\sqrt{\\pi A}}{P}
            = \\sqrt{\\frac{(2 \\sqrt{\\pi A})^2}{P^2}}
            = \\sqrt{\\frac{4 \\pi A}{P^2}}
            = \\sqrt{IPQ}

    Therefore, `isoareal_quotient` is implemented as
    `numpy.sqrt(isoperimetric_quotient(collection))`.
    Importantly, this means that the :math:`IAQ` and :math:`IPQ`
    will rank shapes identically.

    The :math:`IAQ` is scale invariant and due to the inclusion
    of :math:`\\pi` in the formula, it is bounded between 0 and 1, with 1
    representing a perfect circle, the most compact shape by this measure.
    """
    return numpy.sqrt(isoperimetric_quotient(collection))


def minimum_bounding_circle_ratio(collection):
    """
    The Reock compactness measure, defined by the ratio of areas between the
    minimum bounding/containing circle of a shape and the shape itself.

    Measure A1 in :cite:`altman1998Districting`,
    cited for Frolov (1974), but earlier from Reock
    (1963)
    """
    ga = _cast(collection)
    mbca = (shapely.minimum_bounding_radius(ga) ** 2) * numpy.pi
    return shapely.area(ga) / mbca


def radii_ratio(collection):
    """
    The Flaherty & Crumplin (1992) index, OS_3 in :cite:`altman1998Districting`.

    The ratio of the radius of the equi-areal circle to the radius of the MBC
    """
    ga = _cast(collection)
    r_eac = numpy.sqrt(shapely.area(ga) / numpy.pi)
    r_mbc = shapely.minimum_bounding_radius(ga)
    return r_eac / r_mbc


def diameter_ratio(collection, rotated=True):
    """
    The Flaherty & Crumplin (1992) length-width measure, stated as measure LW_7
    in :cite:`altman1998Districting`.

    It is given as the ratio between the minimum and maximum shape diameter.
    """
    ga = _cast(collection)
    if rotated:
        box = shapely.minimum_rotated_rectangle(ga)
        coords = shapely.get_coordinates(box)
        a, b, _, d = (coords[0::5], coords[1::5], coords[2::5], coords[3::5])
        widths = numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))
        heights = numpy.sqrt(numpy.sum((a - d) ** 2, axis=1))
    else:
        box = shapely.bounds(ga)
        (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
        widths, heights = numpy.abs(xmax - xmin), numpy.abs(ymax - ymin)
    return numpy.minimum(widths, heights) / numpy.maximum(widths, heights)


def length_width_diff(collection):
    """
    The Eig & Seitzinger (1981) shape measure, defined as:

    L - W

    Where L is the maximal North-South extent and W is the maximal East-West
    extent.

    Defined as measure LW_5 in :cite:`altman1998Districting`
    """
    ga = _cast(collection)
    box = shapely.bounds(ga)
    (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
    width, height = numpy.abs(xmax - xmin), numpy.abs(ymax - ymin)
    return height - width


def boundary_amplitude(collection):
    """
    The boundary amplitude (adapted from Wang & Huang (2012)) is the
    length of the boundary of the convex hull divided by the length of the
    boundary of the original shape.

    Notes
    -----

    This is inverted from Wang & Huang (2012) in order to provide a value
    between zero and one, like many of the other ideal shape-based indices.
    """
    ga = _cast(collection)
    return shapely.length(shapely.convex_hull(ga)) / shapely.length(ga)


def convex_hull_ratio(collection):
    """
    ratio of the area of the convex hull to the area of the shape itself

    Altman's A_3 measure, from Neimi et al 1991.
    """
    ga = _cast(collection)
    return shapely.area(ga) / shapely.area(shapely.convex_hull(ga))


def fractal_dimension(collection, support="hex"):
    """
    The fractal dimension of the boundary of a shape, assuming a given
    spatial support for the geometries.

    Note that this derivation assumes a specific ideal spatial support
    for the polygon, and is thus may not return valid results for
    complex or highly irregular geometries.
    """
    ga = _cast(collection)
    P = shapely.length(ga)
    A = shapely.area(ga)
    if support == "hex":
        return 2 * numpy.log(P / 6) / numpy.log(A / (3 * numpy.sin(numpy.pi / 3)))
    elif support == "square":
        return 2 * numpy.log(P / 4) / numpy.log(A)
    elif support == "circle":
        return 2 * numpy.log(P / (2 * numpy.pi)) / numpy.log(A / numpy.pi)
    else:
        raise ValueError(
            "The support argument must be one of 'hex', 'circle', or 'square', "
            f"but {support} was provided."
        )


def squareness(collection):
    """
    Measures how different is a given shape from an equi-areal square

    The index is close to 0 for highly irregular shapes and to 1.3 for circular shapes.
    It equals 1 for squares.

    .. math::
        \\begin{equation}
        \\frac{
            \\sqrt{A}}{P^{2}}
            \\times
            \\frac{\\left(4 \\sqrt{\\left.A\\right)}^{2}\\right.}{\\sqrt{A}}
            =
            \\frac{\\left(4 \\sqrt{A}\\right)^{2}}{P{ }^{2}}
            =
            \\left(\\frac{4 \\sqrt{A}}{P}\\right)^{2}
        \\end{equation}

    where :math:`A` is the area and :math:`P` is the perimeter.

    Notes
    -----
    Implementation follows :cite:`basaraner2017`.

    """
    ga = _cast(collection)
    return ((numpy.sqrt(shapely.area(ga)) * 4) / shapely.length(ga)) ** 2


def rectangularity(collection):
    """
    Ratio of the area of the shape to the area
    of its minimum bounding rotated rectangle

    Reveals a polygon’s degree of being curved inward.

    .. math::
        \\frac{A}{A_{MBR}}

    where :math:`A` is the area and :math:`A_{MBR}`
    is the area of minimum bounding
    rotated rectangle.

    Notes
    -----
    Implementation follows :cite:`basaraner2017`.
    """
    ga = _cast(collection)
    return shapely.area(ga) / shapely.area(shapely.minimum_rotated_rectangle(ga))


def shape_index(collection):
    """
    Schumm’s shape index (Schumm (1956) in MacEachren 1985)

    .. math::
        {\\sqrt{{A} \\over {\\pi}}} \\over {R}

    where :math:`A` is the area and :math:`R` is the radius of the minimum bounding
    circle.

    Notes
    -----
    Implementation follows :cite:`maceachren1985compactness`.

    """
    ga = _cast(collection)
    return numpy.sqrt(shapely.area(ga) / numpy.pi) / shapely.minimum_bounding_radius(ga)


def equivalent_rectangular_index(collection):
    """
    Deviation of a polygon from an equivalent rectangle

    .. math::
        \\frac{\\sqrt{A}}{A_{MBR}}
        \\times
        \\frac{P_{MBR}}{P}

    where :math:`A` is the area, :math:`A_{MBR}` is the area of minimum bounding
    rotated rectangle, :math:`P` is the perimeter, :math:`P_{MBR}` is the perimeter
    of minimum bounding rotated rectangle.

    Notes
    -----
    Implementation follows :cite:`basaraner2017`.
    """
    ga = _cast(collection)
    box = shapely.minimum_rotated_rectangle(ga)
    return numpy.sqrt(shapely.area(ga) / shapely.area(box)) * (
        shapely.length(box) / shapely.length(ga)
    )


# -------------------- VOLMETRIC MEASURES ------------------- #


def form_factor(collection, height):
    """
    Computes volumetric compactness

    .. math::
        \\frac{A}{(A \\times H)^{\\frac{2}{3}}}

    where :math:`A` is the area and :math:`H` is polygon's
    height.

    Notes
    -----
    Implementation follows :cite:`bourdic2012`.
    """
    ga = _cast(collection)
    A = shapely.area(ga)
    V = A * height
    zeros = V == 0
    res = numpy.zeros(len(ga))
    res[~zeros] = A[~zeros] / (V[~zeros] ** (2 / 3))
    return res


# -------------------- INERTIAL MEASURES -------------------- #


def moment_of_inertia(collection, normalize=False, ref_pt=None):
    """
    Compute moment of inertia (second moment of area) per geometry.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, numpy.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns moment normalized by reference cricle of same area.
        Default is False.
    ref_pt : GeoSeries, Shapely Point or list of Points, array-like of shape (2,) or (n, 2), optional
        If provided, shifts moment to be with respect to this point or points.
        The default behavior (default: ``None``) is to calculate the moment
        about the centroid of each geometry. Points may be passed as
        as array-like of coordinates or point geometries. If a single point,
        all moments are calculated with respect to that reference point. If
        of length equal to collection, the moment for each geometry in
        collection is calculated with respect to the reference point with
        the same index. To return moment about the origin, explicitly set to (0, 0).

    Returns
    -------
    numpy.ndarray
        Array of moment of inertia values for each geometry in the collection.

    Notes
    -----
    Calculates the moment of inertia of each geometry in the collection. Can handle
    polygons with holes and multipolygons by summing the moments of inertia of each
    part. The moment of inertia is calculated about the centroid of each geometry
    by default, but can also be calculated about a specified reference point or points.

    The moment of inertia is the variance of the distance of all points in a shape to a
    reference point, which can be interpreted as an axis of rotation  perpendicular to
    the plane of the shape. The first parameter should be a GeoDataFrame or array-like
    of Polygons or MultiPolygons. If normalization is requested, the moment of inertia
    is compared with that of a circle of equal area. This is discussed in detail below.

    Moments of inertia with weights (such as population) may be calculated using
    `moment_of_inertia_regions`. Without weights, the moment of inertia is known as the
    area moment of inertia or the second moment of area, and can be quickly calculated
    from the polygon vertices using the shoelace formula. The second moments of area
    about the x and y axes are calculated as:

    .. math::
        I_x = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(y_i^2 + y_i y_{i+1} + y_{i+1}^2)

        I_y = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(x_i^2 + x_i x_{i+1} + x_{i+1}^2)

    where indices wrap around (n+1 = 1).

    The moments are then adjusted to be relative to a reference point using the
    parallel axis theorem. The moment of inertia is then the sum of :math:`I_x` and
    :math:`I_y`. In geographic contexts, the reference point is typically the
    centroid, which is the default if no reference point is provided. However, other
    points of interest may be used, such as a capital city or the residence of
    a political representative.

    The moment of inertia can be normalized to provide a compactness measure
    by comparing it with the moment of inertia of a circle of equal area.
    This is calculated as the ratio of the moment of inertia of the circle
    to that of the shape about its centroid:

    .. math::
        C_{MI} = \\frac{I_{circle}}{I_{shape}}
               = \\frac{A^2}{2 \\pi I_{shape}}

    where :math:`A` is the area of the polygon and :math:`I_{shape}` is the moment of
    inertia of the polygon. This formulation is from Li, et al (2013). The value of
    :math:`C_{MI}` is bounded between 0 and 1, with 1 representing a perfect circle,
    the most compact shape by this measure.

    References
    ----------
    .. [1] Godwin, A. N. 1980. "Simple Calculation of Moments of Inertia for Polygons."
        International Journal of Mathematical Education in Science and Technology
        11 (4): 577–86. https://doi.org/10.1080/0020739800110414.
    .. [2] Li, Wenwen, Michael F. Goodchild, and Richard Church. 2013. "An Efficient
        Measure of Compactness for Two-Dimensional Shapes and Its Application in
        Regionalization Problems." International Journal of Geographical Information
        Science 27 (6): 1227–50. https://doi.org/10.1080/13658816.2012.752093.
    .. [3] https://en.wikipedia.org/wiki/Second_moment_of_area#List_of_second_moments_of_area
    """  # noqa: E501
    ga = _cast(collection)
    ga = orient_polygons(ga)  # shapely.orient_polygons(ga) #

    if ref_pt is not None:
        coords = _cast_pts_as_array(ref_pt)
        if not (coords.shape == (2,) or coords.shape == (len(ga), 2)):
            msg = (
                "`ref_pt` must be a single point (or coordinate pair) or one point "
                f"(or coordinate pairs) per geometry in `collection` ({len(ga)})"
            )
            raise ValueError(msg)

    Js = []
    for i, geom in enumerate(ga):
        A, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid([geom])
        if ref_pt is not None:
            if coords.shape == (2,):
                dx = Cx - coords[0]
                dy = Cy - coords[1]
            else:  # Already tested, if not (2,), must be (len(ga), 2)
                dx = Cx - coords[i][0]
                dy = Cy - coords[i][1]
            J += A * (dx**2 + dy**2)
        if normalize:
            J = A**2 / (2 * numpy.pi * J)
        Js.append(J)

    return numpy.asarray(Js)


# Alias for users familiar with math/engineering terminology
second_moment_of_area = moment_of_inertia


def second_areal_moment(collection):
    # Alias to preserve old API.

    msg = (
        "`second_areal_moment` is deprecated and will be removed in a "
        "future version. Use `moment_of_inertia` instead. The current function "
        "is an alias for `moment_of_inertia` which does not expose all parameters."
    )

    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    return moment_of_inertia(collection, normalize=False, ref_pt=None)


def moment_of_inertia_regions(
    collection, normalize=False, ref_pt=None, regions=None, weights=None
):
    """
    Compute weighted moment of inertia per region. See Notes for behavior when
    either regions or weights are omitted.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, numpy.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns moment normalized by reference cricle of same area and
        mass (sum of weights in the region). Default is False.
    ref_pt : GeoSeries, Shapely Point or list of Points, array-like of shape
        (2,) or (n, 2), or dict of any of these, optional
        If provided, shifts moment to be with respect to this point or points.
        The default behavior (default: ``None``) is to calculate the moment
        about the centroid of each region or geometry. If `regions` is
        provided, this must be a `dict` with one item per region, with the key
        equal to the region identifier and the value equal to a point geometry
        or point coordinates. See `moment_of_inertia` for details.
    regions : array-like, str, optional
        An iterable of region identifiers of the same length as `collection`
        that each geometry is assigned to, or the name of a column in the
        GeoDataFrame to use for region assignment. If None (default), moment of inertia
        is calculated for each geometry in `collection` without regionalization.
    weights : array-like, optional
        An iterable of weights (e.g., population) of the same length as
        `collection` that are applied to each geometry in `collection`, or the
        name of a column in the GeoDataFrame to use for weights. If None (default),
        calculates second moment of area using the shoelace formula.

    Returns
    -------
    pandas.Series or numpy.ndarray
        If `regions` is provided, returns a `pandas.Series` indexed
        by unique region IDs, containing moments per region. If `regions` is
        omitted, retuns an `numpy.ndarray` of moments per geometry in `collection`.

    Calculates the mass moment of inertia for regions defined by assignment of
    geometries in the collection.

    Requires either a column name with region assignments or an array-like of
    region IDs. Weights can be provided as a column name or an array-like. If
    either `regions` or `weights` is a string, `collection` must be a
    GeoDataFrame and the column name must be a valid column in the GeoDataFrame.
    If weights are not provided, geometries weighted by area, which is equivalent
    to the second moment of area.

    Notes
    -----
    See `moment_of_inertia` for an introduction. Note that moment of
    intertia and second moment of area are used interchangeably in some
    disciplines (see Li, et al 2013). In this discussion "mass moment of
    inertia" is used to represent a weighted moment of inertia, and "second
    moment of area" is used for an unweighted moment of inertia. This
    module's `moment_of_inertia` function returns the second moment of
    area only.

    This function extends the `moment_of_inertia` implementation to allow
    regionalization and/or weighting. Region identifiers are provided via
    the `regions` parameter. Weighting is provided via the `weights` parameter.

    If region identifiers are provided, the geometries in `collection` are
    subareas within larger regions defined by the region identifier. The moment
    of inertia (possibly weighted) is calculated for each region as a whole. If
    omitted, moments are calculated for each geometry in `collection`.

    If weights are provided, the mass moment of inertia is calculated using a
    value of interest, such as population, as the mass of the
    shape. Implementation details vary with whether regions are provided
    or normalization is requested. This is discussed in detail below. First,
    if weights are **not** provided, each geometry is weighted by area, and
    the result is equivalent to the second moment of area. If weights are
    provided and normalization is requested, the mass of the shape
    and the reference circle cancel, and the result is equivalent to
    the normalized second moment of area.

    The mass moment of inertia of a point is equal to its mass times the
    distance to a reference point squared. For a shape rotating about its
    centroid the moment of inertia is:

    .. math::
        I = \\sum_{i} m_i r_i^2

    where :math:`r_i` is the distance from the point to a reference point,
    :math:`m_i` is the mass at each point, and there are an infinite number
    of points filling the shape.

    The mass moment of inertia can be calculated for an area of uniform density
    or an area of varying density. For areas of uniform density, this is the
    equivalent of the second moment of area times the mass of the shape
    divided by the area. If `regions` is not provided, the mass moment of inertia
    is calculated for each geometry in `collection` as an area of uniform
    density. This is implemented as:


    .. math::
        I_M = I_A m / A

    where :math:`I_A` is the second moment of area (calculated by `moment_of_inertia`)
    and :math:`m` and :math:`A` are the mass and area of the shape, respectively.

    For a region of varying density, region identifiers must be provided via
    the `regions` parameter. :math:`I_M` is calculated for each geometry in
    `collection` per the equation above, with each geometry representing a
    subarea of the region with mass given
    by `weights`. The mass moment of inertia of the region is then the sum of
    the mass moments of inertia of each subarea shifted to the reference point
    (using the parallel axis theorem) by adding :math:`m d^2`, where :math:`d`
    is the distance from the centroid of each subarea to the reference point
    for the region.

    If reference points are not provided, the mass moment of inertia is
    calculated with respect to the centroid of each region (calculated by the
    algorithm). If provided, `ref_pt` may be a single point about which the
    mass moment of inertia is calculated for all regions, or it may be a `dict`
    with one item per region indicating the reference point (value) to use for
    that region (key). If any region identifier does not appear as a `dict` key,
    an error is raise. Extra keys in the `dict` that do not correspond to any
    region identifiers are ignored with a warning.

    The mass moment of inertia can be normalized to provide a compactness
    measure using the formula from Fan, et al. (2015):

    .. math::
        C_{NMMI} = \\frac{M A}{2 \\pi I_M}

    where :math:`I_M` is the mass moment of inertia of the shape. This
    represents the ratio of the mass moment of inertia of a circle with the
    same area and mass as the shape to the mass moment of inertia of the shape.
    In this case, the measure can exceed 1, as will happen when mass is
    concentrated near the centroid of the shape. This can be interpreted as a
    more compact distribution than a uniform circle.

    References
    ----------
    .. [1] Weaver, James B., and Sidney W. Hess. 1963. "A Procedure for Nonpartisan
        Districting: Development of Computer Techniques."
        Yale Law Journal 73 (2): 288–308. https://doi.org/10.2307/794769
    .. [2] Li, Wenwen, Michael F. Goodchild, and Richard Church. 2013. "An Efficient
        Measure of Compactness for Two-Dimensional Shapes and Its Application in
        Regionalization Problems." International Journal of Geographical Information
        Science 27 (6): 1227–50. https://doi.org/10.1080/13658816.2012.752093.
    .. [3] Fan, Chao, Wenwen Li, Levi J. Wolf and Soe W. Myint. 2015 "A Spatiotemporal
        Compactness Pattern Analysis of Congressional Districts to Assess Partisan
        Gerrymandering: A Case Study with California and North Carolina." Annals of the
        Association of American Geographers 105 (4): 736-753.
        https://doi.org/10.1080/00045608.2015.1039109
    """

    ga = _cast(collection)
    ga = orient_polygons(ga)  # shapely.orient_polygons(ga)

    # Handle weights (masses).
    if weights is not None:
        # If weights is a string, interpret as column name and extract from GeoDataFrame
        if isinstance(weights, str):
            if (
                not isinstance(collection, geopandas.GeoDataFrame)
                or weights not in collection.columns
            ):
                msg = (
                    "If `weights` is a string, it must be the name of a column "
                    "in `collection`, which must be a GeoDataFrame."
                )
                raise ValueError(msg)
            weights = numpy.asarray(collection[weights])
        else:
            weights = numpy.asarray(weights)

    # If regions is missing, calculate MOI per geometry. Must be handled differently
    # depending on whether weights is missing or normalization is requested.
    if regions is None:
        if weights is None or normalize:
            # If weights are missing, this reduces to second moment of area.
            # If weights are present but we are normalizing, this also reduces
            # to (normalized) second moment of area. Pass to moment_of_inertia
            # and return.
            return moment_of_inertia(collection, normalize=normalize, ref_pt=ref_pt)
        else:  # Weights are present but we are not normalizing
            # Adjust second moment of area for mass. I_M = I_A * m / A
            return (
                moment_of_inertia(collection, normalize=False, ref_pt=ref_pt)
                * weights
                / numpy.asarray(shapely.area(ga))
            )

    # Handle region IDs.
    else:
        # If regions is a string, interpret as column name and extract from GeoDataFrame
        if isinstance(regions, str):
            if (
                not isinstance(collection, geopandas.GeoDataFrame)
                or regions not in collection.columns
            ):
                msg = (
                    "If `regions` is a string, it must be the name of a column "
                    "in `collection`, which must be a GeoDataFrame."
                )
                raise ValueError(msg)
            regions = numpy.asarray(collection[regions])
        else:
            regions = numpy.asarray(regions)

        unique_regions = numpy.unique(regions)

    # If we get to this point, regions are present, but weights might be none. Will
    # be handled differently in loop on unique regions.

    # Handle reference point(s), if provided
    if ref_pt is not None:
        if isinstance(ref_pt, dict):
            k = list(ref_pt.keys())

            # Make sure we have one entry per region
            if set(unique_regions) <= set(k):
                if set(unique_regions) < set(k):
                    # Extra unused regions in dict. Issue warning and remove them
                    msg = (
                        "Keys found in `ref_pt` that are not regions in `regions`. "
                        "Extra regions will be ignored."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                # Cast all points in dict values
                ref_pt = {
                    key: _cast_pts_as_array(value) for key, value in ref_pt.items()
                }

            else:
                msg = (
                    "Regions found in `regions` that are not in `ref_pt`. If `ref_pt` "
                    "is a `dict`, every region must have an entry in the `dict`."
                )
                raise ValueError(msg)
        else:
            ref_pt = _cast_pts_as_array(ref_pt)
            # If not passed as a dictionary,
            # this should be single global reference point
            if ref_pt.shape != (2,):
                msg = (
                    "`ref_pt` must be a single point (or coordinate pair) "
                    "or a dictionary with one point per region."
                )
                raise ValueError(msg)

    Js = []

    for region in unique_regions:
        # Use mask to calculate moment of inertia over each region. When user
        # omits and `regions`, this effectively becomes a simple
        # loop over ga, returning mass moment of inertia per geometry.
        mask = regions == region
        geoms = ga[mask]

        if weights is None:
            # Calcluate area moment of inertia per region
            # Determine reference point for shifting, or use centroid
            if ref_pt is None:
                # Use centroid
                pt = None
            elif isinstance(ref_pt, dict):
                # Use regional reference point
                pt = ref_pt[region]
            else:
                # Use global reference point
                pt = ref_pt

            Js.append(moment_of_inertia_global(geoms, normalize=normalize, ref_pt=pt))

        else:
            m = weights[mask]

            # A_tot, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid(geoms)
            moments = numpy.asarray([_moments_about_centroid(geom) for geom in geoms])
            a = moments[:, 0]
            cx = moments[:, 1]
            cy = moments[:, 2]
            c = numpy.column_stack((cx, cy))
            J = moments[:, 5]

            # Area and centroid of region
            A = numpy.sum(a)
            C = numpy.sum(m[:, None] * c, axis=0) / m.sum()

            # Determine reference point for shifting, or use centroid
            if ref_pt is None:
                # Use centroid
                pt = C
            elif isinstance(ref_pt, dict):
                # Use regional reference point
                pt = ref_pt[region]
            else:
                # Use global reference point
                pt = ref_pt

            # Distance squared, don't actually need distance,
            # so don't bother taking square root
            d2 = numpy.sum((c - pt) ** 2, axis=1)

            J = numpy.sum((m / a) * J + m * d2)

            if normalize:
                J = m.sum() * A / (2 * numpy.pi * J)

            Js.append(J)

    return pandas.Series(Js, index=unique_regions)


def moa_ratio(collection):
    """
    Computes the ratio of the second moment of area (like Li et al (2013)) to
    the moment of area of a circle with the same perimeter.
    """
    msg = "`moa_ratio` is deprecated and will be removed in a future version."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    ga = _cast(collection)
    r = shapely.length(ga) / (2 * numpy.pi)
    return (numpy.pi * 0.5 * r**4) / second_areal_moment(ga)


def nmi(collection):
    """
    Computes the normalized moment of inertia

    Notes
    -----
    Implemented as `moment_of_inertia(collection, normalize=True, ref_pt=None)`.
    See `moment_of_inertia` for details.
    """

    return moment_of_inertia(collection, normalize=True, ref_pt=None)


def moment_of_inertia_global(collection, normalize=False, ref_pt=None):
    """
    Compute moment of inertia (second moment of area)
    for an entire collection of geometries combined.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, numpy.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns moment normalized by reference cricle of same area.
        Default is False.
    ref_pt : GeoSeries, Shapely Point, or array-like of shape (2,), optional
        If provided, shifts moment to be with respect to this point.
        The default behavior (default: ``None``) is to calculate the moment
        about the centroid of the entire `collection`. Point may be passed as
        as array-like of coordinates or a point geometry (which can include a
        GeoSeries of length 1). To return moment about the origin,
        explicitly set to (0, 0).

    Returns
    -------
    float
        Moment of inertia for the entire collection.

    Notes
    -------
    This is a convenience function to calculate the second moment of area for
    an entire collection, which will usually be faster than running a geospatial
    dissolve on the geometries and then running `moment_of_inertia` on the
    result.

    This will *not* calculate a mass moment of inertia. To calculate the mass
    moment of inertia for an entire collection, weighted by the masses of the
    geometries in the collection, assign all geometries to the same region:

    ```
    moment_of_inertia_regions(
        collection,
        regions=numpy.repeat(1, len(collection))
    ), weights=<weights vector>
    ````

    The `normalize` and `ref_pt` parameters may be used as usual.
    """
    ga = _cast(collection)
    ga = orient_polygons(ga)  # shapely.orient_polygons(ga)

    A, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid(ga)

    if ref_pt is not None:
        dx = Cx - ref_pt[0]
        dy = Cy - ref_pt[1]
        J += A * (dx**2 + dy**2)

    if normalize:
        J = A**2 / (2 * numpy.pi * J)

    return J


# -------------------------
# Helper Functions for Inertial Measures
# -------------------------


def _dump_rings(geoms):
    """
    Yield all exterior and interior rings of a collection of polygons/multipolygons
    as numpy arrays of coordinates.

    Parameters
    ----------
    geoms : sequence of shapely geometries
        Polygons or multipolygons.

    Yields
    ------
    numpy.ndarray
        Array of shape (n, 2) containing coordinates of a ring.
    """
    for poly in shapely.get_parts(geoms):
        yield shapely.get_coordinates(poly.exterior)
        for interior in poly.interiors:
            yield shapely.get_coordinates(interior)


@njit
def _geometric_moments_ring(pts, shift_to_centroid=True):
    """
    Compute area, centroid, and second moments of a single polygon ring.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of coordinates defining the ring, shape (n, 2).
    shift_to_centroid : bool, default True
        If True, apply the parallel axis theorem to shift moments to the ring centroid.

    Returns
    -------
    A : float
        Signed area of the ring.
    cx, cy : float
        Coordinates of the ring centroid.
    Ixx, Iyy : float
        Second moments of area about the centroid
        (or origin if ``shift_to_centroid=False``).

    Notes
    -----
    - Polar moment of area J can be obtained by summing Ixx + Iyy at the calling level.
    - This function does not compute J itself, leaving derived quantities to
        higher-level functions.
    """
    x = pts[:, 0]
    y = pts[:, 1]

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]

    A = cross.sum() / 2
    Sx = ((x[:-1] + x[1:]) * cross).sum()
    Sy = ((y[:-1] + y[1:]) * cross).sum()

    Ixx_origin = ((y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * cross).sum() / 12
    Iyy_origin = ((x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * cross).sum() / 12

    cx = Sx / (6 * A)
    cy = Sy / (6 * A)

    if shift_to_centroid:
        Ixx = Ixx_origin - A * cy**2
        Iyy = Iyy_origin - A * cx**2
    else:
        Ixx, Iyy = Ixx_origin, Iyy_origin

    return A, cx, cy, Ixx, Iyy


def _moments_about_centroid(geoms):
    """
    Compute combined moments of area for a collection of polygons.

    Parameters
    ----------
    geoms : sequence of shapely geometries
        Input polygons or multipolygons.

    Returns
    -------
    A_tot : float
        Total area of the collection.
    Cx, Cy : float
        Centroid coordinates of the combined geometry collection.
    Ixx, Iyy : float
        Second moments of area about the centroid.
    J : float
        Polar moment of area (Ixx + Iyy) about the centroid.
    """
    A_tot = 0.0
    Mx_tot = 0.0
    My_tot = 0.0
    Ixx0_tot = 0.0
    Iyy0_tot = 0.0

    for ring in _dump_rings(geoms):
        A, cx, cy, Ixx_c, Iyy_c = _geometric_moments_ring(ring, shift_to_centroid=False)

        # First moments
        A_tot += A
        Mx_tot += A * cx
        My_tot += A * cy

        # Accumulate inertia about origin
        Ixx0_tot += Ixx_c
        Iyy0_tot += Iyy_c

    # Centroid of entire collection
    Cx = Mx_tot / A_tot
    Cy = My_tot / A_tot

    # Shift to centroid
    Ixx = Ixx0_tot - A_tot * Cy**2
    Iyy = Iyy0_tot - A_tot * Cx**2

    J = Ixx + Iyy

    return A_tot, Cx, Cy, Ixx, Iyy, J


# -------------------- OTHER MEASURES -------------------- #


def reflexive_angle_ratio(collection):
    """
    The Taylor reflexive angle index, measure OS_4 in :cite:`altman1998Districting`

    (N-R)/(N+R), the difference in number between non-reflexive angles and
    reflexive angles in a polygon, divided by the number of angles in the
    polygon in general.
    """
    angles, geom_indices = get_angles(collection, return_indices=True)
    return (
        pandas.DataFrame.from_dict(dict(is_reflex=angles < 0, geom_ix=geom_indices))
        .groupby("geom_ix")
        .is_reflex.mean()
        .values
    )
