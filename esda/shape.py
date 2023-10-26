import numpy
import pandas
from packaging.version import Version

try:
    import shapely
except (ImportError, ModuleNotFoundError):
    pass  # gets handled at the _cast level.

from .crand import njit, prange


__author__ = (
    "Martin Fleischmann <martin@fleischmann.net>",
    "Levi John Wolf <levi.john.wolf@gmail.com>",
    "Alan Murray <amurray@ucsb.edu>",
    "Jiwan Baik <jiwon.baik@geog.ucsb.edu>",
)


# -------------------- UTILITIES --------------------#
def _cast(collection):
    """
    Cast a collection to a shapely geometry array.
    """
    try:
        import geopandas
        import shapely
    except (ImportError, ModuleNotFoundError) as exception:
        raise type(exception)(
            "shapely and geopandas are required for shape statistics."
        )

    if Version(shapely.__version__) < Version("2"):
        raise ImportError("Shapely 2.0 or newer is required.")

    if isinstance(collection, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        return numpy.asarray(collection.geometry.array)
    else:
        if isinstance(collection, (numpy.ndarray, list)):
            return numpy.asarray(collection)
        else:
            return numpy.array([collection])


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
    offset = int(0)
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
        angle = numpy.math.atan2(a[0] * b[1] - a[1] * b[0], numpy.dot(a, b))
        result.append(angle)
        on_coord += 1
    return result


# -------------------- IDEAL SHAPE MEASURES -------------------- #


def isoperimetric_quotient(collection):
    r"""
    The Isoperimetric quotient, defined as the ratio of a polygon's area to the
    area of the equi-perimeter circle.

    Altman's PA_1 measure :cite:`altman1998Districting`

    Construction:

    let:
    p_d = perimeter of polygon
    a_d = area of polygon

    a_c = area of the constructed circle
    r = radius of constructed circle

    then the relationship between the constructed radius and the polygon
    perimeter is:
    p_d = 2 \pi r
    p_d / (2 \pi) = r

    meaning the area of the circle can be expressed as:
    a_c = \pi r^2
    a_c = \pi (p_d / (2\pi))^2

    implying finally that the IPQ is:

    pp = (a_d) / (a_c) = (a_d) / ((p_d / (2*\pi))^2 * \pi) = (a_d) / (p_d**2 / (4\PI))
    """
    ga = _cast(collection)
    return (4 * numpy.pi * shapely.area(ga)) / (shapely.measurement.length(ga) ** 2)


def isoareal_quotient(collection):
    """
    The Isoareal quotient, defined as the ratio of a polygon's perimeter to the
    perimeter of the equi-areal circle

    Altman's PA_3 measure, and proportional to the PA_4 measure
    :cite:`altman1998Districting`
    """
    ga = _cast(collection)
    return (
        2 * numpy.pi * numpy.sqrt(shapely.area(ga) / numpy.pi)
    ) / shapely.measurement.length(ga)


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

    Where L is the maximal east-west extent and W is the maximal north-south
    extent.

    Defined as measure LW_5 in :cite:`altman1998Districting`
    """
    ga = _cast(collection)
    box = shapely.bounds(ga)
    (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
    width, height = numpy.abs(xmax - xmin), numpy.abs(ymax - ymin)
    return width - height


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


def moment_of_inertia(collection):
    """
    Computes the moment of inertia of the polygon.

    This treats each boundary point as a point-mass of 1.

    Thus, for constant unit mass at each boundary point,
    the MoI of this pointcloud is

    .. math::
        \\sum_i d_{i,c}^2

    where c is the centroid of the polygon

    Altman's OS_1 measure :cite:`altman1998Districting`, cited in Boyce and Clark
    (1964), also used in Weaver and Hess (1963).
    """
    ga = _cast(collection)
    coords = shapely.get_coordinates(ga)
    geom_ixs = numpy.repeat(numpy.arange(len(ga)), shapely.get_num_coordinates(ga))
    centroids = shapely.get_coordinates(shapely.centroid(ga))[geom_ixs]
    squared_euclidean = numpy.sum((coords - centroids) ** 2, axis=1)
    dists = (
        pandas.DataFrame.from_dict(dict(d2=squared_euclidean, geom_ix=geom_ixs))
        .groupby("geom_ix")
        .d2.sum()
    ).values
    return shapely.area(ga) / numpy.sqrt(2 * dists)


def moa_ratio(collection):
    """
    Computes the ratio of the second moment of area (like Li et al (2013)) to
    the moment of area of a circle with the same area.
    """
    ga = _cast(collection)
    r = shapely.length(ga) / (2 * numpy.pi)
    return (numpy.pi * 0.5 * r**4) / second_areal_moment(ga)


def nmi(collection):
    """
    Computes the Normalized Moment of Inertia from Li et al (2013), recognizing
    that it is the relationship between the area of a shape squared divided by
    its second moment of area.
    """
    ga = _cast(collection)
    return shapely.area(ga) ** 2 / (2 * second_areal_moment(ga) * numpy.pi)


def second_areal_moment(collection):
    """
    Using equation listed on en.wikipedia.org/wiki/Second_moment_of_area#Any_polygon, the second
    moment of area is the sum of the inertia across the x and y axes:

    The :math:`x` axis is given by:

    .. math::

        I_x = (1/12)\\sum^{N}_{i=1} (x_i y_{i+1} - x_{i+1}y_i) (x_i^2 + x_ix_{i+1} + x_{i+1}^2)

    While the :math:`y` axis is in a similar form:

    .. math::

        I_y = (1/12)\\sum^{N}_{i=1} (x_i y_{i+1} - x_{i+1}y_i) (y_i^2 + y_iy_{i+1} + y_{i+1}^2)

    where :math:`x_i`, :math:`y_i` is the current point and :math:`x_{i+1}`, :math:`y_{i+1}` is the next point,
    and where :math:`x_{n+1} = x_1, y_{n+1} = y_1`. For multipart polygons with holes,
    all parts are treated as separate contributions to the overall centroid, which
    provides the same result as if all parts with holes are separately computed, and then
    merged together using the parallel axis theorem.

    References
    ----------
    Hally, D. 1987. The calculations of the moments of polygons. Canadian National
    Defense Research and Development Technical Memorandum 87/209.
    https://apps.dtic.mil/dtic/tr/fulltext/u2/a183444.pdf

    """
    ga = _cast(collection)
    import geopandas  # function level, to follow module design

    # construct a dataframe of the fundamental parts of all input polygons
    parts, collection_ix = shapely.get_parts(ga, return_index=True)
    rings, ring_ix = shapely.get_rings(parts, return_index=True)
    # get_rings() always returns the exterior first, then the interiors
    collection_ix = numpy.repeat(
        collection_ix, shapely.get_num_interior_rings(parts) + 1
    )
    # we need to work in polygon-space for the algorithms (centroid, shoelace calculation) to work
    polygon_rings = shapely.polygons(rings)
    is_external = numpy.zeros_like(collection_ix).astype(bool)
    # the first element is always external
    is_external[0] = True
    # and each subsequent element is external iff it is different from the preceeding index
    is_external[1:] = ring_ix[1:] != ring_ix[:-1]
    # now, our analysis frame contains a bunch of (guaranteed-to-be-simple) polygons
    # that represent either exterior rings or holes
    polygon_rings = geopandas.GeoDataFrame(
        dict(
            collection_ix=collection_ix,
            ring_within_geom_ix=ring_ix,
            is_external_ring=is_external,
        ),
        geometry=polygon_rings,
    )
    # the polygonal moi can be calculated using the same ring-based strategy,
    # and this could be parallelized if necessary over the elemental shapes with:

    # from joblib import Parallel, parallel_backend, delayed
    # with parallel_backend('loky', n_jobs=-1):
    #     engine = Parallel()
    #     promise = delayed(_second_moment_of_area_polygon)
    #     result = engine(promise(geom) for geom in polygon_rings.geometry.values)

    # but we will keep simple for now
    polygon_rings["moa"] = polygon_rings.geometry.apply(_second_moment_of_area_polygon)
    # the above algorithm computes an unsigned moa to be insensitive to winding direction.
    # however, we need to subtract the moa of holes. Hence, the sign of the moa is
    # -1 when the polygon is an internal ring and 1 otherwise:
    polygon_rings["sign"] = (1 - polygon_rings.is_external_ring * 2) * -1
    # shapely already uses the correct formulation for centroids
    polygon_rings["centroids"] = shapely.centroid(polygon_rings.geometry)
    # the inertia of parts applies to the overall center of mass:
    original_centroids = shapely.centroid(ga)
    polygon_rings["collection_centroid"] = original_centroids[collection_ix]
    # proportional to the squared distance between the original and part centroids:
    polygon_rings["radius"] = shapely.distance(
        polygon_rings.centroid.values, polygon_rings.collection_centroid.values
    )
    # now, we take the sum of (I+Ar^2) for each ring, treating the
    # contribution of holes as negative. Then, we take the sum of all of the contributions
    return (
        polygon_rings.groupby(["collection_ix", "ring_within_geom_ix"])
        .apply(
            lambda ring_in_part: (
                (ring_in_part.moa + ring_in_part.radius**2 * ring_in_part.area)
                * ring_in_part.sign
            ).sum()
        )
        .groupby(level="collection_ix")
        .sum()
        .values
    )


def _second_moment_of_area_polygon(polygon):
    """
    Compute the absolute value of the moment of area (i.e. ignoring winding direction)
    for an input polygon.
    """
    coordinates = shapely.get_coordinates(polygon)
    centroid = shapely.centroid(polygon)
    centroid_coords = shapely.get_coordinates(centroid)
    moi = _second_moa_ring_xplusy(coordinates - centroid_coords)
    return abs(moi)


@njit
def _second_moa_ring_xplusy(points):
    """
    implementation of the moment of area for a single ring
    """
    moi = 0
    for i in prange(len(points[:-1])):
        x_tail, y_tail = points[i]
        x_head, y_head = points[i + 1]
        xtyh = x_tail * y_head
        xhyt = x_head * y_tail
        xtyt = x_tail * y_tail
        xhyh = x_head * y_head
        moi += (xtyh - xhyt) * (
            x_head**2
            + x_head * x_tail
            + x_tail**2
            + y_head**2
            + y_head * y_tail
            + y_tail**2
        )
    return moi / 12


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
