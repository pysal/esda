import numpy
import pandas

try:
    import pygeos
except (ImportError, ModuleNotFoundError):
    pass  # gets handled at the _cast level.

from .crand import njit, prange


# -------------------- UTILITIES --------------------#
def _cast(collection):
    """
    Cast a collection to a pygeos geometry array.
    """
    try:
        import pygeos, geopandas
    except (ImportError, ModuleNotFoundError) as exception:
        raise type(exception)("pygeos and geopandas are required for shape statistics.")

    if isinstance(collection, (geopandas.GeoSeries, geopandas.GeoDataFrame)):
        return collection.geometry.values.data.squeeze()
    elif pygeos.is_geometry(collection).all():
        if isinstance(collection, (numpy.ndarray, list)):
            return numpy.asarray(collection)
        else:
            return numpy.array([collection])
    elif isinstance(collection, (numpy.ndarray, list)):
        return pygeos.from_shapely(collection).squeeze()
    else:
        return numpy.array([pygeos.from_shapely(collection)])


def get_angles(collection, return_indices=False):
    """
    Get the angles pertaining to each vertex of a set of polygons.
    This assumes the input are polygons.

    Arguments
    ---------
    ga  :   pygeos geometry array
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
    those counts (computed, for example, using pygeos.get_num_coordinates(ga)).
    Likewise, let K be a vector storing the number of parts each geometry has, k_i
    (computed, for example, using pygeos.get_num_geometries(ga))

    Then, the output is of shape (N - K).sum()

    """
    ga = _cast(collection)
    exploded = pygeos.get_parts(ga)
    coords = pygeos.get_coordinates(exploded)
    n_coords_per_geom = pygeos.get_num_coordinates(exploded)
    n_parts_per_geom = pygeos.get_num_geometries(exploded)
    angles = numpy.asarray(_get_angles(coords, n_coords_per_geom))
    if return_indices:
        return angles, numpy.repeat(
            numpy.arange(len(ga)),
            pygeos.get_num_coordinates(ga) - pygeos.get_num_geometries(ga),
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
    start = points[0]
    on_geom = 0
    on_coord = 0
    result = []
    n_points = len(points)
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
    """
    The Isoperimetric quotient, defined as the ratio of a polygon's area to the
    area of the equi-perimeter circle.

    Altman's PA_1 measure

    Construction:
    --------------
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
    return (4 * numpy.pi * pygeos.area(ga)) / (pygeos.measurement.length(ga) ** 2)


def isoareal_quotient(collection):
    """
    The Isoareal quotient, defined as the ratio of a polygon's perimeter to the
    perimeter of the equi-areal circle

    Altman's PA_3 measure, and proportional to the PA_4 measure
    """
    ga = _cast(collection)
    return (
        2 * numpy.pi * numpy.sqrt(pygeos.area(ga) / numpy.pi)
    ) / pygeos.measurement.length(ga)


def minimum_bounding_circle_ratio(collection):
    """
    The Reock compactness measure, defined by the ratio of areas between the
    minimum bounding/containing circle of a shape and the shape itself.

    Measure A1 in Altman (1998), cited for Frolov (1974), but earlier from Reock
    (1963)
    """
    ga = _cast(collection)
    mbc = pygeos.minimum_bounding_circle(ga)
    return pygeos.area(ga) / pygeos.area(mbc)


def radii_ratio(collection):
    """
    The Flaherty & Crumplin (1992) index, OS_3 in Altman (1998).

    The ratio of the radius of the equi-areal circle to the radius of the MBC
    """
    ga = _cast(collection)
    r_eac = numpy.sqrt(pygeos.area(ga) / numpy.pi)
    r_mbc = pygeos.minimum_bounding_radius(ga)
    return r_eac / r_mbc


def diameter_ratio(collection, rotated=True):
    """
    The Flaherty & Crumplin (1992) length-width measure, stated as measure LW_7
    in Altman (1998).

    It is given as the ratio between the minimum and maximum shape diameter.
    """
    ga = _cast(collection)
    if rotated:
        box = pygeos.minimum_rotated_rectangle(ga)
        coords = pygeos.get_coordinates(box)
        a, b, c, d = (coords[0::5], coords[1::5], coords[2::5], coords[3::5])
        widths = numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))
        heights = numpy.sqrt(numpy.sum((a - d) ** 2, axis=1))
    else:
        box = pygeos.bounds(ga)
        (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
        widths, heights = numpy.abs(xmax - xmin), numpy.abs(ymax - ymin)
    return numpy.minimum(widths, heights) / numpy.maximum(widths, heights)


def length_width_diff(collection):
    """
    The Eig & Seitzinger (1981) shape measure, defined as:

    L - W

    Where L is the maximal east-west extent and W is the maximal north-south
    extent.

    Defined as measure LW_5 in Altman (1998)
    """
    ga = _cast(collection)
    box = pygeos.bounds(ga)
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
    return pygeos.measurement.length(
        pygeos.convex_hull(ga)
    ) / pygeos.measurement.length(ga)


def convex_hull_ratio(collection):
    """
    ratio of the area of the convex hull to the area of the shape itself

    Altman's A_3 measure, from Neimi et al 1991.
    """
    ga = _cast(collection)
    return pygeos.area(ga) / pygeos.area(pygeos.convex_hull(ga))


def fractal_dimension(collection, support="hex"):
    """
    The fractal dimension of the boundary of a shape, assuming a given spatial support for the geometries.

    Note that this derivation assumes a specific ideal spatial support for the polygon, and is thus may not return valid results for complex or highly irregular geometries.
    """
    ga = _cast(collection)
    P = pygeos.measurement.length(ga)
    A = pygeos.area(ga)
    if support == "hex":
        return 2 * numpy.log(P / 6) / numpy.log(A / (3 * numpy.sin(numpy.pi / 3)))
    elif support == "square":
        return 2 * numpy.log(P / 4) / numpy.log(A)
    elif support == "circle":
        return 2 * numpy.log(P / (2 * numpy.pi)) / numpy.log(A / numpy.pi)
    else:
        raise ValueError(
            f"The support argument must be one of 'hex', 'circle', or 'square', but {support} was provided."
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
    return ((numpy.sqrt(pygeos.area(ga)) * 4) / pygeos.length(ga)) ** 2


def rectangularity(collection):
    """
    Ratio of the area of the shape to the area of its minimum bounding rotated rectangle

    Reveals a polygon’s degree of being curved inward.

    .. math::
        \\frac{A}{A_{MBR}}

    where :math:`A` is the area and :math:`A_{MBR}` is the area of minimum bounding
    rotated rectangle.

    Notes
    -----
    Implementation follows :cite:`basaraner2017`.
    """
    ga = _cast(collection)
    return pygeos.area(ga) / pygeos.area(pygeos.minimum_rotated_rectangle(ga))


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
    return numpy.sqrt(pygeos.area(ga) / numpy.pi) / pygeos.minimum_bounding_radius(ga)


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
    box = pygeos.minimum_rotated_rectangle(ga)
    return numpy.sqrt(pygeos.area(ga) / pygeos.area(box)) * (
        pygeos.length(box) / pygeos.length(ga)
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
    A = pygeos.area(ga)
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

    \sum_i d_{i,c}^2

    where c is the centroid of the polygon

    Altman's OS_1 measure, cited in Boyce and Clark (1964), also used in Weaver
    and Hess (1963).
    """
    ga = _cast(collection)
    coords = pygeos.get_coordinates(ga)
    geom_ixs = numpy.repeat(numpy.arange(len(ga)), pygeos.get_num_coordinates(ga))
    centroids = pygeos.get_coordinates(pygeos.centroid(ga))[geom_ixs]
    squared_euclidean = numpy.sum((coords - centroids) ** 2, axis=1)
    dists = (
        pandas.DataFrame.from_dict(dict(d2=squared_euclidean, geom_ix=geom_ixs))
        .groupby("geom_ix")
        .d2.sum()
    ).values
    return pygeos.area(ga) / numpy.sqrt(2 * dists)


def moa_ratio(collection):
    """
    Computes the ratio of the second moment of area (like Li et al (2013)) to
    the moment of area of a circle with the same area.
    """
    ga = _cast(collection)
    r = pygeos.measurement.length(ga) / (2 * numpy.pi)
    return (numpy.pi * 0.5 * r ** 4) / second_areal_moment(ga)


def nmi(collection):
    """
    Computes the Normalized Moment of Inertia from Li et al (2013), recognizing
    that it is the relationship between the area of a shape squared divided by
    its second moment of area.
    """
    ga = _cast(collection)
    return pygeos.area(ga) ** 2 / (2 * second_areal_moment(ga) * numpy.pi)


def second_areal_moment(collection):
    """
    Using equation listed on en.wikipedia.org/Second_Moment_of_area, the second
    moment of area is actually the cross-moment of area between the X and Y
    dimensions:

    I_xy = (1/24)\sum^{i=N}^{i=1} (x_iy_{i+1} + 2*x_iy_i + 2*x_{i+1}y_{i+1} +
    x_{i+1}y_i)(x_iy_i - x_{i+1}y_i)

    where x_i, y_i is the current point and x_{i+1}, y_{i+1} is the next point,
    and where x_{n+1} = x_1, y_{n+1} = 1.

    This relation is known as the:
    - second moment of area
    - moment of inertia of plane area
    - area moment of inertia
    - second area moment

    and is *not* the mass moment of inertia, a property of the distribution of
    mass around a shape.
    """
    ga = _cast(collection)
    result = numpy.zeros(len(ga))
    n_holes_per_geom = pygeos.get_num_interior_rings(ga)
    for i, geometry in enumerate(ga):
        n_holes = n_holes_per_geom[i]
        for hole_ix in range(n_holes):
            hole = pygeos.get_coordinates(pygeos.get_interior_ring(ga, hole_ix))
            result[i] -= _second_moa_ring(hole)
        n_parts = pygeos.get_num_geometries(geometry)
        for part in pygeos.get_parts(geometry):
            result[i] += _second_moa_ring(pygeos.get_coordinates(part))
    # must divide everything by 24 and flip if polygon is clockwise.
    signflip = numpy.array([-1, 1])[pygeos.is_ccw(ga).astype(int)]
    return result * (1 / 24) * signflip


@njit
def _second_moa_ring(points):
    """
    implementation of the moment of area for a single ring
    """
    moi = 0
    for i in prange(len(points[:-1])):
        x_tail, y_tail = points[i]
        x_head, y_head = points[i + 1]
        moi += (x_tail * y_head - x_head * y_tail) * (
            x_tail * y_head
            + 2 * x_tail * y_tail
            + 2 * x_head * y_head
            + x_head * y_tail
        )
    return moi


# -------------------- OTHER MEASURES -------------------- #


def reflexive_angle_ratio(collection):
    """
    The Taylor reflexive angle index, measure OS_4 in Altman (1998)

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
