import pygeos
import geopandas, pandas
import numpy
from numba import njit

# -------------------- UTILITIES --------------------#
def _cast(collection, regularize=False):
    """
    Cast a collection to a pygeos geometry array.
    """
    if isinstance(collection, geopandas.GeoSeries):
        return collection.values.data
    elif isinstance(collection, numpy.ndarray):
        if pygeos.is_geometry(collection).all():
            return collection.values.data
        else:
            raise NotImplementedError()
    return pygeos.from_shapely(collection)


def get_angles(ga, return_indices=False):
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
    exploded = pygeos.get_parts(ga)
    coords = pygeos.get_coordinates(exploded)
    n_coords_per_geom = pygeos.get_num_coordinates(exploded)
    n_parts_per_geom = pygeos.get_num_geometries(exploded)
    angles = numpy.asarray(_get_angles(coords, n_coords_per_geom, n_parts_per_geom))
    if return_indices:
        return angles, numpy.repeat(
            numpy.arange(len(ga)),
            pygeos.get_num_coordinates(ga) - pygeos.get_num_geometries(ga),
        )
    else:
        return angles


@njit
def _get_angles(points, n_coords_per_geom, n_parts_per_geom):
    """
    Iterate over points in a set of geometries.
    This assumes that the input geometries are simple, not multi!
    """
    offset = int(0)
    start = points[0]
    on_geom = 0
    on_coord = 0
    result = []
    n_points = len(points)
    while True:
        if on_coord == (n_coords_per_geom[on_geom] - 1):
            offset += on_coord
            on_geom += 1
            on_coord = 0
            if on_geom == len(n_coords_per_geom):
                break
            else:
                continue
        left = points[offset + on_coord % (n_coords_per_geom[on_geom] - 1)]
        center = points[offset + (on_coord + 1) % (n_coords_per_geom[on_geom] - 1)]
        right = points[offset + (on_coord + 2) % (n_coords_per_geom[on_geom] - 1)]
        a = left - center
        b = right - center
        angle = numpy.math.atan2(a[0] * b[1] - a[1] * b[0], numpy.dot(a, b))
        result.append(angle)
        on_coord += 1
    return result


# -------------------- INTERTIAL MEASURES -------------------- #


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
    return (4 * numpy.pi * pygeos.area(collection)) / (
        pygeos.measurement.length(collection) ** 2
    )


def isoareal_quotient(collection):
    """
    The Isoareal quotient, defined as the ratio of a polygon's perimeter to the
    perimeter of the equi-areal circle

    Altman's PA_3 measure, and proportional to the PA_4 measure
    """
    return (
        2 * numpy.pi * numpy.sqrt(pygeos.area(collection) / numpy.pi)
    ) / pygeos.measurement.length(collection)


def minimum_bounding_circle_ratio(collection):
    """
    The Reock compactness measure, defined by the ratio of areas between the
    minimum bounding/containing circle of a shape and the shape itself.

    Measure A1 in Altman (1998), cited for Frolov (1974), but earlier from Reock
    (1963)
    """
    mbc = pygeos.minimum_bounding_circle(collection)
    return pygeos.area(collection) / pygeos.area(mbc)


def radii_ratio(collection):
    """
    The Flaherty & Crumplin (1992) index, OS_3 in Altman (1998).

    The ratio of the radius of the equi-areal circle to the radius of the MBC
    """
    r_eac = numpy.sqrt(pygeos.area(collection) / numpy.pi)
    r_mbc = pygeos.minimum_bounding_radius(pointset)
    return r_eac / r_mbc


def diameter_ratio(collection, rotated=True):
    """
    The Flaherty & Crumplin (1992) length-width measure, stated as measure LW_7
    in Altman (1998).

    It is given as the ratio between the minimum and maximum shape diameter.
    """
    if rotated:
        box = pygeos.minimum_rotated_rectangle(collection)
        a, b, c, d, _ = pygeos.get_coordinates(box)
        width = numpy.sqrt(numpy.sum((a - b) ** 2))
        height = numpy.sqrt(numpy.sum((a - d) ** 2))
    else:
        box = pygeos.bounds(collection)
        (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
        width, height = numpy.abs(xmax - xmin), numpy.abs(ymax - ymin)
    return numpy.minimum(width, height) / numpy.maximum(width, height)


def length_width_diff(collection):
    """
    The Eig & Seitzinger (1981) shape measure, defined as:

    L - W

    Where L is the maximal east-west extent and W is the maximal north-south
    extent.

    Defined as measure LW_5 in Altman (1998)
    """
    box = pygeos.bounds(collection)
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
    return pygeos.measurement.length(
        pygeos.convex_hull(collection)
    ) / pygeos.measurement.length(collection)


def convex_hull_ratio(collection):
    """
    ratio of the area of the convex hull to the area of the shape itself

    Altman's A_3 measure, from Neimi et al 1991.
    """
    return pygeos.area(collection) / pygeos.area(pygeos.convex_hull(collection))


# -------------------- INTERTIAL MEASURES -------------------- #


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
    coords = pygeos.get_coordinates(collection)
    geom_ixs = numpy.tile(
        numpy.arange(len(collection)), pygeos.get_num_coordinates(collection)
    )
    centroids = pygeos.get_coordinates(pygeos.centroid(collection))[geom_ixs]
    squared_euclidean = numpy.sum((coords - centroids) ** 2, axis=1)
    dists = (
        pandas.DataFrame.from_dict(dict(d2=squared_euclidean, geom_ix=geom_ixs))
        .groupby("geom_ix")
        .d2.sum()
    ).values
    return pygeos.area(collection) / numpy.sqrt(2 * dists)


def moa_ratio(collection):
    """
    Computes the ratio of the second moment of area (like Li et al (2013)) to
    the moment of area of a circle with the same area.
    """
    raise NotImplementedError()


def nmi(collection):
    """
    Computes the Normalized Moment of Inertia from Li et al (2013), recognizing
    that it is the relationship between the area of a shape squared divided by
    its second moment of area.
    """
    raise NotImplementedError()


def second_moment_of_area(collection):
    """
    Using equation listed on en.wikipedia.org/Second_Moment_of_area, the second
    moment of area is actually the cross-moment of area between the X and Y
    dimensions:

    I_xy = (1/24)\sum^{i=N}^{i=1} (x_iy_{i+1} + 2*x_iy_i + 2*x_iy_{i+1} +
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
    raise NotImplementedError()


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
    )


def fractal_dimension(collection):
    """"""
    raise NotImplementedError()


if __name__ == "__main__":
    df = geopandas.read_file(geopandas.datasets.get_path("nybb"))
    ga = df.geometry.values.data
    coords = pygeos.get_coordinates(ga)
