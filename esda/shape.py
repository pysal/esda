import contextlib

import numpy as np
import pandas as pd
from packaging.version import Version

# gets handled at the _cast level.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import shapely


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
        import geopandas as gpd
        import shapely
    except (ImportError, ModuleNotFoundError) as exception:
        raise type(exception)(
            "shapely and geopandas are required for shape statistics."
        ) from None

    if Version(shapely.__version__) < Version("2"):
        raise ImportError("Shapely 2.0 or newer is required.")

    if isinstance(collection, gpd.GeoSeries | gpd.GeoDataFrame):
        return np.asarray(collection.geometry.array)
    elif isinstance(collection, np.ndarray | list):
        return np.asarray(collection)
    else:
        return np.array([collection])


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
    angles = np.asarray(_get_angles(coords, n_coords_per_geom))
    if return_indices:
        return angles, np.repeat(
            np.arange(len(ga)),
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
        angle = np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))
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
    collection : GeoSeries, GeoDataFrame, np.ndarray, list
        Input collection of polygons.

    Returns
    -------

    np.ndarray
        An array of the same length as the input collection, containing the 
        Isoperimetric quotient for each polygon in the collection.

    Notes
    -----

    Altman's :math:`PA_1` measure :cite:`altman1998Districting`.

    The formula is given by:

    .. math::
        IPQ = \\frac{4 \\pi A}{P^2}

    Where :math:`A` is the area of the polygon and :math:`P` is the perimeter of the polygon.

    The :math:`IPQ` is scale invariant and due to the inclusion of :math:`\\pi` in the formula, 
    it is bounded between 0 and 1, with 1 representing a perfect circle, the most compact shape
    by this measure.
    
    """

    ga = _cast(collection)
    return (4 * np.pi * shapely.area(ga)) / (shapely.measurement.length(ga) ** 2)


def isoareal_quotient(collection):
    """
    The Isoareal quotient, defined as the ratio of a polygon's perimeter to the
    perimeter of the equi-areal circle.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, np.ndarray, list
        Input collection of polygons.

    Returns
    -------

    np.ndarray
        An array of the same length as the input collection, containing the 
        Isoareal quotient for each polygon in the collection.
        
    Notes
    -----

    Altman's :math:`PA_3` measure :cite:`altman1998Districting`.

    The formula is given by:

    .. math::
        IAQ = \\frac{2 \\sqrt{\\pi A}}{P}

    Where :math:`A` is the area of the polygon and :math:`P` is the perimeter of the polygon.

    With some manipulation, :math:`IAQ` can also be expressed as the square root of the Isoperimetric quotient, given by

    .. math::
        IAQ = \\frac{2 \\sqrt{\\pi A}}{P}
            = \\sqrt{\\frac{(2 \\sqrt{\\pi A})^2}{P^2}}
            = \\sqrt{\\frac{4 \\pi A}{P^2}}
            = \\sqrt{IPQ}

    Therefore, `isoareal_quotient` is implemented as `np.sqrt(isoperimetric_quotient(collection))`. 
    Importantly, this means that the :math:`IAQ` and :math:`IPQ` will rank shapes identically.

    The :math:`IAQ` is scale invariant and due to the inclusion of :math:`\\pi` in the formula, 
    it is bounded between 0 and 1, with 1 representing a perfect circle, the most compact shape
    by this measure.
    
    """
    return np.sqrt(isoperimetric_quotient(collection))


def minimum_bounding_circle_ratio(collection):
    """
    The Reock compactness measure, defined by the ratio of areas between the
    minimum bounding/containing circle of a shape and the shape itself.

    Measure A1 in :cite:`altman1998Districting`,
    cited for Frolov (1974), but earlier from Reock
    (1963)
    """
    ga = _cast(collection)
    mbca = (shapely.minimum_bounding_radius(ga) ** 2) * np.pi
    return shapely.area(ga) / mbca


def radii_ratio(collection):
    """
    The Flaherty & Crumplin (1992) index, OS_3 in :cite:`altman1998Districting`.

    The ratio of the radius of the equi-areal circle to the radius of the MBC
    """
    ga = _cast(collection)
    r_eac = np.sqrt(shapely.area(ga) / np.pi)
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
        widths = np.sqrt(np.sum((a - b) ** 2, axis=1))
        heights = np.sqrt(np.sum((a - d) ** 2, axis=1))
    else:
        box = shapely.bounds(ga)
        (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
        widths, heights = np.abs(xmax - xmin), np.abs(ymax - ymin)
    return np.minimum(widths, heights) / np.maximum(widths, heights)


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
    width, height = np.abs(xmax - xmin), np.abs(ymax - ymin)
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
        return 2 * np.log(P / 6) / np.log(A / (3 * np.sin(np.pi / 3)))
    elif support == "square":
        return 2 * np.log(P / 4) / np.log(A)
    elif support == "circle":
        return 2 * np.log(P / (2 * np.pi)) / np.log(A / np.pi)
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
    return ((np.sqrt(shapely.area(ga)) * 4) / shapely.length(ga)) ** 2


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
    return np.sqrt(shapely.area(ga) / np.pi) / shapely.minimum_bounding_radius(ga)


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
    return np.sqrt(shapely.area(ga) / shapely.area(box)) * (
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
    res = np.zeros(len(ga))
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
    geom_ixs = np.repeat(np.arange(len(ga)), shapely.get_num_coordinates(ga))
    centroids = shapely.get_coordinates(shapely.centroid(ga))[geom_ixs]
    squared_euclidean = np.sum((coords - centroids) ** 2, axis=1)
    dists = (
        pd.DataFrame.from_dict(dict(d2=squared_euclidean, geom_ix=geom_ixs))
        .groupby("geom_ix")
        .d2.sum()
    ).values
    return shapely.area(ga) / np.sqrt(2 * dists)


def moa_ratio(collection):
    """
    Computes the ratio of the second moment of area (like Li et al (2013)) to
    the moment of area of a circle with the same area.
    """
    ga = _cast(collection)
    r = shapely.length(ga) / (2 * np.pi)
    return (np.pi * 0.5 * r**4) / second_areal_moment(ga)


def nmi(collection):
    """
    Computes the Normalized Moment of Inertia from Li et al (2013), recognizing
    that it is the relationship between the area of a shape squared divided by
    its second moment of area.
    """
    ga = _cast(collection)
    return shapely.area(ga) ** 2 / (2 * second_areal_moment(ga) * np.pi)


def second_areal_moment(collection):
    """
    Reimplemented by second_moment_of_area.
    """ 
    return second_moment_of_area(collection)

@njit
def _second_moment_of_area_ring(pts, ref_pt=None):
    """Calculate the second moment of area of a closed polygon using the shoelace formula.
    
    This computes the second moment of area (referred to in some disciplines as the polar 
    second moment) as the sum of I_x (second moment about x-axis) and I_y (second moment 
    about y-axis), measured from the reference point (the centroid by default).
    
    Parameters
    ----------
    pts : Iterable of tuples
        Coordinate pairs (x, y) defining a closed linear ring (polygon boundary).
        The last point should equal the first point.
    ref_pt : tuple, optional
        A point (x_ref, y_ref) to measure moment about. If None (default), moment is 
        measured about the centroid. To return moment about the origin, explicitly
        set to (0, 0).

    Returns
    -------
    float
        A float representing the total (I_x + I_y) second moment of area about 
        the reference point.

    Notes
    -----
    The second moment of area formulas using the shoelace method are:
    
    .. math::
        I_x = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(y_i^2 + y_i y_{i+1} + y_{i+1}^2)
        
        I_y = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(x_i^2 + x_i x_{i+1} + x_{i+1}^2)
    
    where indices wrap around (n+1 = 1).
    
    The moments are then adjusted to be relative to the reference point using the 
    parallel axis theorem.

    The shoelace formula returns a signed area based on the winding direction 
    of the polygon. By convention, counter-clockwise winding returns a positive area,
    while clockwise winding returns a negative area. Geospatial data, including shapely,
    typically uses clockwise winding for exterior rings and counter-clockwise winding for
    interior rings. **The return value should therefore be multiplied by -1 to obtain the 
    conventional positive second moment of area for exterior rings.** Adding exterior and 
    (negative) interior rings together (e.g., for polygons with holes) will yield the 
    correct total second moment of area.
    """
    
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    use_origin = ref_pt is not None and ref_pt == (0, 0)

    # Shoelace formula components
    A = 0  # Area (used for ref_pt/centroid calculation)
    Sx = 0  # First moment about origin (x component)
    Sy = 0  # First moment about origin (y component)
    Ix_origin = 0  # Second moment about x-axis through origin
    Iy_origin = 0  # Second moment about y-axis through origin
    
    # Iterate through consecutive coordinate pairs
    for i in prange(len(pts) - 1):
        cross = x[i] * y[i+1] - x[i+1] * y[i]  # Shoelace term

        # Calculating A and Sx, Sy is not needed if calculating about origin,
        # but there is no time savings in skipping it here.

        # Accumulate area
        A += cross

        # Accumulate first moments (for centroid calculation)
        Sx += (x[i] + x[i+1]) * cross
        Sy += (y[i] + y[i+1]) * cross
        
        # Accumulate second moments about origin
        Ix_origin += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2) * cross
        Iy_origin += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2) * cross
    
    # Normalize second moments
    Ix_origin = Ix_origin / 12
    Iy_origin = Iy_origin / 12

    if use_origin:
        Ix = Ix_origin
        Iy = Iy_origin
    else:
        # Normalize by area
        A = A / 2

        if ref_pt is not None:
            # Use provided reference point
            cx, cy = ref_pt    
        else:
            # Calculate centroid
            cx = Sx / (6 * A)
            cy = Sy / (6 * A)

        
        # Apply parallel axis theorem to shift moments to reference point
        Ix = Ix_origin - A * (cy ** 2)
        Iy = Iy_origin - A * (cx ** 2)

    return Ix + Iy

def _polygon_rings(polygon):
    exterior = np.asarray(polygon.exterior.coords)
    interiors = [np.asarray(ring.coords) for ring in polygon.interiors]
    return exterior, interiors

def _multipolygon_rings(multipolygon):
    exteriors = []
    interiors = []

    for poly in multipolygon.geoms:
        ext, holes = _polygon_rings(poly)
        exteriors.append(ext)
        interiors.extend(holes)

    return exteriors, interiors

def second_moment_of_area(collection):
    """
    New implementation of second moment of area using the shoelace formula.
    """

    ga = _cast(collection)
    moments = []

    for geom in ga:
        total_moa = 0

        # shapely geometries wind "backwards" compared with the mathematical convention;
        # thus, we need to invert the sign of the result to get the expected positive value
        if isinstance(geom, shapely.geometry.Polygon):
            exterior, interiors = _polygon_rings(geom)
            total_moa -= _second_moment_of_area_ring(exterior)
            for interior in interiors:
                total_moa += _second_moment_of_area_ring(interior)
        elif isinstance(geom, shapely.geometry.MultiPolygon):
            exteriors, interiors = _multipolygon_rings(geom)
            for exterior in exteriors:
                total_moa -= _second_moment_of_area_ring(exterior)
            for interior in interiors:
                total_moa += _second_moment_of_area_ring(interior)
        else:
            raise ValueError(
                f"Geometry type {geom.geom_type} not supported. Only Polygon and MultiPolygon are supported."
            )
        moments.append(total_moa)

    return np.array(moments)

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
        pd.DataFrame.from_dict(dict(is_reflex=angles < 0, geom_ix=geom_indices))
        .groupby("geom_ix")
        .is_reflex.mean()
        .values
    )
