import contextlib

import numpy as np
import pandas as pd
from packaging.version import Version

# gets handled at the _cast level.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import shapely
    from shapely.geometry import Polygon, MultiPolygon
    import geopandas as gpd

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


def moment_of_inertia(collection, normalize=False, ref_pt=None, 
                      region_col=None, region_ids=None, wt_col=None, wts=None):
    """
    Calculates the moment of inertia of each geometry in the collection. Can handle
    polygons with holes and multipolygons by summing the moments of inertia of each part.
    The moment of inertia is calculated about the centroid of each geometry by default,
    but can also be calculated about a specified reference point.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, np.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns the normalized moment of inertia as the ratio of the 
        moment of inertia of a circle of the same area to the moment of inertia
        of the shape. Default is False.
    ref_pt : array-like of shape (2,) or (n, 2), optional
        Spatial coordinate(s) of reference point about which the moment of inertia is calculated.
        May be provided as a single coordinate pair ``(x, y)``, or as an
        array-like of multiple coordinate pairs with shape
        ``(n, 2)``.  To return moment about the origin, explicitly set to (0, 0).
        If ``None``, the default behavior is to calculate the moment about the centroid of 
        each geometry.
    region_col : str, optional
        The name of the column in the GeoDataFrame to use for region IDs.
    region_ids : array-like, optional
        An iterable of region IDs of the same length as `collection` that each geometry 
        is assigned to. (What is behavior if both `region_col` and `region_ids` are provided?)
    wt_col : str, optional
        The name of the column in the GeoDataFrame to use for weights, such as
        a population column, to calculate the mass moment of inertia. Weights should be numeric.
        Requires `region_col` or `region_ids` to be provided.
    wts : array-like, optional
        An iterable of weights, such as populations, of the same length as `collection`,
        to calculate the mass moment of inertia. (What is the behavior if both `wt_col` 
        and `wts` are provided?) Requires `region_col` or `region_ids` to be provided.

    Returns
    -------
    array-like
        An array of moment of inertia values for each geometry in the collection.

    Notes
    -----
    The moment of inertia is the variance of the distance of all points (possibly weighted)
    in a shape to a reference point, which can be interpreted as an axis of rotation 
    perpendicular to the plane of the shape. The first parameter should be a GeoDataFrame or 
    array-like of Polygons or MultiPolygons. These geometries may be the shapes for which 
    the moment of inertia is calculated, or they may be subregions within larger regions
    defined by `region_col` or `region_ids`. If `region_col` or `region_ids` are provided,
    the moment of inertia is calculated for each region as a whole. If normalization is
    requested, the moment of inertia is compared with that of a circle of equal area. This
    is discussed in detail below.
    
    When region IDs are not provided or the subregions are unweighted, this is known as the 
    area moment of inertia or the second moment of area, and can be quickly calculated from 
    the polygon vertices using the shoelace formula. The second moments of area about the 
    x and y axes are calculated as:
    
    .. math::
        I_x = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(y_i^2 + y_i y_{i+1} + y_{i+1}^2)
        
        I_y = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(x_i^2 + x_i x_{i+1} + x_{i+1}^2)
    
    where indices wrap around (n+1 = 1).

    The moments are then adjusted to be relative to a reference point using the 
    parallel axis theorem. The moment of inertia is then the sum of :math:`I_x` and 
    :math:`I_y`. In geographic contexts, the reference point is typically the 
    centroid, which by default will be calculated and used if no reference point is provided.
    However, other points of interest may be used, such as a capital city or the residence of
    a political representative. 

    When region IDs and weights are provided, this calculates the mass moment of inertia,
    which weights each point in the shape by a value such as population. This is useful
    for calculating the moment of inertia of a region with varying population density. For 
    continuous regions, this is equivalent to integrating the squared distance of each point
    weighted by a density function over the area of the shape:
     
    .. math::
        I = \\int r^2 \\, \\mathrm{d}m

    where :math:`r` is the distance from the reference point and :math:`\\mathrm{d}m` is
    the mass element at each point. This implementation uses the discrete approximation,
    summing over subregions with known weights, such as census tracts with known populations:

    .. math::
        I = \\sum_{i} m_i r_i^2

    In this implementation, weights are applied as uniform densities
    across each subregion geometry, scaled by the area of each geometry.

    The moment of inertia can be normalized to provide a compactness measure by comparing
    the moment of inertia of the shape to that of a circle of equal area. This is calculated
    as the ratio of the moment of inertia of the circle to that of the shape about its centroid:

    .. math::
        C_{MI} = \\frac{I_{circle}}{I_{shape}}
               = \\frac{A^2}{2 \\pi I_{shape}}

    where :math:`A` is the area of the polygon and :math:`I_{shape}` is the moment of inertia
    of the polygon. This formulation is from Li, et al (2013). The area-based (non-weighted) 
    :math:`C_{MI}` value is bounded between 0 and 1, with 1 representing a perfect circle, 
    the most compact shape by this measure. Although not discussed in Li, et al (2013),
    this formulation can also be applied to the mass moment of inertia when weights are provided,
    yielding a population-weighted compactness measure. In this case, the measure can exceed 1, 
    for a population distribution more compact than a uniform circle.
    
    References
    ----------
    .. [1] Weaver, James B., and Sidney W. Hess. 1963. "A Procedure for Nonpartisan 
        Districting: Development of Computer Techniques." Yale Law Journal 73 (2): 288–308. 
        https://doi.org/10.2307/794769
    .. [2] Godwin, A. N. 1980. "Simple Calculation of Moments of Inertia for Polygons."
        International Journal of Mathematical Education in Science and Technology 11 (4): 577–86. 
        https://doi.org/10.1080/0020739800110414.
    .. [3] Li, Wenwen, Michael F. Goodchild, and Richard Church. 2013. "An Efficient Measure 
        of Compactness for Two-Dimensional Shapes and Its Application in Regionalization 
        Problems." International Journal of Geographical Information Science 27 (6): 1227–50. 
        https://doi.org/10.1080/13658816.2012.752093.
    .. [4] https://en.wikipedia.org/wiki/Second_moment_of_area#List_of_second_moments_of_area

    """

    # Must handle three cases:
    # 1. Unweighted area moment of inertia (second moment of area)
    # 2. Unweighted mass moment of inertia with region IDs
    # 3. Weighted mass moment of inertia with region IDs

    if region_col is None and region_ids is None:
        # Unweighted area moment of inertia
        return second_moment_of_area(collection, normalize=normalize, ref_pt=ref_pt)
    else:
        # Weighted mass moment of inertia
        return mass_moment_of_inertia(collection, region_col=region_col, region_ids=region_ids,
                                        wt_col=wt_col, wts=wts)
    
def mass_moment_of_inertia(collection, normalize=False, ref_pt=None, region_col=None, region_ids=None, wt_col=None, wts=None):
    """
    Calculates the mass moment of inertia for regions defined by assignment of
    geometries in the collection.
    
    Requires either a column name with region assignments (`region_col`), or an array-like of
    region IDs. Weights can be provided as a column name (`wt_col`) or an array-like. If 
    `region_col` or `wt_col` is provided, `collection` must be a GeoDataFrame. If weights
    are not provided, geometries weighted by area, which is equivalent 
    to the second moment of area.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, np.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns the normalized moment of inertia as the ratio of the 
        moment of inertia of a reference circle to the moment of inertia
        of the shape, where the reference circle has the same area and the same
        population uniformly distributed over that area. Default is False.
    ref_pt : array-like of shape (2,) or (n, 2), optional
        Spatial coordinate(s) of reference point about which the moment of inertia is calculated.
        May be provided as a single coordinate pair ``(x, y)``, or as an
        array-like of multiple coordinate pairs with shape
        ``(n, 2)``.  To return moment about the origin, explicitly set to (0, 0).
        If ``None``, the default behavior is to calculate the moment about the centroid of 
        each region.
    region_col : str, optional
        The name of the column in the GeoDataFrame to use for region IDs.
    region_ids : array-like, optional
        An iterable of region IDs of the same length as `collection` that each geometry 
        is assigned to. If both `region_col` and `region_ids` are provided, `region_ids` 
        is ignored.
    wt_col : str, optional
        The name of the column in the GeoDataFrame to use for weights, such as
        a population column, to calculate the mass moment of inertia. Weights should be numeric.
        Requires `region_col` or `region_ids` to be provided.
    wts : array-like, optional
        An iterable of weights, such as populations, of the same length as `collection`,
        to calculate the mass moment of inertia. If both `wt_col` and `wts` are provided,
        `wts` is ignored. Requires `region_col` or `region_ids` to be provided.

    Returns
    -------
    pandas.Series
        A Series of moment of inertia values for each regions, labelled by (sorted) region IDs
        given by either `region_col` or `region_ids`.

    Notes
    -----
    The moment of inertia is the variance of the distance of all areas (weighted)
    in a shape to a reference point, which can be interpreted as an axis of rotation 
    perpendicular to the plane of the shape. The first parameter should be a GeoDataFrame or 
    array-like of Polygons or MultiPolygons. These geometries in this collection are 
    subregions within larger regions defined by `region_col` or `region_ids`. 
    The moment of inertia is calculated for each region as a whole. If weights are not
    provided, this is equivalent to the second moment of area. If normalization is
    requested, the moment of inertia is compared with that of a circle of equal area, and
    equal population distributed uniformly over that area. This
    is discussed in detail below.
    
    The mass moment of inertia weights each area in the shape by a value such as population. 
    This is useful for calculating the moment of inertia of a region with varying population 
    density. For continuous regions, this is equivalent to integrating the squared distance of each point
    weighted by a density function over the area of the shape:
     
    .. math::
        I = \\int r^2 \\, \\mathrm{d}m

    where :math:`r` is the distance from the reference point and :math:`\\mathrm{d}m` is
    the mass element at each point. This implementation uses the discrete approximation,
    summing over subregions with known weights, such as census tracts with known populations:

    .. math::
        I = \\sum_{i} m_i r_i^2

    In this implementation, weights are assumed to be massed at the centroid of each subregion,
    which is equivalent to assuming uniform density across each subregion geometry, scaled by 
    the area of each geometry.

    The moment of inertia can be normalized to provide a compactness measure by comparing
    the moment of inertia of the shape to that of a circle of equal area  (Li, et al. 2013 
    :cite:`LiEtAl2013`). This is calculated as the ratio of the moment of inertia of the 
    circle to that of the shape about its centroid:

    .. math::
        C_{MI} = \\frac{I_{circle}}{I_{shape}}
               = \\frac{A^2}{2 \\pi I_{shape}}

    where :math:`A` is the area of the polygon and :math:`I_{shape}` is the moment of inertia
    of the polygon. As an area-based (non-weighted) measure, the 
    :math:`C_{MI}` value is bounded between 0 and 1. Although not discussed in Li, et al (2013),
    this formulation can also be applied to the mass moment of inertia when weights are provided,
    yielding a population-weighted compactness measure.
    
    .. math::
        C_{MI} = \\frac{M A^2}{2 \\pi I_{shape}}
    
    where :math:`M` is the mass (population) of the shape. In this case, the measure can exceed 1, 
    as will happen when mass (population) is concentrated near the centroid. This can be 
    interpreted as a more compact distribution than a uniform circle.
    
    References
    ----------
    .. [1] Weaver, James B., and Sidney W. Hess. 1963. "A Procedure for Nonpartisan 
        Districting: Development of Computer Techniques." Yale Law Journal 73 (2): 288–308. 
        https://doi.org/10.2307/794769
    .. [2] Li, Wenwen, Michael F. Goodchild, and Richard Church. 2013. "An Efficient Measure 
        of Compactness for Two-Dimensional Shapes and Its Application in Regionalization 
        Problems." International Journal of Geographical Information Science 27 (6): 1227–50. 
        https://doi.org/10.1080/13658816.2012.752093.

    """

    ga = _cast(collection)

    # Handle region IDs. If provided, use directly. Otherwise, extract from GeoDataFrame.
    if region_col is not None:

        if not isinstance(collection, gpd.GeoDataFrame):
            raise ValueError(
                "If `region_col` is provided, `collection` must be a GeoDataFrame."
            )
        region_ids = collection[region_col].values

    # Handle weights (masses). If provided, use directly. Otherwise, extract from GeoDataFrame.
    # If neither is provided, weight by geom area.
    if wt_col is not None:

        if not isinstance(collection, gpd.GeoDataFrame):
            raise ValueError(
                "If `wt_col` is provided, `collection` must be a GeoDataFrame."
            )
        wts = collection[wt_col].values
    elif wts is not None:
        wts = np.asarray(wts)
    else:
        wts = np.array(shapely.area(ga)) # Uniform weights equal to area of each sub-geometry

    unique_regions = np.sort(np.unique(region_ids))
    region_mois = []

    for i, region in enumerate(unique_regions):

        # Centroids and areas of geoms are calculated internally and discarded 
        # by _geometric_moments_ring(). However, calculating them here using shapely
        # is easier to read and understand, and is also 25% faster.

        mask = region_ids == region
        geoms = ga[mask]
        m = wts[mask] # Weight (mass, population, etc.) of each sub-geometry

        # Get centroids as an array of coordinate tuples
        c = np.array([(pt.x, pt.y) for pt in shapely.centroid(geoms)])

        # Moment of inertia about centroid of each sub-geometry
        I_geom = second_moment_of_area(geoms)

        # Area and centroid of region
        A = np.sum(shapely.area(geoms)) # 2X faster than shapely.area(shapely.union_all(geoms))
        C = np.sum(m[:, None] * c, axis=0) / m.sum()

        # Distance squared, don't actually need distance, so don't bother taking square root
        d2 = np.sum((c - C)**2, axis=1)

        I = np.sum(I_geom + m * d2)

        if normalize:
            region_mois.append((m.sum() * A ** 2) / (2 * np.pi * I))
        else:
            region_mois.append(I)

    return pd.Series(region_mois, index=unique_regions)

def moa_ratio(collection, region_col=None, region_ids=None, wt_col=None, wts=None):
    """
    Alias for `nmi`. Computes the Normalized Moment of Inertia.
    """

    # Create deprecation warning
    return nmi(collection, region_col=region_col, region_ids=region_ids,
               wt_col=wt_col, wts=wts)


def nmi(collection, region_col=None, region_ids=None, wt_col=None, wts=None):
    """
    Computes the Normalized Moment of Inertia from Li et al (2013) as the ratio of the 
    moment of inertia of a circle of the same area about its center, to the moment of 
    inertia of the shape about its centroid.

    Notes
    -----
    The formula is given by:

    .. math::
        NMI = \\frac{A^2}{2 \\pi I}
    
    Where :math:`A` is the area of the polygon and :math:`I` is the moment of inertia 
    of the polygon.
    """
    
    return moment_of_inertia(collection, normalize=True, region_col=region_col,
                                 region_ids=region_ids, wt_col=wt_col, wts=wts)


def second_areal_moment(collection):
    """
    Reimplemented by `second_moment_of_area`. Calculates only non-normalized second moment of area
    about the centroid. Use `second_moment_of_area` for more options.
    """ 
    return second_moment_of_area(collection)

def _second_moment_of_area_ring(pts, ref_pt=None):
    """Calculate the second moment of area of a closed polygon using the shoelace formula.
    
    This is a wrapper function that calls `_geometric_moments_ring(pts, ref_pt) and 
    returns only the second moment of area about the reference point (:math:`I_xx + I_yy`)

    Parameters
    ----------
    pts : Iterable of tuples
        Coordinate pairs (x, y) defining a closed linear ring (polygon boundary).
        The last point should equal the first point.
    ref_pt : tuple, optional
        A point (x_ref, y_ref) to measure second moment of area about. If None (default), 
        moment is measured about the centroid. To return moment about the origin, explicitly
        set to (0, 0).

    Returns
    -------
    float
        A float representing the total (:math:`I_x + I_y`) second moment of area about 
        the reference point.

    Notes
    -----
    See `second_moment_of_area` for details.
    """
    A, cx, cy, Ixx, Iyy = _geometric_moments_ring(pts, ref_pt=ref_pt)
    return Ixx + Iyy

@njit
def __geometric_moments_ring(pts, ref_pt=None):
    """Calculate geometric moments of a closed polygon using the shoelace formula.
    
    Returns area, centroid coordinates, and second moments of area about x- and y-axis.

    Parameters
    ----------
    pts : Iterable of tuples
        Coordinate pairs (x, y) defining a closed linear ring (polygon boundary).
        The last point should equal the first point.
    ref_pt : tuple, optional
        A point (x_ref, y_ref) to measure second moment of area about. If None (default), 
        moment is measured about the centroid. To return moment about the origin, explicitly
        set to (0, 0).

    Returns
    -------
    tuple
        A tuple containing:
        - A float :math:`A` representing the area (zeroth moment) of the polygon.
        - Two floats :math:`c_x`, :math:`c_y` representing the centroid coordinates (first moment divided by area).
        - Two floats :math:`I_xx`, :math:`I_yy` representing the second moments of area about         A float representing the total (:math:`I_x + I_y`) second moment of area about 
        the reference point.

    Notes
    -----
    The shoelace formula returns a signed area based on the winding direction 
    of the polygon. By convention, counter-clockwise winding returns a positive area,
    while clockwise winding returns a negative area. Geospatial data, including shapely,
    typically uses clockwise winding for exterior rings and counter-clockwise winding for
    interior rings. This can be handled by the caller in two ways:

    1. Use `shapely.orient_polygons()` to change coordinate direction to use the 
    mathematical convention.
    2. Multiply return values for :math:`A`, :math:`I_xx`, and :math:`I_yy` (elements 0, 
    3, and 4 of the returned tuple) by -1 to get the correctly signed area and second 
    moments.
    
    This will yield positive areas and second moments for exterior rings and negative
    second moments of area for interior rings. The rings can then be added together using
    :math:`I_xx + A d^2` and :math:`I_yy + A d^2` where :math:`d` is the distance between 
    the centroid of the (exterior or interior) ring and the centroid of the entire shape. 
    Adding the final :math:`I_xx` and :math:`I_yy` will yield the correct total second moment
    of area for the shape.
    """
    
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    use_origin = ref_pt == (0, 0)

    # print("use_origin = ", use_origin)
    
    # Shoelace formula components
    A = 0  # Area (used for ref_pt/centroid calculation)
    Sx = 0  # First moment about origin (x component)
    Sy = 0  # First moment about origin (y component)
    Ixx_origin = 0  # Second moment about x-axis through origin
    Iyy_origin = 0  # Second moment about y-axis through origin
    
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
        Ixx_origin += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2) * cross
        Iyy_origin += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2) * cross
    
    Ixx_origin = Ixx_origin / 12
    Iyy_origin = Iyy_origin / 12

    # Calculate the area (zeroth moment)
    A = A / 2

    # Calculate centroid (first moment divided by area)
    cx = Sx / (6 * A)
    cy = Sy / (6 * A)

    if use_origin:
        Ixx = Ixx_origin
        Iyy = Iyy_origin
    else:

        if ref_pt is None:
            # Set reference point to centroid
            ref_x, ref_y = cx, cy
        else:
            ref_x, ref_y = ref_pt
        
        
        # Apply parallel axis theorem to shift moments to reference point
        Ixx = Ixx_origin - A * (ref_y ** 2)
        Iyy = Iyy_origin - A * (ref_x ** 2)

    return A, ref_x, ref_y, Ixx, Iyy

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

def _second_moment_of_area(collection, normalize=False, ref_pt=None):
    """
    Compute the second moment of area (moment of inertia) of each geometry in the 
    collection using the shoelace formula.

    Can handle polygons with holes and multipolygons by summing the moments of inertia 
    of each part. The moment of inertia is calculated about the centroid of each geometry 
    by default, but can also be calculated about a specified reference point.

    Parameters
    ----------
    collection : GeoSeries, GeoDataFrame, np.ndarray, list
        Input collection of polygons or multipolygons.
    normalize : bool, optional
        If True, returns the normalized moment of inertia as the ratio of the 
        moment of inertia of a circle of the same area to the moment of inertia
        of the shape. Default is False.
    ref_pt : array-like of shape (2,) or (n, 2), optional
        Spatial coordinate(s) of reference point about which the moment of inertia is calculated.
        May be provided as a single coordinate pair ``(x, y)``, or as an
        array-like of multiple coordinate pairs with shape
        ``(n, 2)``.  To return moment about the origin, explicitly set to (0, 0).
        If ``None``, the default behavior is to calculate the moment about the centroid of 
        each geometry.

    Returns
    -------
    array-like
        An array of moment of inertia values for each geometry in the collection.

    Notes
    -----
    The second moment of area, also known as the area moment of inertia, or, for 
    brevity, the moment of inertia, is the variance of the distance of all points in a shape
    to a reference point, which can be interpreted as an axis of rotation 
    perpendicular to the plane of the shape.
    
    The moment of inertia can be quickly calculated from 
    the polygon vertices using the shoelace formula. The second moments of area about the 
    x and y axes are calculated as:
    
    .. math::
        I_x = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(y_i^2 + y_i y_{i+1} + y_{i+1}^2)
        
        I_y = \\frac{1}{12}\\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1}y_i)(x_i^2 + x_i x_{i+1} + x_{i+1}^2)
    
    where indices wrap around (n+1 = 1).

    The moments are then adjusted to be relative to a reference point using the 
    parallel axis theorem. The moment of inertia is then the sum of :math:`I_x` and 
    :math:`I_y`. In geographic contexts, the reference point is typically the 
    centroid, which by default will be calculated and used if no reference point is provided.
    However, other points of interest may be used, such as a capital city or the residence of
    a political representative. 

    The moment of inertia can be normalized to provide a compactness measure by comparing
    the moment of inertia of the shape to that of a circle of equal area (Li, et al. 2013 
    :cite:`LiEtAl2013`). This is calculated as the ratio of the moment of inertia of the 
    circle about its center to that of the shape about its centroid:

    .. math::
        C_{MI} = \\frac{I_{circle}}{I_{shape}}
               = \\frac{A^2}{2 \\pi I_{shape}}

    where :math:`A` is the area of the polygon and :math:`I_{shape}` is the moment of inertia
    of the polygon. The value is bounded between 0 and 1, with 1 representing a perfect circle, 
    the most compact shape by this measure.
    
    References
    ----------
    .. [1] Godwin, A. N. 1980. "Simple Calculation of Moments of Inertia for Polygons."
        International Journal of Mathematical Education in Science and Technology 11 (4): 577–86. 
        https://doi.org/10.1080/0020739800110414.
    .. [2] Li, Wenwen, Michael F. Goodchild, and Richard Church. 2013. "An Efficient Measure 
        of Compactness for Two-Dimensional Shapes and Its Application in Regionalization 
        Problems." International Journal of Geographical Information Science 27 (6): 1227–50. 
        https://doi.org/10.1080/13658816.2012.752093.
    .. [3] https://en.wikipedia.org/wiki/Second_moment_of_area#List_of_second_moments_of_area

    """

    if normalize and ref_pt is not None:
        raise ValueError("Normalization is only supported when ref_pt is None (centroid).")
    
    ga = _cast(collection)
    moments = []

    # shapely geometries wind "backwards" compared with the mathematical convention;
    # Use orient_polygons to make the wind counterclockwise
    for geom in shapely.orient_polygons(ga):

        Cx, Cy = geom.centroid.coords[0]
        
        Ixx = 0.0
        Iyy = 0.0

        if isinstance(geom, shapely.geometry.Polygon):
            exterior, interiors = _polygon_rings(geom)
            A, cx, cy, Ixx_c, Iyy_c = _geometric_moments_ring(exterior, ref_pt=ref_pt)

            dx = cx - Cx
            dy = cy - Cy

            Ixx += Ixx_c + A * dy**2
            Iyy += Iyy_c + A * dx**2

            for interior in interiors:
                A, cx, cy, Ixx_c, Iyy_c = _geometric_moments_ring(interior, ref_pt=ref_pt)

                dx = cx - Cx
                dy = cy - Cy

                Ixx += Ixx_c + A * dy**2
                Iyy += Iyy_c + A * dx**2

        elif isinstance(geom, shapely.geometry.MultiPolygon):
            exteriors, interiors = _multipolygon_rings(geom)
            for exterior in exteriors:
                A, cx, cy, Ixx_c, Iyy_c = _geometric_moments_ring(exterior, ref_pt=ref_pt)

                dx = cx - Cx
                dy = cy - Cy

                Ixx += Ixx_c + A * dy**2
                Iyy += Iyy_c + A * dx**2

            for interior in interiors:
                A, cx, cy, Ixx_c, Iyy_c = _geometric_moments_ring(interior, ref_pt=ref_pt)

                dx = cx - Cx
                dy = cy - Cy

                Ixx += Ixx_c + A * dy**2
                Iyy += Iyy_c + A * dx**2

        else:
            raise ValueError(
                f"Geometry type {geom.geom_type} not supported. Only Polygon and MultiPolygon are supported."
            )
        
        if normalize:
            moments.append((geom.area ** 2) / (2 * np.pi * (Ixx + Iyy)))
        else:
            moments.append(Ixx + Iyy)

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

###########################################################
# New method begings here
# Reposition in module once debugged
###########################################################

# -------------------------
# Helper Functions
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
    np.ndarray
        Array of shape (N, 2) containing coordinates of a ring.
    """
    for poly in shapely.get_parts(geoms):
        yield np.asarray(poly.exterior.coords)
        for interior in poly.interiors:
            yield np.asarray(interior.coords)


def _geometric_moments_ring(pts, shift_to_centroid=True):
    """
    Compute area, centroid, and second moments of a single polygon ring.

    Parameters
    ----------
    pts : np.ndarray
        Array of coordinates defining the ring, shape (N, 2).
    shift_to_centroid : bool, default True
        If True, apply the parallel axis theorem to shift moments to the ring centroid.

    Returns
    -------
    A : float
        Signed area of the ring.
    cx, cy : float
        Coordinates of the ring centroid.
    Ixx, Iyy : float
        Second moments of area about the centroid (or origin if shift_to_centroid=False).

    Notes
    -----
    - Polar moment of area J can be obtained by summing Ixx + Iyy at the calling level.
    - This function does not compute J itself, leaving derived quantities to higher-level functions.
    """
    x = pts[:, 0]
    y = pts[:, 1]

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]

    A = cross.sum() / 2
    Sx = ((x[:-1] + x[1:]) * cross).sum()
    Sy = ((y[:-1] + y[1:]) * cross).sum()

    Ixx_origin = ((y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2) * cross).sum() / 12
    Iyy_origin = ((x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2) * cross).sum() / 12

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

        # first moments
        A_tot += A
        Mx_tot += A * cx
        My_tot += A * cy

        # shift ring inertia back to origin
        Ixx0_tot += Ixx_c + A * cy**2
        Iyy0_tot += Iyy_c + A * cx**2

    # centroid of entire collection
    Cx = Mx_tot / A_tot
    Cy = My_tot / A_tot

    # shift to centroid
    Ixx = Ixx0_tot - A_tot * Cy**2
    Iyy = Iyy0_tot - A_tot * Cx**2

    J = Ixx + Iyy

    return A_tot, Cx, Cy, Ixx, Iyy, J

# -------------------------
# Main Functions
# -------------------------

def moment_of_inertia(collection, normalize=False, ref_pt=None):
    """
    Compute moment of inertia (second moment of area) per geometry.

    Parameters
    ----------
    collection : sequence of shapely geometries
        Polygons or multipolygons.
    normalize : bool, default False
        If True, returns moment normalized by reference disk of same area.
    ref_pt : tuple or None, optional
        If provided, shifts the moment to be with respect to this point.

    Returns
    -------
    np.ndarray
        Array of moment of inertia values per geometry.
    """
    ga = _cast(collection)
    ga = shapely.orient_polygons(ga)

    Js = []
    for geom in ga:
        A, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid([geom])
        if ref_pt is not None:
            dx = Cx - ref_pt[0]
            dy = Cy - ref_pt[1]
            J += A * (dx**2 + dy**2)
        if normalize:
            J = A**2 / (2 * np.pi * J)
        Js.append(J)

    return np.asarray(Js)


second_moment_of_area = moment_of_inertia  # alias for users familiar with engineering terminology


def moment_of_inertia_global(collection, normalize=False, ref_pt=None):
    """
    Compute moment of inertia for an entire collection of geometries combined.

    Parameters
    ----------
    collection : sequence of shapely geometries
        Polygons or multipolygons.
    normalize : bool, default False
        If True, returns moment normalized by reference disk of same area.
    ref_pt : tuple or None, optional
        If provided, shifts the moment to be with respect to this point.

    Returns
    -------
    float
        Moment of inertia for the entire collection.
    """
    ga = _cast(collection)
    ga = shapely.orient_polygons(ga)

    A, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid(ga)

    if ref_pt is not None:
        dx = Cx - ref_pt[0]
        dy = Cy - ref_pt[1]
        J += A * (dx**2 + dy**2)

    if normalize:
        J = A**2 / (2 * np.pi * J)

    return J


def moment_of_inertia_regions(collection, regions, weights=None, normalize=False):
    """
    Compute population- or area-weighted moment of inertia per region.

    Parameters
    ----------
    collection : sequence of shapely geometries
        Polygons or multipolygons.
    regions : array-like
        Region identifier for each geometry.
    weights : array-like or None, optional
        Weight per geometry (e.g., population). If None, defaults to polygon area.
    normalize : bool, default False
        Normalize each region's moment of inertia by reference disk.

    Returns
    -------
    pd.Series
        Indexed by unique region IDs, containing moment of inertia per region.
    """
    ga = _cast(collection)
    ga = shapely.orient_polygons(ga)

    regions = np.asarray(regions)
    unique_regions = np.unique(regions)

    if weights is None:
        # default to area
        weights = np.array([geom.area for geom in ga])
    else:
        weights = np.asarray(weights)

    Js = []

    for reg in unique_regions:
        mask = regions == reg
        sub_geoms = ga[mask]
        sub_weights = weights[mask]

        A_tot, Cx, Cy, Ixx, Iyy, J = _moments_about_centroid(sub_geoms)

        # apply population weighting
        if not np.allclose(sub_weights, 1):
            J = J * sub_weights.sum() / sub_weights.size

        if normalize:
            J = sub_weights.sum() * A_tot / (2 * np.pi * J)

        Js.append(J)

    return pd.Series(Js, index=unique_regions)
