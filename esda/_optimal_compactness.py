"""Calculate optimal compactness of an observed polygon."""

import math
import multiprocessing as mp

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution as de
from shapely import Point, Polygon

# a polygon must have at least 3 sides, and we cap it at 32 to prevent it from
# approximating a circle as n -> inf
MIN_N_POLYGON = 3
MAX_N_POLYGON = 32

# hardcode Shapely's default approximation of a circle for reproducibility
CIRCLE_QUAD_SEGS = 16

# how many workers for the DE optimizer to use
cpus = mp.cpu_count()


def _objective_f_circle(args: npt.NDArray[np.float64], poly: Polygon) -> float:
    """
    Evaluate circle non-fit for an observed polygon.

    This objective function evaluates delta_c(x, y, r), the ratio of the
    symmetric-difference area of polygon P and candidate comparison circle C
    to their union area. Minimizing this non-fit objective identifies the
    optimal comparison circle C* used to calculate kappa_c* = 1 - delta_c.

    Parameters
    ----------
    args
        Circle decision variables ordered as center x, center y, and radius r.
    poly
        Observed polygon P to compare with the candidate circle.

    Returns
    -------
    float
        Non-fit score delta_c in [0, 1], where lower values indicate better
        geometric fit.
    """
    x, y, r = args
    circle = Point(x, y).buffer(r, quad_segs=CIRCLE_QUAD_SEGS)
    return float(poly.symmetric_difference(circle).area / poly.union(circle).area)


def _xyr_bounds(poly: Polygon) -> list[tuple[float, float]]:
    """
    Construct shared bounds for center coordinates and size.

    The feasible set constrains x and y to the polygon's bounding box and
    constrains r to be positive and no larger than half the bounding-box
    diagonal. The size parameter r is a circle radius for comparison circles
    and a circumradius for regular comparison polygons.

    Parameters
    ----------
    poly
        Observed polygon P used to define the constrained search space.

    Returns
    -------
    list[tuple[float, float]]
        Lower and upper bounds for x, y, and r.
    """
    minx, miny, maxx, maxy = poly.bounds

    # the center is constrained to the observed polygon's bounding box, while
    # r can grow to half the box diagonal
    maxr = math.hypot(maxx - minx, maxy - miny) / 2
    return [(minx, maxx), (miny, maxy), (1e-6, maxr)]


def _vertex_count(poly: Polygon) -> int:
    """
    Count vertices across a polygon's exterior and interior boundaries.

    Parameters
    ----------
    poly
        Polygon whose boundary vertices should be counted.

    Returns
    -------
    int
        Total number of non-closing vertices in the exterior ring and all
        interior rings.
    """
    return (len(poly.exterior.coords) - 1) + sum(
        len(ring.coords) - 1 for ring in poly.interiors
    )


def _optimal_circle_compactness(poly: Polygon) -> OptimizeResult:
    """
    Optimize the comparison circle for an observed polygon.

    Differential evolution searches over x, y, and r to minimize the circle
    non-fit objective delta_c. The returned objective value is delta_c*, not
    kappa_c*; the corresponding compactness value is 1 - result.fun.

    Parameters
    ----------
    poly
        Observed polygon P whose optimal comparison circle C* should be found.

    Returns
    -------
    OptimizeResult
        Differential evolution result with x = [x*, y*, r*] and fun = delta_c*.
    """
    bounds = _xyr_bounds(poly)

    # differential evolution searches the three continuous circle parameters:
    # center x, center y, and radius
    return de(
        _objective_f_circle,
        bounds,
        args=(poly,),
        popsize=30,
        tol=1e-6,
        seed=0,
        workers=cpus,
    )  # type: ignore[arg-type]


def _objective_f_ngon(args: npt.NDArray[np.float64], poly: Polygon, n: int) -> float:
    """
    Evaluate fixed-n regular-polygon non-fit for an observed polygon.

    This objective function evaluates delta_q_n(x, y, r, a), the ratio of the
    symmetric-difference area of polygon P and candidate regular n-gon Q_n to
    their union area. Here n is fixed, so only the center, circumradius, and
    rotation angle are decision variables.

    Parameters
    ----------
    args
        Decision variables ordered as center x, center y, circumradius r, and
        rotation angle a in degrees.
    poly
        Observed polygon P to compare with the candidate regular n-gon.
    n
        Fixed number of sides in the candidate regular polygon Q_n.

    Returns
    -------
    float
        Non-fit score delta_q_n in [0, 1], where lower values indicate better
        geometric fit.
    """
    x, y, r, a = args

    # fixed-n searches use angle in degrees directly because the symmetry
    # interval is known when bounds are constructed
    other_poly = _regular_polygon(n, x, y, r, a)
    return float(
        poly.symmetric_difference(other_poly).area / poly.union(other_poly).area,
    )


def _objective_f_polygon(args: npt.NDArray[np.float64], poly: Polygon) -> float:
    """
    Evaluate variable-n regular-polygon non-fit for an observed polygon.

    This objective function evaluates delta_q(x, y, r, a, n), the non-fit
    between polygon P and a candidate regular polygon Q. The number of sides n
    is an integer decision variable. Internally, rotation is represented as a
    normalized fraction of the candidate polygon's rotational symmetry interval
    so that each n searches one comparable orientation range.

    Parameters
    ----------
    args
        Decision variables ordered as center x, center y, circumradius r,
        normalized rotation fraction, and number of sides n.
    poly
        Observed polygon P to compare with the candidate regular polygon Q.

    Returns
    -------
    float
        Non-fit score delta_q in [0, 1], where lower values indicate better
        geometric fit.
    """
    x, y, r, rotation_frac, n = args

    # normalize rotation to [0, 1] so each n searches one symmetry interval
    a = rotation_frac * 360 / n
    other_poly = _regular_polygon(n, x, y, r, a)
    return float(
        poly.symmetric_difference(other_poly).area / poly.union(other_poly).area,
    )


def _optimal_polygon_compactness(poly: Polygon, n: int | None) -> OptimizeResult:
    """
    Optimize a regular comparison polygon for an observed polygon.

    If n is fixed, differential evolution minimizes delta_q_n over x, y, r,
    and a. If n is None, it minimizes delta_q over x, y, r, a, and integer n.
    The returned objective value is the optimal non-fit score, not compactness;
    the corresponding compactness value is 1 - result.fun.

    Parameters
    ----------
    poly
        Observed polygon P whose optimal regular comparison polygon should be
        found.
    n
        Fixed number of sides for Q_n. If None, n is treated as an integer
        decision variable and the optimizer searches for Q*.

    Returns
    -------
    OptimizeResult
        Differential evolution result. For fixed n, x = [x*, y*, r*, a*].
        For variable n, x = [x*, y*, r*, a*, n*] after converting the internal
        rotation fraction back to angle degrees. In both cases, fun is the
        minimized non-fit score.
    """
    xyr_bounds = _xyr_bounds(poly)

    if n is not None:
        # a regular n-gon repeats every 360 / n degrees, so the optimizer only
        # needs to search one rotational symmetry interval
        angle_bounds = [(1e-6, 360 / n)]
        bounds = xyr_bounds + angle_bounds
        return de(
            _objective_f_ngon,
            bounds,  # type: ignore[arg-type]
            args=(poly, n),
            popsize=30,
            tol=1e-6,
            seed=0,
            workers=cpus,
        )

    # a regular comparison polygon should not have more sides than the
    # observed polygon has boundary vertices, capped for tractability
    max_n = max(MIN_N_POLYGON, min(_vertex_count(poly), MAX_N_POLYGON))

    # when n varies, optimize a normalized rotation fraction instead of degrees
    # so every candidate n gets exactly one symmetry interval
    rotation_frac_bounds = [(1e-6, 1)]
    n_bounds = [(MIN_N_POLYGON, max_n)]
    bounds = xyr_bounds + rotation_frac_bounds + n_bounds  # type: ignore[operator]

    # the n decision variable is discrete. SciPy rounds integral dimensions
    # before objective evaluation and keeps them fixed during polishing
    result = de(
        _objective_f_polygon,
        bounds,  # type: ignore[arg-type]
        args=(poly,),
        popsize=30,
        tol=1e-6,
        seed=0,
        workers=cpus,
        integrality=[False, False, False, False, True],
    )
    # return the public parameterization as [x, y, r, angle_degrees, n]
    result.x[3] = result.x[3] * 360 / result.x[4]
    return result


def _regular_polygon(n: int, x: float, y: float, r: float, a: float) -> Polygon:
    """
    Construct a regular n-sided polygon from center, radius, and rotation.

    The polygon has n equally spaced vertices on a circumcircle of radius r,
    centered at x, y, then rotated by angle a. This parameterization matches
    the comparison polygon Q_{x,y,r,a,n} used in the optimization objectives.

    Parameters
    ----------
    n
        Number of sides in the regular polygon. Must be at least 3.
    x
        X coordinate of the polygon center.
    y
        Y coordinate of the polygon center.
    r
        Circumradius of the polygon.
    a
        Rotation angle in degrees.

    Returns
    -------
    Polygon
        Regular polygon Q_{x,y,r,a,n}.
    """
    n = int(n)
    if n < MIN_N_POLYGON:
        msg = "A regular polygon must have at least 3 sides."
        raise ValueError(msg)

    # vertices lie on the circumcircle at equal angular spacing, then rotate by
    # a degrees around the requested center
    angles = np.deg2rad(a + np.arange(n) * 360 / n)
    xs = x + r * np.cos(angles)
    ys = y + r * np.sin(angles)
    return Polygon(np.column_stack([xs, ys]))


def _optimal_compactness(
    poly: Polygon,
    circle: bool = True,
    n: int | None = None,
    return_result: bool = False,
) -> float | tuple[float, OptimizeResult]:  # noqa: FBT001,FBT002
    """
    Calculate optimal compactness of an observed polygon.

    This function finds either an optimal comparison circle C* or an optimal
    regular comparison polygon Q*. The minimized objective is the non-fit,
    delta: the area of the symmetric difference divided by the area of the
    union. Compactness is kappa = 1 - delta, so callers should transform
    result.fun if they need compactness rather than minimized non-fit.

    Parameters
    ----------
    poly
        Observed polygon P whose comparison shape should be optimized.
    circle
        If True, optimize a comparison circle C*. If False, optimize a
        comparison regular polygon.
    n
        If circle is False, fixed number of sides for Q_n. If None, n is
        optimized as an integer decision variable between 3 and
        min(vertex_count, 32).
    return_result
        If True, also return the optimal compactness and the scipy
        OptimizeResult object itself. Otherwise only return the optimal
        compactness.

    Returns
    -------
    compactness
        The optimal compactness of poly. If return_result is True, also return
        the scipy OptimizeResult object itself as well.
    """
    if circle:
        result = _optimal_circle_compactness(poly)
    else:
        result = _optimal_polygon_compactness(poly, n)

    compactness = 1 - result.fun
    if return_result:
        return compactness, result
    return compactness
