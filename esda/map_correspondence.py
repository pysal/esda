import numpy, pygeos, pandas
from scipy.special import entr

# from nowosad and stepinski
# https://doi.org/10.1080/13658816.2018.1511794


def _overlay(a, b, return_indices=False):
    """
    Compute geometries from overlaying a onto b
    """
    tree = pygeos.STRtree(a)
    bix, aix = tree.query_bulk(b)
    overlay = pygeos.intersection(a[aix], b[bix])
    if return_indices:
        return aix, bix, overlay
    return overlay


def v_measure(a, b, beta=1):
    """
    conditional entropy of partitions a and b. 
    """ 
    aix, bix, ab = _overlay(a, b, return_indices=True)
    a_areas = pygeos.area(a)
    b_areas = pygeos.area(b)
    ab_areas = pygeos.area(ab)
    b_onto_a = _overlay_entropy(aix, a_areas, ab_areas, standardize=True)
    a_onto_b  _overlay_entropy(bix, b_areas, ab_areas, standardize=True)

    h = 1 - numpy.average(b_onto_a, weights=b_areas)
    c = 1 - numpy.average(a_onto_b, weights=a_areas)

    return (1 + beta) * h * c / ((beta * h) + c)


def overlay_entropy(a, b, standardize=True):
    """
    The entropy of how n zones in a are split by m partitions in b,
    where n is the number of polygons in a and m is the number
    of partitions in b. This is the "overlay entropy", since
    the set of polygons constructed from intersection(a,b) is often
    called the "overlay" of A onto B.

    Larger when zones in a are uniformly split into many even pieces
    by partitions in b, and small when zones in A correspond well to
    zones in B.

    Arguments
    ----------
    a : geometry array of polygons
        a set of polygons (the "target") for whom the areal entropy is calculated
    b : geometry array of polygons
        a set of polygons (the "frame") that splits a

    Returns
    --------
    (n,) array expressing the entropy of the areal distributions
    of a's splits by partition b.
    """
    aix, bix, ab = _overlay(a, b, return_indices=True)
    a_areas = pygeos.area(a)
    h = _overlay_entropy(aix, , pygeos.area(ab))
    if standardize:
        h /= areal_entropy(areas=a_areas)
    return h


def _overlay_entropy(aix, a_areas, ab_areas):
    """
    direct function to compute overlay entropies
    """
    mapping = pandas.DataFrame.from_dict(
        dict(
            a=aix,
            area=ab_areas,
            a_area=a_areas[aix],
        )
    )
    mapping["frac"] = mapping.area / mapping.a_area
    mapping["entropy"] = entr(mapping.frac.values)
    return (mapping.groupby("a").entropy.sum()).values


def areal_entropy(polygons=None, areas=None):
    """
    Compute the entropy of the distribution of polygon areas.
    """
    assert not (
        (polygons is None) & (areas is None)
    ), "either polygons or precomputed areas must be provided"
    if polygons is None:
        assert (areas is not None), "If polygons are not provided, areas should be."
    if areas is None:
        assert (polygons is not None), "If areas are not provided, polygons should be."
        areas = pygeos.area(polygons=None)
    return entr(areas / areas.sum())
=
