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


def external_entropy(a, b, balance=0):
    """
    The harmonic mean summarizing the overlay entropy of two
    sets of polygons: a onto b and b onto a.

    Arguments
    ----------
    a : geometry array of polygons
        array of polygons
    b : geometry array of polygons
        array of polygons
    balance  float
        weight that describing the relative importance of pattern a or pattern b.
        When large and positive, we weight the measure more to ensure polygons in b
        are fully contained by polygons in a. When large and negative,
        we weight the pattern to ensure polygons in A are fully contained
        by polygons in b. Corresponds to the log of beta in Nowosad and Stepinksi (2018).

    Returns
    --------
    (n,) array expressing the entropy of the areal distributions
    of a's splits by partition b.

    """
    beta = numpy.exp(balance)
    aix, bix, ab = _overlay(a, b, return_indices=True)
    a_areas = pygeos.area(a)
    b_areas = pygeos.area(b)
    ab_areas = pygeos.area(ab)
    b_onto_a = _overlay_entropy(aix, a_areas, ab_areas)
    b_onto_a /= areal_entropy(areas=a_areas, partial=False)
    a_onto_b = _overlay_entropy(bix, b_areas, ab_areas)
    a_onto_b /= areal_entropy(areas=b_areas, partial=False)

    c = 1 - numpy.average(b_onto_a, weights=a_areas)
    h = 1 - numpy.average(a_onto_b, weights=b_areas)

    return (1 + beta) * h * c / ((beta * h) + c)


def overlay_entropy(a, b, standardize=True, partial=False):
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
    h = _overlay_entropy(aix, a_areas, pygeos.area(ab))
    if standardize:
        h /= areal_entropy(None, areas=a_areas)
    return h.sum()


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


def areal_entropy(polygons=None, areas=None, partial=False):
    """
    Compute the entropy of the distribution of polygon areas.


    Arguments
    ---------
    polygons: numpy array of geometries
        polygons whose distribution of entropies needs to be computed.
        Should not be provided if areas is provided.
    areas: numpy array
        areas to use to compute entropy. SHould not be provided if
        polygons are provided.
    partial: bool (default: False)
        whether to return the total entropy of the areal distribution
        (False), or to return the contribution to entropy made
        by each of area (True).

    Returns
    -------
    Total map entropy or (n,) vector of partial entropies.
    """
    assert not (
        (polygons is None) & (areas is None)
    ), "either polygons or precomputed areas must be provided"
    if polygons is None:
        assert areas is not None, "If polygons are not provided, areas should be."
    if areas is None:
        assert polygons is not None, "If areas are not provided, polygons should be."
        areas = pygeos.area(polygons)
    result = entr(areas / areas.sum())
    if partial:
        return result
    return result.sum()


if __name__ == "__main__":
    import geopandas

    r1 = geopandas.read_file("./regions.gpkg", layer="regions1")
    r2 = geopandas.read_file("./regions.gpkg", layer="regions2")
    r1a = pygeos.from_shapely(r1.geometry)
    r2a = pygeos.from_shapely(r2.geometry)
    r1areas = pygeos.area(r1a)
    r2areas = pygeos.area(r1a)
    aix, bix, r1r2 = _overlay(r1a, r2a, return_indices=True)
    r1r2areas = pygeos.area(r1r2)
    crosstab = pandas.DataFrame(dict(aix=aix, bix=bix, area=r1r2areas)).pivot(
        index="aix", columns="bix", values="area"
    )

    print(external_entropy(r1a, r2a))
    print(overlay_entropy(r1a, r2a, standardize=True, partial=False))
    print(overlay_entropy(r2a, r1a, standardize=True, partial=False))
