import numpy
import pandas
from scipy.special import entr
from packaging.version import Version

try:
    import shapely
except (ImportError, ModuleNotFoundError):
    pass  # gets handled in the _cast function.

# from nowosad and stepinski
# https://doi.org/10.1080/13658816.2018.1511794


def _overlay(a, b, return_indices=False):
    """
    Compute geometries from overlaying a onto b
    """
    tree = shapely.STRtree(a)
    bix, aix = tree.query(b)
    overlay = shapely.intersection(a[aix], b[bix])
    if return_indices:
        return aix, bix, overlay
    return overlay


def _cast(collection):
    """
    Cast a collection to a shapely geometry array.
    """
    try:
        import geopandas
    except (ImportError, ModuleNotFoundError) as exception:
        raise type(exception)(
            "shapely and geopandas are required for map comparison statistics."
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


def external_entropy(a, b, balance=0, base=numpy.e):
    """
    The harmonic mean summarizing the overlay entropy of two
    sets of polygons: a onto b and b onto a.

    Called the v-measure in :cite:`nowosad2018`

    Parameters
    ----------
    a : geometry array of polygons
        array of polygons
    b : geometry array of polygons
        array of polygons
    balance:  float
        weight that describing the relative importance of pattern a or pattern b.
        When large and positive, we weight the measure more to ensure polygons in b
        are fully contained by polygons in a. When large and negative,
        we weight the pattern to ensure polygons in A are fully contained
        by polygons in b. Corresponds to the log of beta in {cite}`Nowosad2018`.
    base: float
        base of logarithm to use throughout computation
    Returns
    --------
    (n,) array expressing the entropy of the areal distributions
    of a's splits by partition b.

    Example
    -------

    >>> r1 = geopandas.read_file('tests/regions.zip', layer='regions1')
    >>> r2 = geopandas.read_file('tests/regions.zip', layer='regions2')
    >>> external_entropy(r1, r2)
    """
    a = _cast(a)
    b = _cast(b)
    beta = numpy.exp(balance)
    aix, bix, ab = _overlay(a, b, return_indices=True)
    a_areas = shapely.area(a)
    b_areas = shapely.area(b)
    ab_areas = shapely.area(ab)
    b_onto_a = _overlay_entropy(aix, a_areas, ab_areas, base=base)  # SjZ
    # SZ, as sabre has entropy.empirical(rowSums(xtab), unit='log2')
    b_onto_a /= areal_entropy(areas=b_areas, local=False, base=base)
    a_onto_b = _overlay_entropy(bix, b_areas, ab_areas, base=base)  # SjR
    # SR, as sabre has entropy.empirical(colSums(xtab), unit='log2')
    a_onto_b /= areal_entropy(areas=a_areas, local=False, base=base)

    c = 1 - numpy.average(b_onto_a, weights=a_areas)
    h = 1 - numpy.average(a_onto_b, weights=b_areas)

    return (1 + beta) * h * c / ((beta * h) + c)


def completeness(a, b, local=False, base=numpy.e):
    """
    The completeness of the partitions of polygons in a to those in a.
    Closer to 1 when all polygons in a are fully contained within polygons in b.
    From :cite:`nowosad2018`

    Parameters
    ----------
    a : geometry array of polygons
        array of polygons
    b : geometry array of polygons
        array of polygons
    local: bool (default: False)
        whether or not to provide local scores for each polygon. If True, the
        completeness for polygons in a are returned.
    scale: bool (default: None)
        whether to scale the completeness score(s). By default, completeness is
        is scaled for local scores so that the average of the local scores is
        the overall map completeness. If not local, then completeness is returned
        unscaled. You can also set local=True and scale=False to get raw components
        of the completeness, whose sum is the completeness for the entire map.
        Global re-scaled scores (local=False & scale=True) are not supported.
    base: bool (default=None)
        what base to use for the entropy calculations. The default is base e,
        which means entropy is measured in "nats."

    Example
    -------

    >>> r1 = geopandas.read_file('tests/regions.zip', layer='regions1')
    >>> r2 = geopandas.read_file('tests/regions.zip', layer='regions2')
    >>> completeness(r1, r2)
    """
    a = _cast(a)
    b = _cast(b)
    ohi = overlay_entropy(a, b, standardize=True, local=True, base=base)
    a_areas = shapely.area(a)
    w = a_areas / a_areas.sum()
    ci = (w * (1 - ohi)) / w.sum()
    if local:
        return ci
    return ci.sum()


def homogeneity(a, b, local=False, base=numpy.e):
    """
    The homogeneity of polygons from a partitioned by b.

    From :cite:`nowosad2018`

    This is equal to completeness(b,a).

    It is closer to 1 when all polygons in b correspond well to polygons in a.

    Parameters
    ----------
    a : geometry array of polygons
        array of polygons
    b : geometry array of polygons
        array of polygons
    local: bool (default: False)
        whether or not to provide local scores for each polygon. If True, the
        homogeneity for polygons in b are returned.
    scale: bool (default: None)
        whether to scale the completeness score(s). By default, completeness is
        is scaled for local scores so that the average of the local scores is
        the overall map completeness. If not local, then completeness is returned
        unscaled. You can also set local=True and scale=False to get raw components
        of the completeness, whose sum is the completeness for the entire map.
        Global re-scaled scores (local=False & scale=True) are not supported.
    base: bool (default=None)
        what base to use for the entropy calculations. The default is base e,
        which means entropy is measured in "nats."

    Example
    -------

    >>> r1 = geopandas.read_file('tests/regions.zip', layer='regions1')
    >>> r2 = geopandas.read_file('tests/regions.zip', layer='regions2')
    >>> homogeneity(r1, r2)
    """
    return completeness(b, a, local=local, base=base)


def overlay_entropy(a, b, standardize=True, local=False, base=numpy.e):
    """
    The entropy of how n zones in a are split by m partitions in b,
    where n is the number of polygons in a and m is the number
    of partitions in b. This is the "overlay entropy", since
    the set of polygons constructed from intersection(a,b) is often
    called the "overlay" of A onto B.


    Larger when zones in a are uniformly split into many even pieces
    by partitions in b, and small when zones in A correspond well to
    zones in B.

    Parameters
    -----------
    a : geometry array of polygons
        a set of polygons (the "target") for whom the areal entropy is calculated
    b : geometry array of polygons
        a set of polygons (the "frame") that splits a

    Returns
    --------
    (n,) array expressing the entropy of the areal distributions
    of a's splits by partition b.

    Example
    -------

    >>> r1 = geopandas.read_file('tests/regions.zip', layer='regions1')
    >>> r2 = geopandas.read_file('tests/regions.zip', layer='regions2')
    >>> overlay_entropy(r1, r2)
    """
    a = _cast(a)
    b = _cast(b)
    aix, bix, ab = _overlay(a, b, return_indices=True)
    a_areas = shapely.area(a)
    b_areas = shapely.area(b)
    h = _overlay_entropy(aix, a_areas, shapely.area(ab), base=base)
    if standardize:
        h /= areal_entropy(None, areas=b_areas, local=False, base=base)
    if local:
        return h
    return h.sum()


def _overlay_entropy(aix, a_areas, ab_areas, base):
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
    mapping["entropy"] = entr(mapping.frac.values) / numpy.log(base)
    result = mapping.groupby("a").entropy.sum().values
    result[result < 0] = 0
    return result


def areal_entropy(polygons=None, areas=None, local=False, base=numpy.e):
    """
    Compute the entropy of the distribution of polygon areas.

    Parameters
    ----------
    polygons: numpy array of geometries
        polygons whose distribution of entropies needs to be computed.
        Should not be provided if areas is provided.
    areas: numpy array
        areas to use to compute entropy. SHould not be provided if
        polygons are provided.
    local: bool (default: False)
        whether to return the total entropy of the areal distribution
        (False), or to return the contribution to entropy made
        by each of area (True).

    Returns
    -------
    Total map entropy or (n,) vector of local entropies.

    Example
    -------

    >>> r1 = geopandas.read_file('tests/regions.zip', layer='regions1')
    >>> r2 = geopandas.read_file('tests/regions.zip', layer='regions2')
    >>> areal_entropy(polygons=r1)
    """
    assert not (
        (polygons is None) & (areas is None)
    ), "Either polygons or precomputed areas must be provided."
    assert not (
        (polygons is not None) & (areas is not None)
    ), "Only one of polygons or areas should be provided."
    if polygons is None:
        assert areas is not None, "If polygons are not provided, areas should be."
    if areas is None:
        assert polygons is not None, "If areas are not provided, polygons should be."
        polygons = _cast(polygons)
        areas = shapely.area(polygons)
    result = entr(areas / areas.sum()) / numpy.log(base)
    result[result < 0] = 0
    if local:
        return result
    return result.sum()
