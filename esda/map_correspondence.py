import numpy, pygeos, pandas

# from nowosad and stepinski
# https://doi.org/10.1080/13658816.2018.1511794


def v(a, b, beta=1):
    tree = pygeos.STRtree(a)
    bix, aix = tree.query_bulk(b)
    a_areas = pygeos.area(a)
    b_areas = pygeos.area(b)
    ab_areas = pygeos.area(pygeos.intersection(a[aix], b[bix]))
    sjr = _sj(aix, bix, a_areas, b_areas, ab_areas)
    sr = _s(None, areas=a_areas)
    siz = _sj(bix, aix, b_areas, a_areas, ab_areas)
    sz = _s(None, areas=b_areas)

    h = 1 - numpy.sum((b_areas / b_areas.sum()) * (sjr / sr))
    c = 1 - numpy.sum((a_areas / a_areas.sum()) * (siz / sz))

    return (1 + beta) * h * c / ((beta * h) + c)


def sj(aix, bix, a_areas, b_areas, ab_areas):
    mapping = pandas.DataFrame.from_dict(
        dict(
            a=aix,
            b=bix,
            area=ab_areas,
            a_area=a_areas[aix],
            b_areas=b_areas[bix],
        )
    )
    output = -(
        mapping.eval("frac = area/b_areas")
        .eval("entropy = frac * log(frac)")
        .groupby("b")
        .entropy.sum()
    ).values
    output[output < 0] = 0
    return output


def s(a, areas=None):
    if areas is None:
        areas = pygeos.area(a)
    return -numpy.sum(areas / areas.sum() * numpy.log(areas / areas.sum()))
