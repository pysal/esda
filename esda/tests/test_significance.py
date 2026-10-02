import numpy
import pytest
from libpysal.weights import Voronoi

import esda
from esda.significance import calculate_significance

numpy.random.seed(2478879)
coordinates = numpy.random.random(size=(800, 2))
x = numpy.random.normal(size=(800,))
w = Voronoi(coordinates, clip="bounding_box", use_index=False)
w.transform = "r"

with pytest.WARN_ALT_HYPOTHESIS_DEPR:
    stat = esda.Moran_Local(x, w, permutations=19)


@pytest.mark.parametrize(
    "alternative", ["two-sided", "directed", "lesser", "greater", "folded"]
)
def test_execution_and_range(alternative):
    out = calculate_significance(stat.Is, stat.rlisas, alternative=alternative)
    assert (out > 0).all() & (out <= 1).all(), (
        f"p-value out of bounds for method {alternative}"
    )
    if alternative == "directed":
        assert out.max() <= 0.5, f"max p-value is too large for method {alternative}"
    else:
        assert out.max() >= 0.5, f"max p-value is too small for method {alternative}"


def test_alternative_relationships():
    two_sided = calculate_significance(stat.Is, stat.rlisas, alternative="two-sided")
    directed = calculate_significance(stat.Is, stat.rlisas, alternative="directed")
    lesser = calculate_significance(stat.Is, stat.rlisas, alternative="lesser")
    greater = calculate_significance(stat.Is, stat.rlisas, alternative="greater")
    folded = calculate_significance(stat.Is, stat.rlisas, alternative="folded")

    numpy.testing.assert_allclose(
        lesser + greater,
        numpy.ones_like(lesser) + (1 / (stat.permutations + 1)),
        err_msg="greater p-value should be complement of lesser",
    )
    assert (directed <= two_sided).all(), (
        "directed is bigger than two_sided and should not be"
    )
    one_or_the_other = (directed == lesser) | (directed == greater)
    assert one_or_the_other.all(), (
        "some directed p-value is neither the greater nor lesser p-value"
    )
    assert (two_sided < folded).mean() < (directed < folded).mean(), (
        "Directed p-values should tend to be much "
        "smaller than two_sided p-values or folded p-values."
    )


def test_two_sided_degenerate_null():
    # GH #504: a constant reference distribution makes the percentile-based
    # two-sided p-value degenerate. With the test statistic equal to the
    # constant, every permutation lands in both tails, so the old percentile
    # count was 2 * p_permutations and produced 1.95. The clipped pseudo
    # p-value is exactly one here.
    reference = numpy.full((1, 19), 5.0)
    degenerate = calculate_significance(5.0, reference, alternative="two-sided")
    numpy.testing.assert_allclose(degenerate, 1.0)

    reference = numpy.full((3, 19), 2.0)
    degenerate = calculate_significance(
        numpy.full(3, 2.0), reference, alternative="two-sided"
    )
    numpy.testing.assert_allclose(degenerate, numpy.ones(3))


def test_two_sided_degenerate_null_off_constant():
    # The second failure mode of the old percentile formula: a constant
    # reference distribution with the test statistic away from the constant.
    # The lower percentile collapsed to the constant, so both tail counts
    # again covered the whole distribution and the p-value exceeded one. The
    # pseudo p-value here counts only the side the constant falls on:
    # greater = 19, lesser = 0, so 2 * (0 + 1) / 20 = 0.1.
    reference = numpy.full((1, 19), 5.0)
    p_value = calculate_significance(3.0, reference, alternative="two-sided")
    numpy.testing.assert_allclose(p_value, 0.1)


def test_two_sided_non_degenerate_null():
    numpy.random.seed(2478879)
    reference = numpy.random.normal(size=(1, 999))
    # greater = 7, lesser = 992, so 2 * (7 + 1) / 1000 = 0.016.
    p_value = calculate_significance(2.5, reference, alternative="two-sided")
    numpy.testing.assert_allclose(p_value, 0.016)


def test_two_sided_is_twice_directed_one_sided():
    # On a clearly one-sided statistic (well into the upper tail) the directed
    # p-value picks the smaller tail, so the two-sided value is exactly twice
    # the directed value as long as the clip at one does not bind.
    numpy.random.seed(2478879)
    reference = numpy.random.normal(size=(1, 999))
    two_sided = calculate_significance(2.5, reference, alternative="two-sided")
    directed = calculate_significance(2.5, reference, alternative="directed")
    numpy.testing.assert_allclose(two_sided, 2 * directed)


def test_two_sided_counts_ties_in_both_tails():
    # A discrete reference distribution ties with the observed statistic, and
    # both tails count those ties. The percentile formula this replaced swept
    # the whole tied block into both tails at once, which is what moved the
    # local join count p-values.
    reference = numpy.array([[0] * 2 + [1] * 4 + [2] * 3 + [3]], dtype=float)
    # greater = 4 (the three 2s and the 3), lesser = 9 (everything but the 3),
    # so the answer comes from the upper tail: 2 * (4 + 1) / 11.
    p_value = calculate_significance(2.0, reference, alternative="two-sided")
    numpy.testing.assert_allclose(p_value, 10 / 11)

    # Dropping the ties from the upper tail would leave a count of 1 and a
    # p-value of 4 / 11, which understates how ordinary a 2 is here.
    assert p_value > 4 / 11


def test_local_join_counts_two_sided_uses_simulations():
    # Local join counts have a discrete reference distribution, so ties with
    # the observed count are common. Check p_sim against the two-sided formula
    # applied to the stored simulations, and that it stays a valid p-value.
    weights = pytest.importorskip("libpysal.weights")

    w = weights.lat2W(4, 4)
    y = numpy.ones(16)
    y[0:8] = 0
    local = esda.Join_Counts_Local(
        connectivity=w, seed=12345, alternative="two-sided"
    ).fit(y)

    focal = local.LJC > 0
    expected = calculate_significance(
        local.LJC[focal], local.rjoins[focal], alternative="two-sided"
    )
    numpy.testing.assert_allclose(local.p_sim[focal], expected, rtol=1e-6)
    assert (local.p_sim[focal] > 0).all()
    assert (local.p_sim[focal] <= 1).all()


GLOBAL_ESTIMATORS = {
    "Moran": (lambda a: esda.Moran(x, w, permutations=19, alternative=a), "I", "p_sim"),
    "Moran_BV": (
        lambda a: esda.Moran_BV(x, x[::-1], w, permutations=19, alternative=a),
        "I",
        "p_sim",
    ),
    "Moran_Rate": (
        lambda a: esda.Moran_Rate(
            numpy.abs(x) + 1, numpy.full(800, 100.0), w, permutations=19, alternative=a
        ),
        "I",
        "p_sim",
    ),
    "Geary": (lambda a: esda.Geary(x, w, permutations=19, alternative=a), "C", "p_sim"),
    "G": (
        lambda a: esda.G(numpy.abs(x), w, permutations=19, alternative=a),
        "G",
        "p_sim",
    ),
    "Gamma": (
        lambda a: esda.Gamma(x, w, permutations=19, alternative=a),
        "g",
        "p_sim_g",
    ),
}


@pytest.mark.parametrize("name", list(GLOBAL_ESTIMATORS))
def test_global_estimators_use_calculate_significance(name):
    build, statistic, p_attr = GLOBAL_ESTIMATORS[name]
    estimator = build("two-sided")
    reference = estimator.sim_g if name == "Gamma" else estimator.sim
    expected = calculate_significance(
        getattr(estimator, statistic), reference, alternative="two-sided"
    )
    p_value = getattr(estimator, p_attr)
    assert isinstance(p_value, numpy.float64)
    numpy.testing.assert_allclose(p_value, expected)


@pytest.mark.parametrize("name", list(GLOBAL_ESTIMATORS))
def test_global_estimators_warn_without_alternative(name):
    build, _, p_attr = GLOBAL_ESTIMATORS[name]
    with pytest.warns(DeprecationWarning, match="permutation inference"):
        estimator = build(None)
    reference = estimator.sim_g if name == "Gamma" else estimator.sim
    statistic = getattr(estimator, GLOBAL_ESTIMATORS[name][1])
    expected = calculate_significance(statistic, reference, alternative="directed")
    numpy.testing.assert_allclose(getattr(estimator, p_attr), expected)


def test_join_counts_use_greater():
    y = (x > 0).astype(float)
    jc = esda.Join_Counts(y, w, permutations=19)
    expected = calculate_significance(
        float(jc.bb), jc.sim_bb.astype(float), alternative="greater"
    )
    numpy.testing.assert_allclose(jc.p_sim_bb, expected)


def test_spatial_pearson_alternative():
    from esda.lee import Spatial_Pearson, Spatial_Pearson_Local

    z = x.reshape(-1, 1)
    sp = Spatial_Pearson(w.sparse, permutations=19, alternative="two-sided").fit(
        z, z[::-1]
    )
    expected = calculate_significance(
        sp.association_.ravel(),
        sp.reference_distribution_.reshape(19, -1).T,
        alternative="two-sided",
    ).reshape(2, 2)
    numpy.testing.assert_allclose(sp.significance_, expected)

    spl = Spatial_Pearson_Local(w.sparse, permutations=19, alternative="two-sided")
    spl.fit(z, z[::-1])
    expected = calculate_significance(
        spl.associations_, spl.reference_distribution_.T, alternative="two-sided"
    )
    numpy.testing.assert_allclose(spl.significance_, expected)

    with pytest.warns(DeprecationWarning, match="permutation inference"):
        Spatial_Pearson(w.sparse, permutations=19).fit(z, z[::-1])


def test_geary_local_mv_alternative():
    from esda.geary_local_mv import Geary_Local_MV

    glmv = Geary_Local_MV(w, permutations=19, alternative="two-sided").fit([x, x[::-1]])
    expected = calculate_significance(glmv.localG, glmv.Gs, alternative="two-sided")
    numpy.testing.assert_allclose(glmv.p_sim, expected)

    with pytest.warns(DeprecationWarning, match="permutation inference"):
        Geary_Local_MV(w, permutations=19).fit([x, x[::-1]])
