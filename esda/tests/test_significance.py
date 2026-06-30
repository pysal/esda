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
