import numpy
import esda
from esda.significance import calculate_significance
import pytest
from libpysal.weights import Voronoi

numpy.random.seed(2478879)
coordinates = numpy.random.random(size=(800, 2))
x = numpy.random.normal(size=(800,))
w = Voronoi(coordinates, clip="bounding_box")
w.transform = "r"
stat = esda.Moran_Local(x, w, permutations=19)

@pytest.mark.parametrize("alternative", ["two-sided", "directed", "lesser", "greater", "folded"])
def test_execution_and_range(alternative):
    out = calculate_significance(stat.Is, stat.rlisas, alternative=alternative)
    assert (out > 0).all() & (out <= 1).all(), f'p-value out of bounds for method {alternative}'
    if alternative == 'directed':
        assert out.max() <= .5, f"max p-value is too large for method {alternative}"
    else:
        assert out.max() >= .5, f"max p-value is too small for method {alternative}"
    

def test_alternative_relationships():
    two_sided = calculate_significance(
        stat.Is, 
        stat.rlisas, 
        alternative="two-sided"
    )
    directed = calculate_significance(
        stat.Is, 
        stat.rlisas, 
        alternative="directed"
    )
    lesser = calculate_significance(
        stat.Is, 
        stat.rlisas, 
        alternative="lesser"
    )
    greater = calculate_significance(
        stat.Is, 
        stat.rlisas, 
        alternative="greater"
    )
    folded = calculate_significance(
        stat.Is, 
        stat.rlisas, 
        alternative="folded"
    )

    numpy.testing.assert_allclose(lesser + greater,
                                  numpy.ones_like(lesser) + (1/(stat.permutations + 1)),
        err_msg = "greater p-value should be complement of lesser"
    )
    assert (directed <= two_sided).all(), "directed is bigger than two_sided and should not be"
    one_or_the_other = (directed == lesser) | (directed == greater)
    assert one_or_the_other.all(), "some directed p-value is neither the greater nor lesser p-value"
    assert (two_sided < folded).mean() < (directed < folded).mean(), (
	"Directed p-values should tend to be much smaller than two_sided p-values or folded p-values."
    )
