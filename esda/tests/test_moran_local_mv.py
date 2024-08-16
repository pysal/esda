import numpy
import geodatasets
import geopandas
import pytest
from libpysal.weights import Queen
from libpysal.graph import Graph
from sklearn.linear_model import TheilSenRegressor
from esda.moran_local_mv import Partial_Moran_Local, Auxiliary_Moran_Local
from esda.moran import Moran_Local_BV

df = geopandas.read_file(geodatasets.get_path("geoda.lansing1"))
df = df[df.FIPS.str.match("2606500[01234]...") | (df.FIPS == "26065006500")]

y = df.HH_INC.values.reshape(-1,1)
X = df.HSG_VAL.values.reshape(-1,1)
w_classic = Queen.from_dataframe(df) 
w_classic.transform = 'r'
g = Graph.build_contiguity(df).transform("r")

test_over_w = pytest.mark.parametrize("w", [w_classic, g], ids=['W', 'Graph'])

@test_over_w
def test_partial_runs(w):
    """Check if the class computes successfully in a default configuration"""
    m = Partial_Moran_Local(y,X,w, permutations=1)
    # done, just check if it runs

@test_over_w
def test_partial_accuracy(w):
    """Check if the class outputs expected results at a given seed"""
    numpy.random.seed(111221)
    m = Partial_Moran_Local(y,X,w, permutations=10)
    # compute result by hand
    zy = (y - y.mean())/y.std()
    wz = (w.sparse @ zy)
    zx = (X - X.mean(axis=0))/X.std(axis=0)
    rho = numpy.corrcoef(zy.squeeze(), zx.squeeze())[0,1]
    left = zy - rho * zx
    scale = (w.n-1) / (w.n * (1 - rho**2))
    # (y - rho x)*wy
    manual = (left*wz).squeeze() * scale
    # check values
    numpy.testing.assert_allclose(manual, m.associations_)

    # check significances are about 18
    numpy.testing.assert_allclose((m.significances_ < .01).sum(), 18, atol=1)
    numpy.testing.assert_equal((m.significances_[:5] < .1), [True, True, True, False, False])

    # check quad
    is_cluster = numpy.prod(m.partials_, axis=1) >= 0
    is_odd_label = m.labels_ % 2
    numpy.testing.assert_equal(is_cluster, is_odd_label)

@test_over_w
def test_partial_unscaled(w):
    """Check if the variance scaling behaves as expected"""
    m = Partial_Moran_Local(y,X,w, permutations=0, unit_scale=True)
    m2 = Partial_Moran_Local(y,X,w, permutations=0, unit_scale=False)
    # variance in the partials_ should be different
    s1y,s1x = m.partials_.std(axis=0)
    s2y,s2x = m2.partials_.std(axis=0)
    assert s1y > s2y, "variance is incorrectly scaled for y"
    assert s1x < s2x, "variance is incorrectly scaled for x"

@test_over_w
def test_partial_uvquads(w):
    """Check that the quadrant decisions vary correctly with the inputs"""
    m = Partial_Moran_Local(y,X,w, permutations=0, mvquads=False)
    bvx = Moran_Local_BV(X,y,w,permutations=0)
    # TODO: this currently fails, and it should pass. I am probably mis-calculating the bivariate quadrants for this option, and need to correct the code. 
    numpy.testing.assert_array_equal(m.quads_, bvx.q)

@test_over_w
def test_aux_runs(w):
    """Check that the class completes successfully in a default configuration"""
    a = Auxiliary_Moran_Local(y,X, w, permutations=1)
    #done, just check if it runs

@test_over_w
def test_aux_accuracy(w):
    """Check that the class outputs expected values for a given seed"""
    numpy.random.seed(111221)
    a = Auxiliary_Moran_Local(y,X,w, permutations=10)

    # compute result by hand
    zy = (y - y.mean())/y.std()
    wz = (w.sparse @ zy)
    zx = (X - X.mean(axis=0))/X.std(axis=0)
    wzx = w.sparse @ zx
    rho = numpy.corrcoef(zy.squeeze(), zx.squeeze())[0,1]
    mean = zy * wz - rho * zx * wz - rho * zy * wzx + rho**2 * zx * wzx
    scale = (w.n-1) / (w.n * (1 - rho**2))
    
    manual = numpy.asarray(mean * scale).squeeze()
    # check values, may not be identical because of the 
    # matrix inversion least squares estimator used in scikit
    numpy.testing.assert_allclose(manual, a.associations_)

    # check significances
    numpy.testing.assert_equal((a.significances_ < .01).sum(), 18)
    numpy.testing.assert_equal((a.significances_[:5] < .1), [False, False, True, False, False])

    is_cluster = numpy.prod(a.partials_, axis=1) >= 0
    is_odd_label = (a.labels_ % 2).astype(bool)
    numpy.testing.assert_equal(is_cluster, is_odd_label)

@test_over_w
def test_aux_unscaled(w):
    """Check that the variance scaling behaves as expected"""
    a = Auxiliary_Moran_Local(y,X/10000,w, permutations=0, unit_scale=True)
    a2 = Auxiliary_Moran_Local(y,X,w, permutations=0, unit_scale=False)
    assert (a.partials_.std(axis=0) < a2.partials_.std(axis=0)).all(), (
        "variance is not scaled correctly in partial regression."
        )

@test_over_w
def test_aux_transformer(w):
    """Check that an alternative regressor can be used to calculate y|X"""
    a = Auxiliary_Moran_Local(y,X,w, permutations=0, transformer=TheilSenRegressor)
    # done, should just complete