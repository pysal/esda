import numpy
import geodatasets
import geopandas
import pytest
from libpysal.weights import Rook
from libpysal.graph import Graph
from sklearn.linear_model import TheilSenRegressor
from esda.moran_local_mv import MoranLocalPartial, MoranLocalConditional
from esda.moran import Moran_Local_BV

def rsrook(df):
    w_classic = Rook.from_dataframe(df) 
    w_classic.transform = 'r'
    return w_classic

@pytest.fixture(scope='module')
def data():
    df = geopandas.read_file(geodatasets.get_path("geoda.lansing1"))
    df = df[df.FIPS.str.match("2606500[01234]...") | (df.FIPS == "26065006500")]
    df=df.reset_index(drop=True)
    y = df.HH_INC.values.reshape(-1,1)
    X = df.HSG_VAL.values.reshape(-1,1)
    yield y,X,df

@pytest.fixture(scope='module', 
                params = [
                        rsrook,
                        lambda df: Graph.build_contiguity(df.reset_index(drop=True)).transform('r')
                        ],
                ids=['W', 'Graph']
                )
def graph(data, request):
    _,_,df = data
    return request.param(df)

def test_partial_runs(data, graph):
    """Check if the class computes successfully in a default configuration"""
    y,X,df = data
    m = MoranLocalPartial(permutations=1).fit(X,y,graph)
    # done, just check if it runs

def test_partial_accuracy(data, graph):
    """Check if the class outputs expected results at a given seed"""
    y,X,df = data
    numpy.random.seed(111221)
    m = MoranLocalPartial(permutations=10).fit(X,y,graph)
    # compute result by hand
    zy = (y - y.mean())/y.std()
    wz = (graph.sparse @ zy)
    zx = (X - X.mean(axis=0))/X.std(axis=0)
    rho = numpy.corrcoef(zy.squeeze(), zx.squeeze())[0,1]
    left = zy - rho * zx
    scale = (graph.n-1) / (graph.n * (1 - rho**2))
    # (y - rho x)*wy
    manual = (left*wz).squeeze() * scale
    # check values
    numpy.testing.assert_allclose(manual, m.association_)

    # check significances are about 18
    print(m.significance_)
    numpy.testing.assert_allclose((m.significance_ < .4).sum(), 26, atol=1)
    numpy.testing.assert_equal((m.significance_[:5] < .4), [True, True, True, False, False])

    # check quad
    is_cluster = numpy.prod(m.partials_, axis=1) >= 0
    is_odd_label = m.labels_ % 2
    numpy.testing.assert_equal(is_cluster, is_odd_label)

def test_partial_unscaled(data, graph):
    """Check if the variance scaling behaves as expected"""
    y,X,df = data
    m = MoranLocalPartial(permutations=0, unit_scale=True).fit(X,y,graph)
    m2 = MoranLocalPartial(permutations=0, unit_scale=False).fit(X,y,graph)
    # variance in the partials_ should be different
    s1y,s1x = m.partials_.std(axis=0)
    s2y,s2x = m2.partials_.std(axis=0)
    assert s1y > s2y, "variance is incorrectly scaled for y"
    assert s1x < s2x, "variance is incorrectly scaled for x"

def test_partial_uvquads(data, graph):
    """Check that the quadrant decisions vary correctly with the inputs"""
    y,X,df = data
    m = MoranLocalPartial(permutations=0, partial_labels=False).fit(X,y,graph)
    bvx = Moran_Local_BV(X,y,graph,permutations=0)
    numpy.testing.assert_array_equal(m.labels_, bvx.q)

def test_conditional_runs(data, graph):
    """Check that the class completes successfully in a default configuration"""
    y,X,df = data
    a = MoranLocalConditional(permutations=1).fit(X,y,graph)
    #done, just check if it runs

def test_conditional_accuracy(data, graph):
    """Check that the class outputs expected values for a given seed"""
    y,X,df = data
    numpy.random.seed(111221)
    a = MoranLocalConditional(permutations=10).fit(X,y,graph)

    # compute result by hand
    zy = (y - y.mean())/y.std()
    wz = (graph.sparse @ zy)
    zx = (X - X.mean(axis=0))/X.std(axis=0)
    wzx = graph.sparse @ zx
    rho = numpy.corrcoef(zy.squeeze(), zx.squeeze())[0,1]
    mean = zy * wz - rho * zx * wz - rho * zy * wzx + rho**2 * zx * wzx
    scale = (graph.n-1) / (graph.n * (1 - rho**2))
    
    manual = numpy.asarray(mean * scale).squeeze()
    # check values, may not be identical because of the 
    # matrix inversion least squares estimator used in scikit
    numpy.testing.assert_allclose(manual, a.association_)

    # check significances
    numpy.testing.assert_equal((a.significance_ < .4).sum(), 31)
    numpy.testing.assert_equal((a.significance_[:5] < .4), [False, False, True, True, False])

    is_cluster = numpy.prod(a.partials_, axis=1) >= 0
    is_odd_label = (a.labels_ % 2).astype(bool)
    numpy.testing.assert_equal(is_cluster, is_odd_label)

def test_conditional_unscaled(data, graph):
    """Check that the variance scaling behaves as expected"""
    y,X,df = data
    a = MoranLocalConditional(permutations=0, unit_scale=True).fit(X,y,graph)
    a2 = MoranLocalConditional(permutations=0, unit_scale=False).fit(X,y,graph)
    assert (a.partials_.std(axis=0) < a2.partials_.std(axis=0)).all(), (
        "variance is not scaled correctly in partial regression."
        )

def test_conditional_transformer(data, graph):
    """Check that an alternative regressor can be used to calculate y|X"""
    y,X, df = data
    a = MoranLocalConditional(permutations=0, transformer=TheilSenRegressor).fit(X,y,graph)
    # done, should just complete