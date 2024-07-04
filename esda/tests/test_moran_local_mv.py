import numpy
import geodatasets
import geopandas
from libpysal.weights import Queen
from sklearn.linear_model import TheilSenRegressor
from esda.multivariate_moran import Partial_Moran_Local, Auxiliary_Moran_Local

df = geopandas.read_file(geodatasets.get_path("geoda.lansing1"))
df = df[df.FIPS.str.match("2606500[01234]...") | (df.FIPS == "26065006500")]

y = df.HH_INC.values
X = df.HSG_VAL.values
w = Queen.from_dataframe(df) 

def test_partial_runs():
    """Check if the class computes successfully in a default configuration"""
    m = Partial_Moran_Local(y,X,w, permutations=1)
    # done, just check if it runs

def test_partial_accuracy():
    """Check if the class outputs expected results at a given seed"""
    numpy.random.seed(111221)
    m = Partial_Moran_Local(y,X,w, permutations=10)
    # compute result by hand

def test_partial_unscaled():
    """Check if the variance scaling behaves as expected"""
    m = Partial_Moran_Local(y,X,w, permutations=0, unit_scale=True)
    m2 = Partial_Moran_Local(y,X,w, permutations=0, unit_scale=False)
    # variance in the partials_ should be different

def test_partial_uvquads():
    """Check that the quadrant decisions vary correctly with the inputs"""
    m = Partial_Moran_Local(y,X,w, permutations=0, mvquads=False)
    ...

def test_aux_runs():
    """Check that the class completes successfully in a default configuration"""
    m = Auxiliary_Moran_Local(y,X,w, permutations=1)
    ...

def test_aux_accuracy():
    """Check that the class outputs expected values for a given seed"""
    numpy.random.seed(111221)
    m = Auxiliary_Moran_Local(y,X,w, permutations=10)
    ...

def test_aux_unscaled():
    """Check that the variance scaling behaves as expected"""
    m = Auxiliary_Moran_Local(y,X,w, permutations=0, unit_scale=True)
    m2 = Auxiliary_Moran_Local(y,X,w, permutations=0, unit_scale=False)
    ...

def test_aux_transformer():
    """Check that an alternative regressor can be used to calculate y|X"""
    m = Auxiliary_Moran_Local(y,X,w, permutations=0, transformer=TheilSenRegressor)
    ...