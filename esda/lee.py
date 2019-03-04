import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn import utils
from libpysal.weights import lag_spatial, WSP

def _lee_from_R(x,y,W, local, zero_policy=None, NAOK=False):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r as R, numpy2ri
    from rpy2.rinterface import NULL

    zero_policy = NULL if zero_policy is None else zero_policy

    spdep = importr('spdep')
    
    numpy2ri.activate()
    W_matrix_R = numpy2ri.py2ro(W.sparse.toarray())
    W_listw_R = spdep.mat2listw(W_matrix_R)
    result = spdep.lee(x.flatten(), y.flatten(), W_listw_R,
                       len(x.flatten()), zero_policy=zero_policy, NAOK=NAOK)
    
    if local:
        result = numpy2ri.ri2py(result[1])
    else:
        result = numpy2ri.ri2py(result[0])
    numpy2ri.deactivate()
    return result

def RSpatial_Pearson(x,y,W, zero_policy=None, NAOK=False):
    return _lee_from_R(x,y,W, local=False, zero_policy=zero_policy, NAOK=NAOK)

def RLocal_Spatial_Pearson(x,y,W, zero_policy=None, NAOK=False):
    return _lee_from_R(x,y,W, local=True, zero_policy=zero_policy, NAOK=NAOK)

class Spatial_Pearson(BaseEstimator):
    def __init__(self, connectivity=None, permutations=999):
        self.connectivity = connectivity
        self.permutations = permutations

    def fit(self, x, y):
        """
        bivariate spatial pearson's R based on Eq. 18 of Lee (2001).
        
        L = \dfrac{Z^T (V^TV) Z}{1^T (V^TV) 1}
        
        Lee, Sang Il. (2001), "Developing a bivariate spatial 
        association measure: An integration of Pearson's r and 
        Moran's I." Journal of Geographical Systems, 3(4):369-385.
        
        """
        x = utils.check_array(x)
        y = utils.check_array(y)
        Z = numpy.column_stack((preprocessing.StandardScaler().fit_transform(x),
                                preprocessing.StandardScaler().fit_transform(y)))
        if self.connectivity is None:
            self.connectivity = sparse.eye(Z.shape[0])
        self.association_ = self.__statistic(Z, self.connectivity) 

        if (self.permutations is None):
            return self
        elif self.permutations < 1:
            return self
        
        if self.permutations:
            simulations = [self.__statistic(numpy.random.permutation(Z), self.connectivity)
                           for _ in range(self.permutations)]
            self.reference_distribution_ = simulations = numpy.array(simulations)
            above = simulations >= self.association_
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.significance_ = (larger + 1.) / (self.permutations + 1.)
        return self

    @staticmethod
    def __statistic(Z,W):
        ctc = W.T @ W
        ones = numpy.ones(ctc.shape[0])
        return (Z.T @ ctc @ Z) / (ones.T @ ctc @ ones)


class Local_Spatial_Pearson(BaseEstimator):
    def __init__(self, connectivity = None, permutations=999):
        self.connectivity = connectivity
        self.permutations = permutations

    def fit(self, x, y):
        """
        bivariate local pearson's R based on Eq. 22 in Lee (2001), using 
        site-wise conditional randomization from Moran_Local_BV.
        
        L_i = \dfrac{
                     n \cdot
                       \Big[\big(\sum_i w_{ij}(x_j - \bar{x})\big)
                            \big(\sum_i w_{ij}(y_j - \bar{y})\big) \Big]
                     } 
                    {
                     \sqrt{\sum_i (x_i - \bar{x})^2}
                     \sqrt{\sum_i (y_i - \bar{y})^2}}
            = \dfrac{
                     n \cdot
                       (\tilde{x}_j - \bar{x})
                       (\tilde{y}_j - \bar{y})
                     } 
                    {
                     \sqrt{\sum_i (x_i - \bar{x})^2}
                     \sqrt{\sum_i (y_i - \bar{y})^2}}
        
        Lee, Sang Il. (2001), "Developing a bivariate spatial 
        association measure: An integration of Pearson's r and 
        Moran's I." Journal of Geographical Systems, 3(4):369-385.
        
        """
        x = utils.check_array(x)
        y = utils.check_array(y)
        #x = preprocessing.StandardScaler().fit_transform(x)
        #y = preprocessing.StandardScaler().fit_transform(y)
        Z = numpy.column_stack((preprocessing.StandardScaler().fit_transform(x),
                                preprocessing.StandardScaler().fit_transform(y)))
        
        #slx = self.connectivity @ x
        #sly = self.connectivity @ y
        n, _ = x.shape
        #self.associations_ = n * (slx - x.mean()) * (sly - y.mean())
        #self.associations_ /= x.std() * y.std() # should always be one

        self.associations_ = self.__statistic(Z,self.connectivity)

        if self.permutations:
            W = WSP(self.connectivity).to_W()
            self.reference_distribution_ = numpy.empty((n, self.permutations))
            k = W.max_neighbors + 1
            random_ids = numpy.array([numpy.random.permutation(n - 1)[0:W.max_neighbors + 1]
                                      for i in range(self.permutations)])
            ids = numpy.arange(n)
            weights = [W.weights[idx] for idx in W.id_order]
            cardinalities = [W.cardinalities[idx] for idx in W.id_order]

            for i in range(n):
                ids_not_i = ids[ids != i]
                numpy.random.shuffle(ids_not_i)
                randomizer = random_ids[:, 0:cardinalities[i]]
                random_neighbors = ids_not_i[randomizer]
                random_neighbor_x = x[random_neighbors]
                random_neighbor_y = y[random_neighbors]

                self.reference_distribution_[i]  = ((random_neighbor_x * weights[i]).sum() - x.mean())
                self.reference_distribution_[i] *= ((random_neighbor_y * weights[i]).sum() - y.mean())
            self.reference_distribution_ *= (n / x.std() * y.std())
        return self

    @staticmethod
    def __statistic(Z,W):
        ctc = W.T @ W
        ones = numpy.ones(ctc.shape[0])
        return (Z[:,1] @ W.T) * (W @ Z[:,0]) 

if __name__ == '__main__':
    import geopandas
    import pysal
    df = geopandas.read_file(pysal.examples.get_path('columbus.shp'))
    x = df[['HOVAL']].values
    y = df[['CRIME']].values
    zx = preprocessing.StandardScaler().fit_transform(x)
    zy = preprocessing.StandardScaler().fit_transform(y)
    w = pysal.weights.Queen.from_dataframe(df)
    w.transform = 'r'
    testglobal_r = RSpatial_Pearson(x,y,w)
    testglobal = Spatial_Pearson(connectivity=w.sparse).fit(x,y)
    testlocal_r = RLocal_Spatial_Pearson(x,y,w)
    testlocal = Local_Spatial_Pearson(connectivity=w.sparse).fit(x,y)
