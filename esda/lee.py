import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn import utils
from libpysal.weights import WSP

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
        self.association_ = self._statistic(Z, self.connectivity) 

        if (self.permutations is None):
            return self
        elif self.permutations < 1:
            return self

        if self.permutations:
            simulations = [self._statistic(numpy.random.permutation(Z), self.connectivity)
                           for _ in range(self.permutations)]
            self.reference_distribution_ = simulations = numpy.array(simulations)
            above = simulations >= self.association_
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.significance_ = (larger + 1.) / (self.permutations + 1.)
        return self

    @staticmethod
    def _statistic(Z,W):
        ctc = W.T @ W
        ones = numpy.ones(ctc.shape[0])
        return (Z.T @ ctc @ Z) / (ones.T @ ctc @ ones)


class Local_Spatial_Pearson(BaseEstimator):
    def __init__(self, connectivity=None, permutations=999):
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
        Z = numpy.column_stack((preprocessing.StandardScaler().fit_transform(x),
                                preprocessing.StandardScaler().fit_transform(y)))
        
        n, _ = x.shape

        self.associations_ = self._statistic(Z,self.connectivity)

        if self.permutations:
            W = WSP(self.connectivity).to_W()
            self.reference_distribution_ = numpy.empty((n, self.permutations))
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

                self.reference_distribution_[i]  = ((random_neighbor_x * weights[i])
                                                    .sum() - x.mean())
                self.reference_distribution_[i] *= ((random_neighbor_y * weights[i])
                                                    .sum() - y.mean())
            self.reference_distribution_ *= (n / (x.std() * y.std()))
            above = self.reference_distribution_ >= self.associations_.reshape(-1,1)
            larger = above.sum(axis=1)
            extreme = numpy.minimum(larger, self.permutations - larger)
            self.significance_ = (extreme + 1.) / (self.permutations + 1.)

        return self

    @staticmethod
    def _statistic(Z,W):
        return (Z[:,1] @ W.T) * (W @ Z[:,0]) 

if __name__ == '__main__':
    import geopandas
    import libpysal
    df = geopandas.read_file(libpysal.examples.get_path('columbus.shp'))
    x = df[['HOVAL']].values
    y = df[['CRIME']].values
    zx = preprocessing.StandardScaler().fit_transform(x)
    zy = preprocessing.StandardScaler().fit_transform(y)
    w = libpysal.weights.Queen.from_dataframe(df)
    w.transform = 'r'
    testglobal = Spatial_Pearson(connectivity=w.sparse).fit(x,y)
    testlocal = Local_Spatial_Pearson(connectivity=w.sparse).fit(x,y)
