import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator
import libpysal as lp


class Geary_Local_MV(BaseEstimator):

    """Local Geary - Multivariate"""

    def __init__(self, connectivity=None, permutations=999):
        """
        Initialize a Local_Geary_MV estimator
        Arguments
        ---------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing
                           the relationships between observed units.
                           Need not be row-standardized.
        permutations     : int
                           (default=999)
                           number of random permutations for calculation
                           of pseudo p_values
        Attributes
        ----------
        localG          : numpy array
                          array containing the observed multivariate
                          Local Geary values.
        p_sim           : numpy array
                          array containing the simulated
                          p-values for each unit.
        """

        self.connectivity = connectivity
        self.permutations = permutations

    def fit(self, variables):
        """
        Arguments
        ---------
        variables        : numpy.ndarray
                           array containing continuous data

        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations can be found in :cite:`Anselin1995`.

        Examples
        --------
        Guerry data replication GeoDa tutorial
        >>> import libpysal
        >>> import geopandas as gpd
        >>> guerry = lp.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> import libpysal
        >>> import geopandas as gpd
        >>> guerry = lp.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> x1 = guerry_ds['Donatns']
        >>> x2 = guerry_ds['Suicids']
        >>> lG_mv = Local_Geary(connectivity=w).fit([x1,x2])
        >>> lG_mv.localG[0:5]
        >>> lG_mv.p_sim[0:5]
        """
        self.variables = np.array(variables, dtype='float')

        w = self.connectivity
        w.transform = 'r'

        self.n = len(variables[0])
        self.w = w
        
        permutations = self.permutations

        # Caclulate z-scores for input variables
        # to be used in _statistic and _crand
        zvariables = [stats.zscore(i) for i in variables]

        self.localG = self._statistic(variables, zvariables, w)

        if permutations:
            self._crand(zvariables)
            sim = np.transpose(self.Gs)
            above = sim >= self.localG
            larger = above.sum(0)
            low_extreme = (permutations - larger) < larger
            larger[low_extreme] = permutations - larger[low_extreme]
            self.p_sim = (larger + 1.0) / (permutations + 1.0)

        return self

    @staticmethod
    def _statistic(variables, zvariables, w):
        # Define denominator adjustment
        k = len(variables)
        # Create focal and neighbor values
        adj_list = w.to_adjlist(remove_symmetric=False)
        zseries = [pd.Series(i, index=w.id_order) for i in zvariables]
        focal = [zseries[i].loc[adj_list.focal].values for
                 i in range(len(variables))]
        neighbor = [zseries[i].loc[adj_list.neighbor].values for
                    i in range(len(variables))]
        # Carry out local Geary calculation
        gs = adj_list.weight.values * \
            (np.array(focal) - np.array(neighbor))**2
        # Reorganize data
        temp = pd.DataFrame(gs).T
        temp['ID'] = adj_list.focal.values
        adj_list_gs = temp.groupby(by='ID').sum()
        localG = np.array(adj_list_gs.sum(axis=1)/k)

        return (localG)

    def _crand(self, zvariables):
        """
        conditional randomization

        for observation i with ni neighbors,  the candidate set cannot include
        i (we don't want i being a neighbor of i). we have to sample without
        replacement from a set of ids that doesn't include i. numpy doesn't
        directly support sampling wo replacement and it is expensive to
        implement this. instead we omit i from the original ids,  permute the
        ids and take the first ni elements of the permuted ids as the
        neighbors to i in each randomization.

        """
        nvars = self.variables.shape[0]
        n = self.variables.shape[1]
        Gs = np.zeros((self.n, self.permutations))
        n_1 = self.n - 1
        prange = list(range(self.permutations))
        k = self.w.max_neighbors + 1
        nn = self.n - 1
        rids = np.array([np.random.permutation(nn)[0:k] for i in prange])
        ids = np.arange(self.w.n)
        ido = self.w.id_order
        w = [self.w.weights[ido[i]] for i in ids]
        wc = [self.w.cardinalities[ido[i]] for i in ids]

        for i in range(self.w.n):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            vars_rand = []
            for j in range(nvars):
                vars_rand.append(zvariables[j][idsi[rids[:, 0:wc[i]]]])
            # vars rand as tmp
            # Calculate diff
            diff = []
            for z in range(nvars):
                diff.append((np.array((zvariables[z][i]-vars_rand[z])**2
                                      * w[i])).sum(1)/nvars)
            # add up differences
            temp = np.array([sum(x) for x in zip(*diff)])
            # Assign to object to be returned
            Gs[i] = temp
        self.Gs = Gs
