import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from libpysal import weights
from esda.crand import (
    crand as _crand_plus,
    njit as _njit,
    _prepare_univariate
)


class Local_Join_Count(BaseEstimator):

    """Univariate Local Join Count Statistic"""

    def __init__(self, connectivity=None, permutations=999, n_jobs=1, 
                 keep_simulations=True, seed=None):
        """
        Initialize a Local_Join_Count estimator
        Arguments
        ---------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing
                           the relationships between observed units.
                           Need not be row-standardized.
        permutations     : int
                           number of random permutations for calculation of pseudo
                           p_values
        n_jobs           : int
                           Number of cores to be used in the conditional randomisation. If -1,
                           all available cores are used.    
        keep_simulations : Boolean
                           (default=True)
                           If True, the entire matrix of replications under the null 
                           is stored in memory and accessible; otherwise, replications 
                           are not saved
        seed             : None/int
                           Seed to ensure reproducibility of conditional randomizations. 
                           Must be set here, and not outside of the function, since numba 
                           does not correctly interpret external seeds 
                           nor numpy.random.RandomState instances.              
                           
        Attributes
        ----------
        LJC             : numpy array
                          array containing the univariate
                          Local Join Count (LJC).
        p_sim           : numpy array
                          array containing the simulated
                          p-values for each unit.

        """

        self.connectivity = connectivity
        self.permutations = permutations
        self.n_jobs = n_jobs
        self.keep_simulations = keep_simulations
        self.seed = seed

    def fit(self, x):
        """
        Arguments
        ---------
        x               : numpy.ndarray
                          array containing binary (0/1) data
        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations found in :cite:`AnselinLi2019`.

        Examples
        --------
        >>> import libpysal
        >>> w = libpysal.weights.lat2W(4, 4)
        >>> x = np.ones(16)
        >>> x[0:8] = 0
        >>> LJC_uni = Local_Join_Count(connectivity=w).fit(x)
        >>> LJC_uni.LJC
        >>> LJC_uni.p_sim

        Guerry data replicating GeoDa tutorial
        >>> import libpysal
        >>> import geopandas as gpd
        >>> guerry = libpysal.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> guerry_ds['SELECTED'] = 0
        >>> guerry_ds.loc[(guerry_ds['Donatns'] > 10997), 'SELECTED'] = 1
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> LJC_uni = Local_Join_Count(connectivity=w).fit(guerry_ds['SELECTED'])
        >>> LJC_uni.LJC
        >>> LJC_uni.p_sim
        """
        # Need to ensure that the np.array() are of
        # dtype='float' for numba
        x = np.array(x, dtype='float')

        w = self.connectivity
        # Fill the diagonal with 0s
        w = weights.util.fill_diagonal(w, val=0)
        w.transform = 'b'
        
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs
        seed = self.seed
        
        permutations = self.permutations
        
        self.x = x
        self.n = len(x)
        self.w = w

        self.LJC = self._statistic(x, w)
        
        if permutations:
            self.p_sim, self.rjoins = _crand_plus(
                z=self.x, 
                w=self.w, 
                observed=self.LJC,
                permutations=permutations, 
                keep=keep_simulations, 
                n_jobs=n_jobs,
                stat_func=_ljc_uni
            )
            # Set p-values for those with LJC of 0 to NaN
            self.p_sim[self.LJC == 0] = 'NaN'
        
        del (self.n, self.keep_simulations, self.n_jobs, 
             self.permutations, self.seed, self.w, self.x,
             self.connectivity, self.rjoins)
        
        return self

    @staticmethod
    def _statistic(x, w):
        # Create adjacency list. Note that remove_symmetric=False - this is
        # different from the esda.Join_Counts() function.
        adj_list = w.to_adjlist(remove_symmetric=False)
        zseries = pd.Series(x, index=w.id_order)
        focal = zseries.loc[adj_list.focal].values
        neighbor = zseries.loc[adj_list.neighbor].values
        LJC = (focal == 1) & (neighbor == 1)
        adj_list_LJC = pd.DataFrame(adj_list.focal.values,
                                    LJC.astype('uint8')).reset_index()
        adj_list_LJC.columns = ['LJC', 'ID']
        adj_list_LJC = adj_list_LJC.groupby(by='ID').sum()
        LJC = np.array(adj_list_LJC.LJC.values, dtype='float')
        return (LJC)

# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: scaling not used

@_njit(fastmath=True)
def _ljc_uni(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return zi * (zrand @ weights_i)