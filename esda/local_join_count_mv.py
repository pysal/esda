import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from libpysal import weights
from esda.crand import (
    crand as _crand_plus,
    njit as _njit,
    _prepare_univariate
)


class Local_Join_Count_MV(BaseEstimator):

    """Multivariate Local Join Count Statistic"""

    def __init__(self, connectivity=None, permutations=999, n_jobs=1, 
                 keep_simulations=True, seed=None):
        """
        Initialize a Local_Join_Count_MV estimator
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
                           
        """

        self.connectivity = connectivity
        self.permutations = permutations
        self.n_jobs = n_jobs
        self.keep_simulations = keep_simulations
        self.seed = seed

    def fit(self, variables):
        """
        Arguments
        ---------
        variables     : numpy.ndarray
                        array(s) containing binary (0/1) data

        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations can be found in :cite:`AnselinLi2019`.

        Examples
        --------
        >>> import libpysal
        >>> w = libpysal.weights.lat2W(4, 4)
        >>> x = np.ones(16)
        >>> x[0:8] = 0
        >>> z = [0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,1]
        >>> y = [0,1,1,1,1,1,1,1,0,0,0,1,0,0,1,1]
        >>> LJC_MV = Local_Join_Count_MV(connectivity=w).fit([x, y, z])
        >>> LJC_MV.LJC
        >>> LJC_MV.p_sim

        Guerry data extending GeoDa tutorial
        >>> import libpysal
        >>> import geopandas as gpd
        >>> guerry = libpysal.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> guerry_ds['infq5'] = 0
        >>> guerry_ds['donq5'] = 0
        >>> guerry_ds['suic5'] = 0
        >>> guerry_ds.loc[(guerry_ds['Infants'] > 23574), 'infq5'] = 1
        >>> guerry_ds.loc[(guerry_ds['Donatns'] > 10973), 'donq5'] = 1
        >>> guerry_ds.loc[(guerry_ds['Suicids'] > 55564), 'suic5'] = 1
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> LJC_MV = Local_Join_Count_MV(connectivity=w).fit([guerry_ds['infq5'], guerry_ds['donq5'], guerry_ds['suic5']])
        >>> LJC_MV.LJC
        >>> LJC_MV.p_sim
        """

        w = self.connectivity
        # Fill the diagonal with 0s
        w = weights.util.fill_diagonal(w, val=0)
        w.transform = 'b'

        self.n = len(variables[0])
        self.w = w
        
        permutations = self.permutations

        self.variables = np.array(variables, dtype='float')
        
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs
        seed = self.seed

        # Need to ensure that the product is an 
        # np.array() of dtype='float' for numba
        self.ext = np.array(np.prod(np.vstack(variables), axis=0), 
                            dtype='float')

        self.LJC = self._statistic(variables, w)

        if permutations:
            self.p_sim, self.rjoins = _crand_plus(
                z=self.ext, 
                w=self.w, 
                observed=self.LJC,
                permutations=permutations, 
                keep=True, 
                n_jobs=n_jobs,
                stat_func=_ljc_mv
            )
            # Set p-values for those with LJC of 0 to NaN
            self.p_sim[self.LJC == 0] = 'NaN'
        
        del (self.n, self.keep_simulations, self.n_jobs, 
             self.permutations, self.seed, self.w, self.ext,
             self.variables, self.connectivity, self.rjoins)

        return self

    @staticmethod
    def _statistic(variables, w):
        # Create adjacency list. Note that remove_symmetric=False -
        # different from the esda.Join_Counts() function.
        adj_list = w.to_adjlist(remove_symmetric=False)

        # The zseries
        zseries = [pd.Series(i, index=w.id_order) for i in variables]
        # The focal values
        focal = [zseries[i].loc[adj_list.focal].values for
                 i in range(len(variables))]
        # The neighbor values
        neighbor = [zseries[i].loc[adj_list.neighbor].values for
                    i in range(len(variables))]

        # Find instances where all surrounding
        # focal and neighbor values == 1
        focal_all = np.array(np.all(np.dstack(focal) == 1,
                                    axis=2))
        neighbor_all = np.array(np.all(np.dstack(neighbor) == 1,
                                       axis=2))
        MCLC = (focal_all == True) & (neighbor_all == True)
        # Convert list of True/False to boolean array 
        # and unlist (necessary for building pd.DF)
        MCLC = list(MCLC*1)

        # Create a df that uses the adjacency list
        # focal values and the BBs counts
        adj_list_MCLC = pd.DataFrame(adj_list.focal.values,
                                     MCLC).reset_index()
        # Temporarily rename the columns
        adj_list_MCLC.columns = ['MCLC', 'ID']
        adj_list_MCLC = adj_list_MCLC.groupby(by='ID').sum()

        return (np.array(adj_list_MCLC.MCLC.values, dtype='float'))

# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: scaling not used

@_njit(fastmath=True)
def _ljc_mv(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return zi * (zrand @ weights_i)