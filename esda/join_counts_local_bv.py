import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from sklearn.base import BaseEstimator
from libpysal import weights
from esda.crand import (
    crand as _crand_plus,
    njit as _njit,
    _prepare_univariate,
    _prepare_bivariate
)


PERMUTATIONS = 999


class Join_Counts_Local_BV(BaseEstimator):

    """Univariate Local Join Count Statistic"""

    def __init__(self, connectivity=None, permutations=PERMUTATIONS, n_jobs=1, 
                 keep_simulations=True, seed=None):
        """
        Initialize a Local_Join_Counts_BV estimator
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

    def fit(self, x, z, case="CLC", n_jobs=1, permutations=999):
        """
        Arguments
        ---------
        x                : numpy.ndarray
                           array containing binary (0/1) data
        z                : numpy.ndarray
                           array containing binary (0/1) data
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
        >>> LJC_BV_C1 = Local_Join_Counts_BV(connectivity=w).fit(x, z, case="BJC")
        >>> LJC_BV_C2 = Local_Join_Counts_BV(connectivity=w).fit(x, z, case="CLC")
        >>> LJC_BV_C1.LJC
        >>> LJC_BV_C1.p_sim
        >>> LJC_BV_C2.LJC
        >>> LJC_BV_C2.p_sim

        Commpop data replicating GeoDa tutorial (Case 1)
        >>> import libpysal
        >>> import geopandas as gpd
        >>> commpop = gpd.read_file("https://github.com/jeffcsauer/GSOC2020/raw/master/validation/data/commpop.gpkg")
        >>> w = libpysal.weights.Queen.from_dataframe(commpop)
        >>> LJC_BV_Case1 = Local_Join_Counts_BV(connectivity=w).fit(commpop['popneg'], commpop['popplus'], case='BJC')
        >>> LJC_BV_Case1.LJC
        >>> LJC_BV_Case1.p_sim

        Guerry data replicating GeoDa tutorial (Case 2)
        >>> import libpysal
        >>> import geopandas as gpd
        >>> guerry = libpysal.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> guerry_ds['infq5'] = 0
        >>> guerry_ds['donq5'] = 0
        >>> guerry_ds.loc[(guerry_ds['Infants'] > 23574), 'infq5'] = 1
        >>> guerry_ds.loc[(guerry_ds['Donatns'] > 10973), 'donq5'] = 1
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> LJC_BV_Case2 = Local_Join_Counts_BV(connectivity=w).fit(guerry_ds['infq5'], guerry_ds['donq5'], case='CLC')
        >>> LJC_BV_Case2.LJC
        >>> LJC_BV_Case2.p_sim
        """
        # Need to ensure that the np.array() are of
        # dtype='float' for numba
        x = np.array(x, dtype='float')
        z = np.array(z, dtype='float')

        w = self.connectivity
        # Fill the diagonal with 0s
        w = weights.util.fill_diagonal(w, val=0)
        w.transform = 'b'

        self.x = x
        self.z = z
        self.n = len(x)
        self.w = w
        self.case = case
        
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs
        seed = self.seed

        self.LJC = self._statistic(x, z, w, case=case)

        if permutations:
            if case == "BJC":
                self.p_sim, self.rjoins = _crand_plus(
                    z=np.column_stack((x, z)),
                    w=self.w, 
                    observed=self.LJC,
                    permutations=permutations, 
                    keep=True, 
                    n_jobs=n_jobs,
                    stat_func=_ljc_bv_case1
                )
                # Set p-values for those with LJC of 0 to NaN
                self.p_sim[self.LJC == 0] = 'NaN'
            elif case == "CLC":
                self.p_sim, self.rjoins = _crand_plus(
                    z=np.column_stack((x, z)),
                    w=self.w, 
                    observed=self.LJC,
                    permutations=permutations, 
                    keep=True, 
                    n_jobs=n_jobs,
                    stat_func=_ljc_bv_case2
                )
                # Set p-values for those with LJC of 0 to NaN
                self.p_sim[self.LJC == 0] = 'NaN'
            else:
                raise NotImplementedError(f'The requested LJC method ({case}) \
                is not currently supported!')

        return self

    @staticmethod
    def _statistic(x, z, w, case):
        # Create adjacency list. Note that remove_symmetric=False - this is
        # different from the esda.Join_Counts() function.
        adj_list = w.to_adjlist(remove_symmetric=False)

        # First, set up a series that maps the values
        # to the weights table
        zseries_x = pd.Series(x, index=w.id_order)
        zseries_z = pd.Series(z, index=w.id_order)

        # Map the values to the focal (i) values
        focal_x = zseries_x.loc[adj_list.focal].values
        focal_z = zseries_z.loc[adj_list.focal].values

        # Map the values to the neighbor (j) values
        neighbor_x = zseries_x.loc[adj_list.neighbor].values
        neighbor_z = zseries_z.loc[adj_list.neighbor].values

        if case == "BJC":
            BJC = (focal_x == 1) & (focal_z == 0) & \
                  (neighbor_x == 0) & (neighbor_z == 1)
            adj_list_BJC = pd.DataFrame(adj_list.focal.values,
                                        BJC.astype('uint8')).reset_index()
            adj_list_BJC.columns = ['BJC', 'ID']
            adj_list_BJC = adj_list_BJC.groupby(by='ID').sum()
            return (np.array(adj_list_BJC.BJC.values, dtype='float'))
        elif case == "CLC":
            CLC = (focal_x == 1) & (focal_z == 1) & \
                  (neighbor_x == 1) & (neighbor_z == 1)
            adj_list_CLC = pd.DataFrame(adj_list.focal.values,
                                        CLC.astype('uint8')).reset_index()
            adj_list_CLC.columns = ['CLC', 'ID']
            adj_list_CLC = adj_list_CLC.groupby(by='ID').sum()
            return (np.array(adj_list_CLC.CLC.values, dtype='float'))
        else:
            raise NotImplementedError(f'The requested LJC method ({case}) \
            is not currently supported!')

# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: scaling not used

@_njit(fastmath=True)
def _ljc_bv_case1(i, z, permuted_ids, weights_i, scaling):
    zx = z[:, 0]
    zy = z[:, 1]
    zyi, zyrand = _prepare_univariate(i, zy, permuted_ids, weights_i)
    return zx[i] * (zyrand @ weights_i)

@_njit(fastmath=True)
def _ljc_bv_case2(i, z, permuted_ids, weights_i, scaling):
    zx = z[:, 0]
    zy = z[:, 1]
    zxi, zxrand, zyi, zyrand = _prepare_bivariate(i, z, permuted_ids, weights_i)
    zf = zxrand * zyrand
    return zy[i] * (zf @ weights_i)
