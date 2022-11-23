import numpy as np
import pandas as pd
from libpysal import weights
from sklearn.base import BaseEstimator

from esda.crand import _prepare_univariate
from esda.crand import crand as _crand_plus
from esda.crand import njit as _njit

PERMUTATIONS = 999


class Join_Counts_Local(BaseEstimator):

    """Univariate Local Join Count Statistic"""

    def __init__(
        self,
        connectivity=None,
        permutations=PERMUTATIONS,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,
        drop_islands=True,
    ):
        """
        Initialize a Local_Join_Count estimator

        Parameters
        ----------
        connectivity : scipy.sparse matrix object
            the connectivity structure describing
            the relationships between observed units.
            Need not be row-standardized.
        permutations : int
            number of random permutations for calculation of pseudo
            p_values
        n_jobs : int
            Number of cores to be used in the conditional randomisation. If -1,
            all available cores are used.
        keep_simulations : bool (default True)
           If True, the entire matrix of replications under the null
           is stored in memory and accessible; otherwise, replications
           are not saved
        seed : None/int
           Seed to ensure reproducibility of conditional randomizations.
           Must be set here, and not outside of the function, since numba
           does not correctly interpret external seeds
           nor numpy.random.RandomState instances.
        island_weight:
            value to use as a weight for the "fake" neighbor for every island.
            If numpy.nan, will propagate to the final local statistic depending
            on the `stat_func`. If 0, then the lag is always zero for islands.
        drop_islands : bool (default True)
            Whether or not to preserve islands as entries in the adjacency
            list. By default, observations with no neighbors do not appear
            in the adjacency list. If islands are kept, they are coded as
            self-neighbors with zero weight. See ``libpysal.weights.to_adjlist()``.

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
        self.island_weight = island_weight
        self.drop_islands = drop_islands

    def fit(self, y, n_jobs=1, permutations=999):
        """
        Parameters
        ----------
        y               : numpy.ndarray
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
        >>> y = np.ones(16)
        >>> y[0:8] = 0
        >>> LJC_uni = Local_Join_Count(connectivity=w).fit(y)
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
        y = np.array(y, dtype="float")

        w = self.connectivity
        # Fill the diagonal with 0s
        w = weights.util.fill_diagonal(w, val=0)
        w.transform = "b"

        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs
        # seed = self.seed

        self.y = y
        self.n = len(y)
        self.w = w

        self.LJC = self._statistic(y, w, self.drop_islands)

        if permutations:
            self.p_sim, self.rjoins = _crand_plus(
                z=self.y,
                w=self.w,
                observed=self.LJC,
                permutations=permutations,
                keep=keep_simulations,
                n_jobs=n_jobs,
                stat_func=_ljc_uni,
                island_weight=self.island_weight,
            )
            # Set p-values for those with LJC of 0 to NaN
            self.p_sim[self.LJC == 0] = "NaN"

        return self

    @staticmethod
    def _statistic(y, w, drop_islands):
        # Create adjacency list. Note that remove_symmetric=False - this is
        # different from the esda.Join_Counts() function.
        adj_list = w.to_adjlist(remove_symmetric=False, drop_islands=drop_islands)
        zseries = pd.Series(y, index=w.id_order)
        focal = zseries.loc[adj_list.focal].values
        neighbor = zseries.loc[adj_list.neighbor].values
        LJC = (focal == 1) & (neighbor == 1)
        adj_list_LJC = pd.DataFrame(
            adj_list.focal.values, LJC.astype("uint8")
        ).reset_index()
        adj_list_LJC.columns = ["LJC", "ID"]
        adj_list_LJC = adj_list_LJC.groupby(by="ID").sum()
        LJC = np.array(adj_list_LJC.LJC.values, dtype="float")
        return LJC


# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: scaling not used


@_njit(fastmath=True)
def _ljc_uni(i, z, permuted_ids, weights_i, scaling):
    # self_weight = weights_i[0]
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return zi * (zrand @ other_weights)
