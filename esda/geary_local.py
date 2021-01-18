import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator
import libpysal as lp
from esda.crand import (
    crand as _crand_plus,
    njit as _njit,
    _prepare_univariate
)


class Geary_Local(BaseEstimator):

    """Local Geary - Univariate"""

    def __init__(self, connectivity=None, labels=False, sig=0.05,
                 permutations=999, n_jobs=1, keep_simulations=True,
                 seed=None):
        """
        Initialize a Local_Geary estimator
        Arguments
        ---------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing
                           the relationships between observed units.
                           Need not be row-standardized.
        labels           : boolean
                           (default=False)
                           If True use, label if an observation
                           belongs to an outlier, cluster, other,
                           or non-significant group. 1 = outlier,
                           2 = cluster, 3 = other, 4 = non-significant.
                           Note that this is not the exact same as the
                           cluster map produced by GeoDa.
        sig              : float
                           (default=0.05)
                           Default significance threshold used for
                           creation of labels groups.
        permutations     : int
                           (default=999)
                           number of random permutations for calculation
                           of pseudo p_values
        n_jobs           : int
                           (default=1)
                           Number of cores to be used in the conditional
                           randomisation. If -1, all available cores are used.
        keep_simulations : Boolean
                           (default=True)
                           If True, the entire matrix of replications under
                           the null is stored in memory and accessible;
                           otherwise, replications are not saved
        seed             : None/int
                           Seed to ensure reproducibility of conditional
                           randomizations. Must be set here, and not outside
                           of the function, since numba does not correctly
                           interpret external seeds nor
                           numpy.random.RandomState instances.

        Attributes
        ----------
        localG          : numpy array
                          array containing the observed univariate
                          Local Geary values.
        p_sim           : numpy array
                          array containing the simulated
                          p-values for each unit.
        labs            : numpy array
                          array containing the labels for if each observation.
        """

        self.connectivity = connectivity
        self.labels = labels
        self.sig = sig
        self.permutations = permutations
        self.n_jobs = n_jobs
        self.keep_simulations = keep_simulations
        self.seed = seed

    def fit(self, x):
        """
        Arguments
        ---------
        x                : numpy.ndarray
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
        >>> import libpysal as lp
        >>> import geopandas as gpd
        >>> guerry = lp.examples.load_example('Guerry')
        >>> guerry_ds = gpd.read_file(guerry.get_path('Guerry.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(guerry_ds)
        >>> y = guerry_ds['Donatns']
        >>> lG = Local_Geary(connectivity=w).fit(y)
        >>> lG.localG[0:5]
        >>> lG.p_sim[0:5]
        """
        x = np.asarray(x).flatten()

        w = self.connectivity
        w.transform = 'r'
        
        permutations = self.permutations
        sig = self.sig
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs
        seed = self.seed

        self.localG = self._statistic(x, w)

        if permutations:
            self.p_sim, self.rlocalG = _crand_plus(
                z=(x - np.mean(x))/np.std(x),
                w=w,
                observed=self.localG,
                permutations=permutations,
                keep=keep_simulations,
                n_jobs=n_jobs,
                stat_func=_local_geary
            )

        if self.labels:
            Eij_mean = np.mean(self.localG)
            x_mean = np.mean(x)
            # Create empty vector to fill
            self.labs = np.empty(len(x)) * np.nan
            # Outliers
            self.labs[(self.localG < Eij_mean) &
                      (x > x_mean) &
                      (self.p_sim <= sig)] = 1
            # Clusters
            self.labs[(self.localG < Eij_mean) &
                      (x < x_mean) &
                      (self.p_sim <= sig)] = 2
            # Other
            self.labs[(self.localG > Eij_mean) &
                      (self.p_sim <= sig)] = 3
            # Non-significant
            self.labs[self.p_sim > sig] = 4

        return self

    @staticmethod
    def _statistic(x, w):
        # Caclulate z-scores for x
        zscore_x = (x - np.mean(x))/np.std(x)
        # Create focal (xi) and neighbor (zi) values
        adj_list = w.to_adjlist(remove_symmetric=False)
        zseries = pd.Series(zscore_x, index=w.id_order)
        zi = zseries.loc[adj_list.focal].values
        zj = zseries.loc[adj_list.neighbor].values
        # Carry out local Geary calculation
        gs = adj_list.weight.values * (zi-zj)**2
        # Reorganize data
        adj_list_gs = pd.DataFrame(adj_list.focal.values, gs).reset_index()
        adj_list_gs.columns = ['gs', 'ID']
        adj_list_gs = adj_list_gs.groupby(by='ID').sum()

        localG = adj_list_gs.gs.values

        return (localG)

# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: does not using the scaling parameter

@_njit(fastmath=True)
def _local_geary(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return (zi-zrand)**2 @ weights_i
