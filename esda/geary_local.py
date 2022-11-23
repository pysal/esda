import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from esda.crand import _prepare_univariate
from esda.crand import crand as _crand_plus
from esda.crand import njit as _njit


class Geary_Local(BaseEstimator):

    """Local Geary - Univariate"""

    def __init__(
        self,
        connectivity=None,
        labels=False,
        sig=0.05,
        permutations=999,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,
        drop_islands=True,
    ):
        """
        Initialize a Local_Geary estimator

        Parameters
        ----------
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
        island_weight :
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
        self.island_weight = island_weight
        self.drop_islands = drop_islands

    def fit(self, x):
        """
        Parameters
        ----------
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
        w.transform = "r"

        permutations = self.permutations
        sig = self.sig
        keep_simulations = self.keep_simulations
        n_jobs = self.n_jobs

        self.localG = self._statistic(x, w, self.drop_islands)

        if permutations:
            self.p_sim, self.rlocalG = _crand_plus(
                z=(x - np.mean(x)) / np.std(x),
                w=w,
                observed=self.localG,
                permutations=permutations,
                keep=keep_simulations,
                n_jobs=n_jobs,
                stat_func=_local_geary,
                island_weight=self.island_weight,
            )

        if self.labels:
            Eij_mean = np.mean(self.localG)
            x_mean = np.mean(x)
            # Create empty vector to fill
            self.labs = np.empty(len(x)) * np.nan
            # Outliers
            locg_lt_eij = self.localG < Eij_mean
            p_leq_sig = self.p_sim <= sig
            self.labs[locg_lt_eij & (x > x_mean) & p_leq_sig] = 1
            # Clusters
            self.labs[locg_lt_eij & (x < x_mean) & p_leq_sig] = 2
            # Other
            self.labs[(self.localG > Eij_mean) & p_leq_sig] = 3
            # Non-significant
            self.labs[self.p_sim > sig] = 4

        return self

    @staticmethod
    def _statistic(x, w, drop_islands):
        # Caclulate z-scores for x
        zscore_x = (x - np.mean(x)) / np.std(x)
        # Create focal (xi) and neighbor (zi) values
        adj_list = w.to_adjlist(remove_symmetric=False, drop_islands=drop_islands)
        zseries = pd.Series(zscore_x, index=w.id_order)
        zi = zseries.loc[adj_list.focal].values
        zj = zseries.loc[adj_list.neighbor].values
        # Carry out local Geary calculation
        gs = adj_list.weight.values * (zi - zj) ** 2
        # Reorganize data
        adj_list_gs = pd.DataFrame(adj_list.focal.values, gs).reset_index()
        adj_list_gs.columns = ["gs", "ID"]
        adj_list_gs = adj_list_gs.groupby(by="ID").sum()
        # Rearrange data based on w id order
        adj_list_gs["w_order"] = w.id_order
        adj_list_gs.sort_values(by="w_order", inplace=True)

        localG = adj_list_gs.gs.values

        return localG


# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------

# Note: does not using the scaling parameter


@_njit(fastmath=True)
def _local_geary(i, z, permuted_ids, weights_i, scaling):
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zi - zrand) ** 2 @ other_weights
