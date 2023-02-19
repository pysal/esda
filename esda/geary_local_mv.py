import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


class Geary_Local_MV(BaseEstimator):

    """Local Geary - Multivariate"""

    def __init__(self, connectivity=None, permutations=999, drop_islands=True):
        """
        Initialize a Local_Geary_MV estimator

        Parameters
        ----------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing
                           the relationships between observed units.
                           Need not be row-standardized.
        permutations     : int
                           (default=999)
                           number of random permutations for calculation
                           of pseudo p_values
        drop_islands : bool (default True)
            Whether or not to preserve islands as entries in the adjacency
            list. By default, observations with no neighbors do not appear
            in the adjacency list. If islands are kept, they are coded as
            self-neighbors with zero weight. See ``libpysal.weights.to_adjlist()``.

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
        self.drop_islands = drop_islands

    def fit(self, variables):
        """
        Parameters
        ----------
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
        self.variables = check_array(
            variables,
            accept_sparse=False,
            dtype="float",
            force_all_finite=True,
            estimator=self,
        )

        w = self.connectivity
        w.transform = "r"

        self.n = len(variables[0])
        self.w = w

        permutations = self.permutations

        # Caclulate z-scores for input variables
        # to be used in _statistic and _crand
        zvariables = stats.zscore(variables, axis=1)

        self.localG = self._statistic(variables, zvariables, w, self.drop_islands)

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
    def _statistic(variables, zvariables, w, drop_islands):
        # Define denominator adjustment
        k = len(variables)
        # Create focal and neighbor values
        adj_list = w.to_adjlist(remove_symmetric=False, drop_islands=drop_islands)
        zseries = [pd.Series(i, index=w.id_order) for i in zvariables]
        focal = [zseries[i].loc[adj_list.focal].values for i in range(len(variables))]
        neighbor = [
            zseries[i].loc[adj_list.neighbor].values for i in range(len(variables))
        ]
        # Carry out local Geary calculation
        gs = adj_list.weight.values * (np.array(focal) - np.array(neighbor)) ** 2
        # Reorganize data
        temp = pd.DataFrame(gs).T
        temp["ID"] = adj_list.focal.values
        adj_list_gs = temp.groupby(by="ID").sum()
        # Rearrange data based on w id order
        adj_list_gs["w_order"] = w.id_order
        adj_list_gs.sort_values(by="w_order", inplace=True)
        adj_list_gs.drop(columns=['w_order'], inplace=True)
        localG = np.array(adj_list_gs.sum(axis=1, numeric_only=True) / k)

        return localG

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
        Gs = np.zeros((self.n, self.permutations))
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
                vars_rand.append(zvariables[j][idsi[rids[:, 0 : wc[i]]]])  # noqa E203
            # vars rand as tmp
            # Calculate diff
            diff = []
            for z in range(nvars):
                _diff = (np.array((zvariables[z][i] - vars_rand[z]) ** 2 * w[i]))
                diff.append(_diff.sum(1) / nvars)
            # add up differences
            temp = np.array([sum(x) for x in zip(*diff)])
            # Assign to object to be returned
            Gs[i] = temp
        self.Gs = Gs
