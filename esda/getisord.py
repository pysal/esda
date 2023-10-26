"""
Getis and Ord G statistic for spatial autocorrelation
"""
__author__ = "Sergio J. Rey <srey@asu.edu>, Myunghwa Hwang <mhwang4@gmail.com> "
__all__ = ["G", "G_Local"]

import warnings

import numpy as np
from scipy import stats

from libpysal.weights.spatial_lag import lag_spatial as slag
from libpysal.weights.util import fill_diagonal

from .crand import _prepare_univariate
from .crand import crand as _crand_plus
from .crand import njit as _njit
from .tabular import _univariate_handler

PERMUTATIONS = 999


class G(object):
    """
    Global G Autocorrelation Statistic

    Parameters
    ----------
    y             : array (n,1)
                    Attribute values
    w             : W
                   DistanceBand W spatial weights based on distance band
    permutations  : int
                    the number of random permutations for calculating pseudo p_values

    Attributes
    ----------
    y : array
        original variable
    w : W
        DistanceBand W spatial weights based on distance band
    permutation : int
        the number of permutations
    G : float
        the value of statistic
    EG : float
        the expected value of statistic
    VG : float
        the variance of G under normality assumption
    z_norm : float
        standard normal test statistic
    p_norm : float
        p-value under normality assumption (one-sided)
    sim : array
        (if permutations > 0)
        vector of G values for permutated samples
    p_sim : float
        p-value based on permutations (one-sided)
        null: spatial randomness
        alternative: the observed G is extreme it is either
        extremely high or extremely low
    EG_sim : float
        average value of G from permutations
    VG_sim : float
        variance of G from permutations
    seG_sim : float
        standard deviation of G under permutations.
    z_sim : float
        standardized G based on permutations
    p_z_sim : float
        p-value based on standard normal approximation from permutations (one-sided)

    Notes
    -----
    Moments are based on normality assumption.

    For technical details see :cite:`Getis_2010` and :cite:`Ord_2010`.


    Examples
    --------
    >>> import libpysal
    >>> import numpy
    >>> numpy.random.seed(10)

    Preparing a point data set

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points

    >>> w = libpysal.weights.DistanceBand(points,threshold=15)
    >>> w.transform = "B"

    Preparing a variable

    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord G test

    >>> from esda.getisord import G
    >>> g = G(y,w)

    Examining the results

    >>> round(g.G, 3)
    0.557

    >>> round(g.p_norm, 3)
    0.173

    """

    def __init__(self, y, w, permutations=PERMUTATIONS):
        y = np.asarray(y).flatten()
        self.n = len(y)
        self.y = y
        w.transform = "B"
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.y2 = y * y
        y = y.reshape(
            len(y), 1
        )  # Ensure that y is an n by 1 vector, otherwise y*y.T == y*y
        self.den_sum = (y * y.T).sum() - (y * y).sum()
        self.G = self.__calc(self.y)
        self.z_norm = (self.G - self.EG) / np.sqrt(self.VG)
        self.p_norm = 1.0 - stats.norm.cdf(np.abs(self.z_norm))

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.y)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.G
            larger = sum(above)
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EG_sim = sum(sim) / permutations
            self.seG_sim = sim.std()
            self.VG_sim = self.seG_sim**2
            self.z_sim = (self.G - self.EG_sim) / self.seG_sim
            self.p_z_sim = 1.0 - stats.norm.cdf(np.abs(self.z_sim))

    def __moments(self):
        y = self.y
        n = self.n
        w = self.w
        n2 = n * n
        s0 = w.s0
        self.EG = s0 / (n * (n - 1))
        s02 = s0 * s0
        s1 = w.s1
        s2 = w.s2
        b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
        b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
        b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
        b4 = s1 - s2 + s02
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        y2 = y * y
        y3 = y * y2
        y4 = y2 * y2
        EG2 = b0 * (sum(y2) ** 2) + b1 * sum(y4) + b2 * (sum(y) ** 2) * sum(y2)
        EG2 += b3 * sum(y) * sum(y3) + b4 * (sum(y) ** 4)
        EG2NUM = EG2
        EG2DEN = ((sum(y) ** 2 - sum(y2)) ** 2) * n * (n - 1) * (n - 2) * (n - 3)
        self.EG2 = EG2NUM / EG2DEN
        self.VG = self.EG2 - self.EG**2

    def __calc(self, y):
        yl = slag(self.w, y)
        self.num = y * yl
        return self.num.sum() / self.den_sum

    @property
    def _statistic(self):
        """Standardized accessor for esda statistics"""
        return self.G

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a G statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        cols : string or list of string
            name or list of names of columns to use to compute the statistic
        w : pysal weights object
            a weights object aligned with the dataframe. If not provided, this
            is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named 'column_g'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the G statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the G statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the G statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """

        msg = (
            "The `.by_cols()` methods are deprecated and will be "
            "removed in a future version of `esda`."
        )
        warnings.warn(msg, FutureWarning)

        return _univariate_handler(
            df,
            cols,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            stat=cls,
            swapname=cls.__name__.lower(),
            **stat_kws,
        )


class G_Local(object):
    """
    Generalized Local G Autocorrelation

    Parameters
    ----------
    y : array
        variable
    w : W
        DistanceBand, weights instance that is based on threshold distance
        and is assumed to be aligned with y
    transform : {'R', 'B'}
        the type of w, either 'B' (binary) or 'R' (row-standardized)
    permutations : int
        the number of random permutations for calculating
        pseudo p values
    star : boolean or float
        whether or not to include focal observation in sums (default: False)
        if the row-transformed weight is provided, then this is the default
        value to use within the spatial lag. Generally, weights should be
        provided in binary form, and standardization/self-weighting will be
        handled by the function itself.
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Attributes
    ----------
    y : array
       original variable
    w : DistanceBand W
       original weights object
    permutations : int
                  the number of permutations
    Gs : array
        of floats, the value of the orginal G statistic in Getis & Ord (1992)
    EGs : float
         expected value of Gs under normality assumption
         the values is scalar, since the expectation is identical
         across all observations
    VGs : array
         of floats, variance values of Gs under normality assumption
    Zs : array
        of floats, standardized Gs
    p_norm : array
            of floats, p-value under normality assumption (one-sided)
            for two-sided tests, this value should be multiplied by 2
    sim : array
         of arrays of floats (if permutations>0), vector of I values
         for permutated samples
    p_sim : array
        of floats, p-value based on permutations (one-sided)
        null - spatial randomness
        alternative - the observed G is extreme
            (it is either extremely high or extremely low)
    EG_sim : array
            of floats, average value of G from permutations
    VG_sim : array
            of floats, variance of G from permutations
    seG_sim : array
             of floats, standard deviation of G under permutations.
    z_sim : array
           of floats, standardized G based on permutations
    p_z_sim : array
             of floats, p-value based on standard normal approximation from
             permutations (one-sided)

    Notes
    -----
    To compute moments of Gs under normality assumption,
    PySAL considers w is either binary or row-standardized.
    For binary weights object, the weight value for self is 1
    For row-standardized weights object, the weight value for self is
    1/(the number of its neighbors + 1).


    For technical details see :cite:`Getis_2010` and :cite:`Ord_2010`.


    Examples
    --------
    >>> import libpysal
    >>> import numpy
    >>> numpy.random.seed(10)

    Preparing a point data set

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points

    >>> w = libpysal.weights.DistanceBand(points,threshold=15)

    Preparing a variable

    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord local G test using a binary weights object

    >>> from esda.getisord import G_Local
    >>> lg = G_Local(y,w,transform='B')

    Examining the results

    >>> lg.Zs
    array([-1.0136729 , -0.04361589,  1.31558703, -0.31412676,  1.15373986,
            1.77833941])
    >>> round(lg.p_sim[0], 3)
    0.101

    p-value based on standard normal approximation from permutations
    >>> round(lg.p_z_sim[0], 3)
    0.154

    >>> numpy.random.seed(10)

    Applying Getis and Ord local G* test using a binary weights object

    >>> lg_star = G_Local(y,w,transform='B',star=True)

    Examining the results

    >>> lg_star.Zs
    array([-1.39727626, -0.28917762,  0.65064964, -0.28917762,  1.23452088,
            2.02424331])
    >>> round(lg_star.p_sim[0], 3)
    0.101

    >>> numpy.random.seed(12345)

    Applying Getis and Ord local G test using a row-standardized weights object

    >>> lg = G_Local(y,w,transform='R')

    Examining the results

    >>> lg.Zs
    array([-0.62074534, -0.01780611,  1.31558703, -0.12824171,  0.28843496,
            1.77833941])
    >>> round(lg.p_sim[0], 3)
    0.103

    >>> numpy.random.seed(10)

    Applying Getis and Ord local G* test using a row-standardized weights object

    >>> lg_star = G_Local(y,w,transform='R',star=True)

    Examining the results

    >>> lg_star.Zs
    array([-0.62488094, -0.09144599,  0.41150696, -0.09144599,  0.24690418,
            1.28024388])
    >>> round(lg_star.p_sim[0], 3)
    0.101

    """

    def __init__(
        self,
        y,
        w,
        transform="R",
        permutations=PERMUTATIONS,
        star=False,
        keep_simulations=True,
        n_jobs=-1,
        seed=None,
        island_weight=0,
    ):
        y = np.asarray(y).flatten()
        self.n = len(y)
        self.y = y
        w, star = _infer_star_and_structure_w(w, star, transform)
        w.transform = transform
        self.w_transform = transform
        self.w = w
        self.permutations = permutations
        self.star = star
        self.calc()
        self.p_norm = stats.norm.sf(np.abs(self.Zs))
        if permutations:
            self.p_sim, self.rGs = _crand_plus(
                y,
                w,
                self.Gs,
                permutations,
                keep_simulations,
                n_jobs=n_jobs,
                stat_func=_g_local_star_crand if star else _g_local_crand,
                scaling=y.sum(),
                seed=seed,
                island_weight=island_weight,
            )
            if keep_simulations:
                self.sim = sim = self.rGs.T
                self.EG_sim = sim.mean(axis=0)
                self.seG_sim = sim.std(axis=0)
                self.VG_sim = self.seG_sim * self.seG_sim
                self.z_sim = (self.Gs - self.EG_sim) / self.seG_sim
                self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))

    def __crand(self, keep_simulations):
        y = self.y
        if keep_simulations:
            rGs = np.zeros((self.n, self.permutations))
        larger = np.zeros((self.n,))
        n_1 = self.n - 1
        rid = list(range(n_1))
        prange = list(range(self.permutations))
        k = self.w.max_neighbors + 1
        rids = np.array([np.random.permutation(rid)[0:k] for i in prange])
        ids = np.arange(self.w.n)
        wc = self.__getCardinalities()
        if self.w_transform == "r":
            den = np.array(wc) + self.star
        else:
            den = np.ones(self.w.n)
        for i in range(self.w.n):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            yi_star = y[i] * self.star
            wci = wc[i]
            rGs_i = (y[idsi[rids[:, 0:wci]]]).sum(1) + yi_star
            rGs_i = (np.array(rGs_i) / den[i]) / (self.y_sum - (1 - self.star) * y[i])
            if keep_simulations:
                rGs[i] = rGs_i
            larger[i] = (rGs_i >= self.Gs[i]).sum()
        if keep_simulations:
            self.rGs = rGs
        below = (self.permutations - larger) < larger
        larger[below] = self.permutations - larger[below]
        self.p_sim = (larger + 1) / (self.permutations + 1)

    def __getCardinalities(self):
        ido = self.w.id_order
        self.wc = np.array([self.w.cardinalities[ido[i]] for i in range(self.n)])
        return self.wc

    def calc(self):
        w = self.w
        W = w.sparse

        self.y_sum = self.y.sum()

        y = self.y
        remove_self = not self.star
        N = self.w.n - remove_self

        statistic = (W @ y) / (y.sum() - y * remove_self)

        # ----------------------------------------------------#
        # compute moments necessary for analytical inference  #
        # ----------------------------------------------------#

        empirical_mean = (y.sum() - y * remove_self) / N
        # variance looks complex, yes, but it obtains from E[x^2] - E[x]^2.
        # So, break it down to allow subtraction of the self-neighbor.
        mean_of_squares = ((y**2).sum() - (y**2) * remove_self) / N
        empirical_variance = mean_of_squares - empirical_mean**2

        # Since we have corrected the diagonal, this should work
        cardinality = np.asarray(W.sum(axis=1)).squeeze()
        expected_value = cardinality / N

        expected_variance = cardinality * (N - cardinality)
        expected_variance /= N - 1
        expected_variance *= 1 / N**2
        expected_variance *= empirical_variance / (empirical_mean**2)

        z_scores = (statistic - expected_value) / np.sqrt(expected_variance)

        self.Gs = statistic
        self.EGs = expected_value
        self.VGs = expected_variance
        self.Zs = z_scores

    @property
    def _statistic(self):
        """Standardized accessor for esda statistics"""
        return self.Gs

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a G_Local statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        cols : string or list of string
            name or list of names of columns to use to compute the statistic
        w : pysal weights object
            a weights object aligned with the dataframe. If not provided, this
            is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named 'column_g_local'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the G_Local statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            G_Local statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the G_Local statistic.

        Returns
        -------
        pandas.DataFrame
                        If inplace, None, and operation is conducted on dataframe
                        in memory. Otherwise, returns a copy of the dataframe with
                        the relevant columns attached.

        """

        msg = (
            "The `.by_col()` methods are deprecated and will be "
            "removed in a future version of `esda`."
        )
        warnings.warn(msg, FutureWarning)

        return _univariate_handler(
            df,
            cols,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            stat=cls,
            swapname=cls.__name__.lower(),
            **stat_kws,
        )


def _infer_star_and_structure_w(weights, star, transform):
    assert transform.lower() in ("r", "b"), (
        f'Transforms must be binary "b" or row-standardized "r".'
        f"Recieved: {transform}"
    )
    adj_matrix = weights.sparse
    diagonal = adj_matrix.diagonal()
    zero_diagonal = (diagonal == 0).all()

    # Gi has a zero diagonal, Gi* has a nonzero diagonal
    star = (not zero_diagonal) if star is None else star

    # Want zero diagonal but do not have it
    if (not zero_diagonal) & (star is False):
        weights = fill_diagonal(weights, 0)
    # Want nonzero diagonal and have it
    elif (not zero_diagonal) & (star is True):
        weights = weights
    # Want zero diagonal and have it
    elif zero_diagonal & (star is False):
        weights = weights
    # Want nonzero diagonal and do not have it
    elif zero_diagonal & (star is True):
        # if the input is binary or requested transform is binary,
        # set the diagonal to 1.
        if transform.lower() == "b" or weights.transform.lower() == "b":
            weights = fill_diagonal(weights, 1)
        # if we know the target is row-standardized, use the row max
        # this works successfully for effectively binary but "O"-transformed input
        elif transform.lower() == "r":
            # This warning is presented in the documentation as well
            warnings.warn(
                "Gi* requested, but (a) weights are already row-standardized,"
                " (b) no weights are on the diagonal, and"
                " (c) no default value supplied to star. Assuming that the"
                " self-weight is equivalent to the maximum weight in the"
                " row. To use a different default (like, .5), set `star=.5`,"
                " or use libpysal.weights.fill_diagonal() to set the diagonal"
                " values of your weights matrix and use `star=None` in Gi_Local."
            )
            weights = fill_diagonal(
                weights, np.asarray(adj_matrix.max(axis=1).todense()).flatten()
            )
    else:  # star was something else, so try to fill the weights with it
        try:
            weights = fill_diagonal(weights, star)
        except TypeError:
            raise TypeError(
                f"Type of star ({type(star)}) not understood."
                f" Must be an integer, boolean, float, or numpy.ndarray."
            )
    star = (weights.sparse.diagonal() > 0).any()
    weights.transform = transform

    return weights, star


# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------


@_njit(fastmath=True)
def _g_local_crand(i, z, permuted_ids, weights_i, scaling):
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zrand @ other_weights) / (scaling - zi)


@_njit(fastmath=True)
def _g_local_star_crand(i, z, permuted_ids, weights_i, scaling):
    self_weight = weights_i[0]
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zrand @ other_weights + self_weight * zi) / scaling


if __name__ == "__main__":

    import geopandas
    import numpy
    from libpysal import examples, weights

    import esda

    df = geopandas.read_file(examples.get_path("NAT.shp"))

    w = weights.Rook.from_dataframe(df)

    for transform in ("r", "b"):
        for star in (True, False):
            test = esda.getisord.G_Local(df.GI89, w, transform=transform, star=star)
            out = test._calc2()
            (
                statistic,
                expected_value,
                expected_variance,
                z_scores,
                empirical_mean,
                empirical_variance,
            ) = out

            numpy.testing.assert_allclose(statistic, test.Gs)
            numpy.testing.assert_allclose(expected_value, test.EGs)
            numpy.testing.assert_allclose(expected_variance, test.VGs)
            numpy.testing.assert_allclose(z_scores, test.Zs)
            numpy.testing.assert_allclose(empirical_mean, test.yl_mean)
            numpy.testing.assert_allclose(empirical_variance, test.s2)

    # Also check that the None configuration works
    test = esda.getisord.G_Local(df.GI89, w, star=None)
