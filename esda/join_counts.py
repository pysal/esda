"""
Spatial autocorrelation for binary attributes

"""
__author__ = "Serge Rey <sjsrey@gmail.com> , Luc Anselin <luc.anselin@asu.edu>"

from libpysal.weights.spatial_lag import lag_spatial
from .tabular import _univariate_handler
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np
import pandas as pd
import warnings

__all__ = ["Join_Counts"]

PERMUTATIONS = 999


class Join_Counts(object):
    """Binary Join Counts


    Parameters
    ----------

    y               : array
                      binary variable measured across n spatial units
    w               : W
                      spatial weights instance
    permutations    : int
                      number of random permutations for calculation of pseudo-p_values

    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of permutations
    bb           : float
                   number of black-black joins
    ww           : float
                   number of white-white joins
    bw           : float
                   number of black-white joins
    J            : float
                   number of joins
    sim_bb       : array
                   (if permutations>0)
                   vector of bb values for permuted samples
    p_sim_bb     : array
                  (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed bb is greater than under randomness
    mean_bb      : float
                   average of permuted bb values
    min_bb       : float
                   minimum of permuted bb values
    max_bb       : float
                   maximum of permuted bb values
    sim_bw       : array
                   (if permutations>0)
                   vector of bw values for permuted samples
    p_sim_bw     : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed bw is greater than under randomness
    mean_bw      : float
                   average of permuted bw values
    min_bw       : float
                   minimum of permuted bw values
    max_bw       : float
                   maximum of permuted bw values
    pos          : float
                   bb+ww
    p_sim_pos    : float
                   p-value based on permutations (one-sided) for pos
    crosstab     : DataFrame
                   Contingency table for observed join counts
    expected     : DataFrame
                   Expected contingency table under the null
    chi2         : float
                   Observed value of chi2 for join count contingency table (see Notes).
    p_sim_chi2   : float
                   p-value for chi2 under random spatial permutations


    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> w = libpysal.weights.lat2W(4, 4)
    >>> y = np.ones(16)
    >>> y[0:8] = 0
    >>> np.random.seed(12345)
    >>> from esda.join_counts import Join_Counts
    >>> jc = Join_Counts(y, w)
    >>> jc.bb
    10.0
    >>> jc.bw
    4.0
    >>> jc.ww
    10.0
    >>> jc.J
    24.0
    >>> len(jc.sim_bb)
    999
    >>> round(jc.p_sim_bb, 3)
    0.003
    >>> round(np.mean(jc.sim_bb), 3)
    5.547
    >>> np.max(jc.sim_bb)
    10.0
    >>> np.min(jc.sim_bb)
    0.0
    >>> len(jc.sim_bw)
    999
    >>> jc.p_sim_bw
    1.0
    >>> np.mean(jc.sim_bw)
    12.811811811811811
    >>> np.max(jc.sim_bw)
    24.0
    >>> np.min(jc.sim_bw)
    7.0
    >>> jc.p_sim_chi2
    0.008
    >>> jc.pos
    20.0
    >>> jc.p_sim_pos
    0.001

    Notes
    -----

    Analytical inference using the chi2 is approximate and is thus not used.
    The independence assumption is clearly violated for join counts even
    if the data is free from spatial autocorrelation as neighboring join counts
    will be correlated by construction. Thus only, the chi2 attribute is
    reported, no analytical p-values are reported.

    Instead, `p_sim_chi2` is reported which uses the sampling distribution of
    the chi2 statistic under the null based on random spatial permutations of
    the data.

    Warnings will be issued when zero values for specific expected values of
    join counts are encountered in the sample or when carrying out the
    permutations. In the former case, no inference related attributes are set
    on the object, while in the latter, realizations with zero expected counts
    are not used in constructing the sampling distribution for the chi2
    statistic.

    Technical details and derivations can be found in :cite:`cliff81`.
    """

    def __init__(self, y, w, permutations=PERMUTATIONS):
        y = np.asarray(y).flatten()
        w.transformation = "b"  # ensure we have binary weights
        self.w = w
        self.adj_list = self.w.to_adjlist() # full symmetry needed for tables
        self.y = y
        self.permutations = permutations
        self.J = w.s0 / 2.0
        results = self.__calc(self.y)
        if results:
            self.bb = results[0]
            self.ww = results[1]
            self.bw = results[2]
            self.pos = self.bb + self.ww
            self.neg = self.bw # bw==wb
            self.chi2 = results[3]
            crosstab = pd.DataFrame(data=results[-2])
            id_names = ["W", "B"]
            idx = pd.Index(id_names, name="Focal")
            crosstab.set_index(idx, inplace=True)
            crosstab.columns = pd.Index(id_names, name="Neighbor")
            self.crosstab = crosstab
            expected = pd.DataFrame(data=results[-1])
            expected.set_index(idx, inplace=True)
            expected.columns = pd.Index(id_names, name="Neighbor")
            self.expected = expected
            self.calc = self.__calc

            if permutations:
                sim = []
                i = 0
                while i < permutations:
                    try:
                        res = self.__calc(np.random.permutation(self.y))
                        sim.append(res)
                        i += 1
                    except ValueError:
                        warnings.warn('Zero expected joins encountered, ignoring realization.')
                        pass
                sim_jc = np.array(sim)
                self.sim_bb = sim_jc[:, 0]
                self.sim_ww = sim_jc[:, 1]
                self.sim_pos = self.sim_bb + self.sim_ww
                self.min_bb = np.min(self.sim_bb)
                self.mean_bb = np.mean(self.sim_bb)
                self.max_bb = np.max(self.sim_bb)
                self.sim_bw = sim_jc[:, 2]
                self.sim_neg = self.sim_bw
                self.min_bw = np.min(self.sim_bw)
                self.mean_bw = np.mean(self.sim_bw)
                self.max_bw = np.max(self.sim_bw)
                self.sim_chi2 = sim_jc[:, 3]
                self.p_sim_bb = self.__pseudop(self.sim_bb, self.bb)
                self.p_sim_bw = self.__pseudop(self.sim_bw, self.bw)
                self.p_sim_ww = self.__pseudop(self.sim_ww, self.ww)
                self.p_sim_pos = self.__pseudop(self.sim_pos, self.pos)
                self.p_sim_neg = self.__pseudop(self.sim_neg, self.neg)
                self.p_sim_chi2 = self.__pseudop(self.sim_chi2, self.chi2)

    def __calc(self, z):
        adj_list = self.adj_list
        zseries = pd.Series(z, index=self.w.id_order)
        focal = zseries.loc[adj_list.focal].values
        neighbor = zseries.loc[adj_list.neighbor].values
        sim = focal == neighbor
        dif = 1 - sim
        bb = (focal * sim).sum() / 2.
        ww = ((1 - focal) * sim).sum() / 2.
        bw = (focal * dif).sum() / 2.
        wb = ((1 - focal) * dif).sum() / 2.
        table = [[ww, wb], [bw, bb]]
        try:
            chi2 = chi2_contingency(table)
        except ValueError:
            msg = 'Zero expected join count encountered. No inference made.'
            msg += str(table)
            warnings.warn(msg)
            return None
        stat, pvalue, dof, expected = chi2
        return (bb, ww, bw + wb, stat, np.array(table), expected)

    def __pseudop(self, sim, jc):
        above = sim >= jc
        larger = sum(above)
        psim = (larger + 1.0) / (self.permutations + 1.0)
        return psim

    @property
    def _statistic(self):
        return self.bw

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a Join_Count statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        cols        :   string or list of string
                        name or list of names of columns to use to compute the statistic
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_join_count'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Join_Count statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the
                        Join_Count statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Join_Count statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        """
        if outvals is None:
            outvals = []
            outvals.extend(["bb", "p_sim_bw", "p_sim_bb"])
            pvalue = ""
        return _univariate_handler(
            df,
            cols,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            stat=cls,
            swapname="bw",
            **stat_kws
        )
