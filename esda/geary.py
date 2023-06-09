"""
Geary's C statistic for spatial autocorrelation
"""
__author__ = "Serge Rey <sjsrey@gmail.com> "

import warnings

import numpy as np
import scipy.stats as stats
from libpysal import weights

from .tabular import _univariate_handler

__all__ = ["Geary"]


class Geary(object):
    """
    Global Geary C Autocorrelation statistic

    Parameters
    ----------
    y              : array
                     (n, 1) attribute vector
    w              : W
                     spatial weights
    transformation : {'R', 'B', 'D', 'U', 'V'}
                     weights transformation, default is row-standardized.
                     Other options include "B": binary, "D":
                     doubly-standardized, "U": untransformed (general
                     weights), "V": variance-stabilizing.
    permutations   : int
                     number of random permutations for calculation of
                     pseudo-p_values

    Attributes
    ----------
    y              : array
                     original variable
    w              : W
                     spatial weights
    permutations   : int
                     number of permutations
    C              : float
                     value of statistic
    EC             : float
                     expected value
    VC             : float
                     variance of G under normality assumption
    z_norm         : float
                     z-statistic for C under normality assumption
    z_rand         : float
                     z-statistic for C under randomization assumption
    p_norm         : float
                     p-value under normality assumption (one-tailed)
    p_rand         : float
                     p-value under randomization assumption (one-tailed)
    sim            : array
                     (if permutations!=0)
                     vector of I values for permutated samples
    p_sim          : float
                     (if permutations!=0)
                     p-value based on permutations (one-tailed)
                     null: sptial randomness
                     alternative: the observed C is extreme
                     it is either extremely high or extremely low
    EC_sim         : float
                     (if permutations!=0)
                     average value of C from permutations
    VC_sim         : float
                     (if permutations!=0)
                     variance of C from permutations
    seC_sim        : float
                     (if permutations!=0)
                     standard deviation of C under permutations.
    z_sim          : float
                     (if permutations!=0)
                     standardized C based on permutations
    p_z_sim        : float
                     (if permutations!=0)
                     p-value based on standard normal approximation from
                     permutations (one-tailed)

    Examples
    --------
    >>> import libpysal
    >>> from esda.geary import Geary
    >>> w = libpysal.io.open(libpysal.examples.get_path("book.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("book.txt"))
    >>> y = np.array(f.by_col['y'])
    >>> c = Geary(y,w,permutations=0)
    >>> round(c.C,7)
    0.3330108
    >>> round(c.p_norm,7)
    9.2e-05
    >>>


    Notes
    -----
    Technical details and derivations can be found in :cite:`cliff81`.

    """

    def __init__(self, y, w, transformation="r", permutations=999):
        if not isinstance(w, weights.W):
            raise TypeError(
                f"w must be a pysal weights object, got {type(w)} instead."
            )
        y = np.asarray(y).flatten()
        self.n = len(y)
        self.y = y
        w.transform = transformation
        self.w = w
        self._focal_ix, self._neighbor_ix = w.sparse.nonzero()
        self._weights = w.sparse.data
        self.permutations = permutations
        self.__moments()
        xn = range(len(y))
        self.xn = xn
        self.y2 = y * y
        yd = y - y.mean()
        yss = sum(yd * yd)

        self.den = yss * self.w.s0 * 2.0
        self.C = self.__calc(y)
        de = self.C - 1.0
        self.EC = 1.0
        self.z_norm = de / self.seC_norm
        self.z_rand = de / self.seC_rand
        if de > 0:
            self.p_norm = stats.norm.sf(self.z_norm)
            self.p_rand = stats.norm.sf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.y)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.C
            larger = sum(above)
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EC_sim = sum(sim) / permutations
            self.seC_sim = np.array(sim).std()
            self.VC_sim = self.seC_sim**2
            self.z_sim = (self.C - self.EC_sim) / self.seC_sim
            self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))

    @property
    def _statistic(self):
        """a standardized accessor for esda statistics"""
        return self.C

    def __moments(self):
        y = self.y
        n = self.n
        w = self.w
        s0 = w.s0
        s1 = w.s1
        s2 = w.s2
        s02 = s0 * s0
        yd = y - y.mean()
        yd4 = yd**4
        yd2 = yd**2
        n2 = n * n
        k = (yd4.sum() / n) / ((yd2.sum() / n) ** 2)
        A = (n - 1) * s1 * (n2 - 3 * n + 3 - (n - 1) * k)
        B = (1.0 / 4) * ((n - 1) * s2 * (n2 + 3 * n - 6 - (n2 - n + 2) * k))
        C = s02 * (n2 - 3 - (n - 1) ** 2 * k)
        vc_rand = (A - B + C) / (n * (n - 2) * (n - 3) * s02)
        vc_norm = (1 / (2 * (n + 1) * s02)) * ((2 * s1 + s2) * (n - 1) - 4 * s02)

        self.VC_rand = vc_rand
        self.VC_norm = vc_norm
        self.seC_rand = vc_rand ** (0.5)
        self.seC_norm = vc_norm ** (0.5)

    def __calc(self, y):
        num = (self._weights * ((y[self._focal_ix] - y[self._neighbor_ix]) ** 2)).sum()
        a = (self.n - 1) * num
        return a / self.den

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a Geary statistic on a dataframe

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
            operating inplace, with default configurations,
            the derived columns will be named like 'column_geary' and 'column_p_sim'
        pvalue  : string
            a string denoting which pvalue should be returned. Refer to the
            the Geary statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Geary statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the Geary statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        Notes
        -----
        Technical details and derivations can be found in :cite:`cliff81`.

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
            **stat_kws
        )
