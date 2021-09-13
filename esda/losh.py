import numpy as np
import warnings
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator
import libpysal as lp


class LOSH(BaseEstimator):
    """Local spatial heteroscedasticity (LOSH)"""

    def __init__(self, connectivity=None, inference=None):
        """
        Initialize a losh estimator

        Arguments
        ---------
        connectivity     : scipy.sparse matrix object
                           the connectivity structure describing the
                           relationships between observed units.
        inference        : str
                           describes type of inference to be used. options are
                           "chi-square" or "permutation" methods.

        Attributes
        ----------
        Hi               : numpy array
                           Array of LOSH values for each spatial unit.
        ylag             : numpy array
                           Spatially lagged y values.
        yresid           : numpy array
                           Spatially lagged residual values.
        VarHi            : numpy array
                           Variance of Hi.
        pval             : numpy array
                           P-values for inference based on either
                           "chi-square" or "permutation" methods.
        """

        self.connectivity = connectivity
        self.inference = inference

    def fit(self, y, a=2):
        """
        Arguments
        ---------
        y                : numpy.ndarray
                           array containing continuous data
        a                : int
                           residual multiplier. Default is 2 in order
                           to generate a variance measure. Users may
                           use 1 for absolute deviations.

        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations can be found in :cite:`OrdGetis2012`.

        Examples
        --------
        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        >>> y = np.array(f.by_col['HR8893'])
        >>> from esda import losh
        >>> ls = losh(connectivity=w, inference="chi-square").fit(y)
        >>> np.round(ls.Hi[0], 3)
        >>> np.round(ls.pval[0], 3)

        Boston housing data replicating R spdep::LOSH()
        >>> import libpysal
        >>> import geopandas as gpd
        >>> boston = libpysal.examples.load_example('Bostonhsg')
        >>> boston_ds = gpd.read_file(boston.get_path('boston.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(boston_ds)
        >>> ls = losh(connectivity=w, inference="chi-square").fit(boston['NOX'])
        >>> np.round(ls.Hi[0], 3)
        >>> np.round(ls.VarHi[0], 3)
        """
        y = np.asarray(y).flatten()

        w = self.connectivity

        self.Hi, self.ylag, self.yresid, self.VarHi = self._stat_func(y, w, a)

        if self.inference is None:
            return self
        elif self.inference == "chi-square":
            if a != 2:
                warnings.warn(
                    f"Chi-square inference assumes that a=2, but \
                a={a}. This means the inference will be invalid!"
                )
            else:
                dof = 2 / self.VarHi
                Zi = (2 * self.Hi) / self.VarHi
                self.pval = 1 - stats.chi2.cdf(Zi, dof)
        else:
            raise NotImplementedError(
                f"The requested inference method \
            ({self.inference}) is not currently supported!"
            )

        return self

    @property
    def _statistic(self):
        return self.Hi

    @staticmethod
    def _stat_func(y, w, a):
        # Define what type of variance to use
        if a is None:
            a = 2
        else:
            a = a

        rowsum = np.array(w.sparse.sum(axis=1)).flatten()

        # Calculate spatial mean
        ylag = lp.weights.lag_spatial(w, y) / rowsum
        # Calculate and adjust residuals based on multiplier
        yresid = abs(y - ylag) ** a
        # Calculate denominator of Hi equation
        denom = np.mean(yresid) * np.array(rowsum)
        # Carry out final Hi calculation
        Hi = lp.weights.lag_spatial(w, yresid) / denom
        # Calculate average of residuals
        yresid_mean = np.mean(yresid)
        # Calculate VarHi
        n = len(y)
        squared_rowsum = np.asarray(w.sparse.multiply(w.sparse).sum(axis=1)).flatten()

        VarHi = (
            ((n - 1) ** -1)
            * (denom ** -2)
            * ((np.sum(yresid ** 2) / n) - yresid_mean ** 2)
            * ((n * squared_rowsum) - (rowsum ** 2))
        )

        return (Hi, ylag, yresid, VarHi)
