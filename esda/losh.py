import numpy as np
import warnings
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator
import libpysal as lp


class LOSH(BaseEstimator):
    """Local spatial heteroscedasticity (LOSH)"""

    def __init__(self, connectivity=None, inference=None, a=2):
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
        a                : int
                           residual multiplier. Default is 2 in order
                           to generate a variance measure. Users may
                           use 1 for absolute deviations.

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
        self.a = a

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
        Technical details and derivations can be found in :cite:`OrdGetis2012`.

        Examples
        --------
        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        >>> x = np.array(f.by_col['HR8893'])
        >>> from esda import losh
        >>> ls = losh(connectivity=w, inference="chi-square").fit(x)
        >>> np.round(ls.Hi[0], 3)
        >>> np.round(ls.pval[0], 3)

        Boston housing data replicating R spdep::LOSH()
        >>> import libpysal
        >>> boston = libpysal.examples.load_example('Bostonhsg')
        >>> boston_ds = geopandas.read_file(boston.get_path('boston.shp'))
        >>> w = libpysal.weights.Queen.from_dataframe(boston_ds)
        >>> ls = losh(connectivity=w, inference="chi-square").fit(boston['NOX'])
        >>> np.round(ls.Hi[0], 3)
        >>> np.round(ls.VarHi[0], 3)
        """
        x = np.asarray(x).flatten()

        w = self.connectivity

        a = self.a

        self.Hi, self.ylag, self.yresid, self.VarHi = self._statistic(x, w, a)

        if self.inference is None:
            return self
        elif self.inference == 'chi-square':
            if a != 2:
                warnings.warn(f'Chi-square inference assumes that a=2, but \
                a={a}. This means the inference will be invalid!')
            else:
                dof = 2/self.VarHi
                Zi = (2*self.Hi)/self.VarHi
                self.pval = 1 - stats.chi2.cdf(Zi, dof)
        else:
            raise NotImplementedError(f'The requested inference method \
            ({self.inference}) is not currently supported!')

        return self

    @staticmethod
    def _statistic(x, w, a):
        # Define what type of variance to use
        if a is None:
            a = 2
        else:
            a = a

        rowsum = np.array(w.sparse.sum(axis=1)).flatten()

        # Calculate spatial mean
        ylag = lp.weights.lag_spatial(w, x)/rowsum
        # Calculate and adjust residuals based on multiplier
        yresid = abs(x-ylag)**a
        # Calculate denominator of Hi equation
        denom = np.mean(yresid) * np.array(rowsum)
        # Carry out final Hi calculation
        Hi = lp.weights.lag_spatial(w, yresid) / denom
        # Calculate average of residuals
        yresid_mean = np.mean(yresid)
        # Calculate VarHi
        n = len(x)
        squared_rowsum = np.asarray(w.sparse.multiply(w.sparse).sum(axis=1)).flatten()

        VarHi = ((n-1)**-1) * \
                (denom**-2) * \
                ((np.sum(yresid**2)/n) - yresid_mean**2) * \
                ((n*squared_rowsum) - (rowsum**2))

        return (Hi, ylag, yresid, VarHi)