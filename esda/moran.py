"""
Moran's I Spatial Autocorrelation Statistics

"""

__author__ = (
    "Sergio J. Rey <srey@asu.edu>, "
    "Dani Arribas-Bel <daniel.arribas.bel@gmail.com>, "
    "Levi John Wolf <levi.john.wolf@gmail.com>"
)

from warnings import simplefilter, warn

import numpy as np
import pandas as pd
import scipy.stats as stats
from libpysal.weights import W
from libpysal.weights.spatial_lag import lag_spatial
from scipy import sparse

from .crand import _prepare_univariate
from .crand import crand as _crand_plus
from .crand import njit as _njit
from .smoothing import assuncao_rate
from .tabular import _bivariate_handler, _univariate_handler

__all__ = [
    "Moran",
    "Moran_Local",
    "Moran_BV",
    "Moran_BV_matrix",
    "Moran_Local_BV",
    "Moran_Rate",
    "Moran_Local_Rate",
    "plot_moran_facet",
]

PERMUTATIONS = 999


def _slag(w, y):
    """Helper to compute lag either for W or for Graph"""
    if isinstance(w, W):
        return lag_spatial(w, y)
    else:
        return w.lag(y)


def _transform(w, transformation):
    """Helper to transform W or Graph"""
    if isinstance(w, W):
        w.transform = transformation
        return w
    else:
        return w.transform(transformation)


class Moran:
    """Moran's I Global Autocorrelation Statistic

    Parameters
    ----------

    y               : array
                      variable measured across n spatial units
    w               : W | Graph
                      spatial weights instance as W or Graph aligned with y
    transformation  : {'R', 'B', 'D', 'U', 'V'}
                  weights transformation, default is row-standardized "r".
                  Other options include
                  "B": binary,
                  "D": doubly-standardized,
                  "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
                  "V": variance-stabilizing.
    permutations    : int
                      number of random permutations for calculation of
                      pseudo-p_values
    two_tailed      : boolean
                      If True (default) analytical p-values for Moran are two
                      tailed, otherwise if False, they are one-tailed.

    Attributes
    ----------
    y            : array
                   original variable
    w            : W | Graph
                   original w object
    z            : array
                   zero-mean, unit standard deviation normalized y
    permutations : int
                   number of permutations
    I            : float
                   value of Moran's I
    EI           : float
                   expected value under normality assumption
    VI_norm      : float
                   variance of I under normality assumption
    seI_norm     : float
                   standard deviation of I under normality assumption
    z_norm       : float
                   z-value of I under normality assumption
    p_norm       : float
                   p-value of I under normality assumption
    VI_rand      : float
                   variance of I under randomization assumption
    seI_rand     : float
                   standard deviation of I under randomization assumption
    z_rand       : float
                   z-value of I under randomization assumption
    p_rand       : float
                   p-value of I under randomization assumption
    two_tailed   : boolean
                   If True p_norm and p_rand are two-tailed, otherwise they
                   are one-tailed.
    sim          : array
                   (if permutations>0)
                   vector of I values for permuted samples
    p_sim        : array
                   (if permutations>0)
                   p-value based on permutations (one-tailed)
                   null: spatial randomness
                   alternative: the observed I is extreme if
                   it is either extremely greater or extremely lower
                   than the values obtained based on permutations
    EI_sim       : float
                   (if permutations>0)
                   average value of I from permutations
    VI_sim       : float
                   (if permutations>0)
                   variance of I from permutations
    seI_sim      : float
                   (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float
                   (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float
                   (if permutations>0)
                   p-value based on standard normal approximation from
                   permutations

    Notes
    -----
    Technical details and derivations can be found in :cite:`cliff81`.


    Examples
    --------
    >>> import libpysal
    >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
    >>> y = np.array(f.by_col['HR8893'])
    >>> from esda.moran import Moran
    >>> mi = Moran(y,  w)
    >>> round(mi.I, 3)
    0.244
    >>> mi.EI
    -0.012987012987012988
    >>> mi.p_norm
    0.00027147862770937614

    SIDS example replicating OpenGeoda

    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> SIDR = np.array(f.by_col("SIDR74"))
    >>> mi = Moran(SIDR,  w)
    >>> round(mi.I, 3)
    0.248
    >>> mi.p_norm
    0.0001158330781489969

    One-tailed

    >>> mi_1 = Moran(SIDR,  w, two_tailed=False)
    >>> round(mi_1.I, 3)
    0.248
    >>> round(mi_1.p_norm, 4)
    0.0001

    """  # noqa: E501

    def __init__(
        self, y, w, transformation="r", permutations=PERMUTATIONS, two_tailed=True
    ):
        y = np.asarray(y).flatten()
        self.y = y
        w = _transform(w, transformation)
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.I = self.__calc(self.z)  # noqa: E741
        self.z_norm = (self.I - self.EI) / self.seI_norm
        self.z_rand = (self.I - self.EI) / self.seI_rand

        if self.z_norm > 0:
            self.p_norm = stats.norm.sf(self.z_norm)
            self.p_rand = stats.norm.sf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        if two_tailed:
            self.p_norm *= 2.0
            self.p_rand *= 2.0

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.z)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            if self.z_sim > 0:
                self.p_z_sim = stats.norm.sf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

        # provide .z attribute that is znormalized
        sy = y.std()
        self.z /= sy

    def __moments(self):
        self.n = len(self.y)
        y = self.y
        z = y - y.mean()
        self.z = z
        self.z2ss = (z * z).sum()
        self.EI = -1.0 / (self.n - 1)
        n = self.n
        n2 = n * n
        if isinstance(self.w, W):
            s1 = self.w.s1
            s0 = self.w.s0
            s2 = self.w.s2
        else:
            self.summary = self.w.summary()
            s1 = self.summary.s1
            s0 = self.summary.s0
            s2 = self.summary.s2
        s02 = s0 * s0
        v_num = n2 * s1 - n * s2 + 3 * s02
        v_den = (n - 1) * (n + 1) * s02
        self.VI_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
        self.seI_norm = self.VI_norm ** (1 / 2.0)

        # variance under randomization
        xd4 = z**4
        xd2 = z**2
        k_num = xd4.sum() / n
        k_den = (xd2.sum() / n) ** 2
        k = k_num / k_den
        EI = self.EI
        A = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
        B = k * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        VIR = (A - B) / ((n - 1) * (n - 2) * (n - 3) * s02) - EI * EI
        self.VI_rand = VIR
        self.seI_rand = VIR ** (1 / 2.0)

    def __calc(self, z):
        zl = _slag(self.w, z)
        inum = (z * zl).sum()
        s0 = self.w.s0 if isinstance(self.w, W) else self.summary.s0
        return self.n / s0 * inum / self.z2ss

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.I

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a Moran statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        cols : string or list of string
            name or list of names of columns to use to compute the statistic
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named 'column_moran'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
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

    def plot_scatter(
        self,
        ax=None,
        scatter_kwds=None,
        fitline_kwds=None,
    ):
        """
        Plot a Moran scatterplot with optional coloring for significant points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        scatter_kwds : dict, optional
            Additional keyword arguments for scatter plot, by default None.
        fitline_kwds : dict, optional
            Additional keyword arguments for fit line, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.
        """
        return _scatterplot(
            self,
            crit_value=None,
            ax=ax,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )

    def plot_simulation(self, ax=None, legend=False, fitline_kwds=None, **kwargs):
        """
        Global Moran's I simulated reference distribution.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        legend : bool, optional
            Plot a legend, by default False
        fitline_kwds : dict, optional
            Additional keyword arguments for vertical Moran fit line, by default None.
        **kwargs : keyword arguments, optional
            Additional keyword arguments for KDE plot passed to ``seaborn.kdeplot``,
            by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.

        Notes
        -----
        This requires optional dependencies ``matplotlib`` and ``seaborn``.

        Examples
        --------
        >>> import libpysal
        >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        >>> y = np.array(f.by_col['HR8893'])
        >>> from esda.moran import Moran
        >>> mi = Moran(y,  w)

        Default plot:

        >>> mi.plot_simulation()

        Customized styling that turns the distribution into a pink line and line
        indicating I to a black line:

        >>> mi.plot_simulation(fitline_kwds={"color": "k"}, color="pink", shade=False)
        """
        return _simulation_plot(
            self,
            ax=ax,
            legend=legend,
            bivariate=False,
            fitline_kwds=fitline_kwds,
            **kwargs,
        )


class Moran_BV:  # noqa: N801
    """
    Bivariate Moran's I

    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        wy will be on y axis
    w : W | Graph
        spatial weights instance as W or Graph aligned with x and y
    transformation  : {'R', 'B', 'D', 'U', 'V'}
                      weights transformation, default is row-standardized "r".
                      Other options include
                      "B": binary,
                      "D": doubly-standardized,
                      "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
                      "V": variance-stabilizing.
    permutations    : int
                      number of random permutations for calculation of pseudo
                      p_values

    Attributes
    ----------
    zx            : array
                    original x variable standardized by mean and std
    zy            : array
                    original y variable standardized by mean and std
    w             : W | Graph
                    original w object
    permutation   : int
                    number of permutations
    I             : float
                    value of bivariate Moran's I
    sim           : array
                    (if permutations>0)
                    vector of I values for permuted samples
    p_sim         : float
                    (if permutations>0)
                    p-value based on permutations (one-sided)
                    null: spatial randomness
                    alternative: the observed I is extreme
                    it is either extremely high or extremely low
    EI_sim        : array
                    (if permutations>0)
                    average value of I from permutations
    VI_sim        : array
                    (if permutations>0)
                    variance of I from permutations
    seI_sim       : array
                    (if permutations>0)
                    standard deviation of I under permutations.
    z_sim         : array
                    (if permutations>0)
                    standardized I based on permutations
    p_z_sim       : float
                    (if permutations>0)
                    p-value based on standard normal approximation from
                    permutations

    Notes
    -----

    Inference is only based on permutations as analytical results are not too
    reliable.

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np

    Set random number generator seed so we can replicate the example

    >>> np.random.seed(10)

    Open the sudden infant death dbf file and read in rates for 74 and 79
    converting each to a numpy array

    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> SIDR74 = np.array(f.by_col['SIDR74'])
    >>> SIDR79 = np.array(f.by_col['SIDR79'])

    Read a GAL file and construct our spatial weights object

    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()

    Create an instance of Moran_BV

    >>> from esda.moran import Moran_BV
    >>> mbi = Moran_BV(SIDR79,  SIDR74,  w)

    What is the bivariate Moran's I value

    >>> round(mbi.I, 3)
    0.156

    Based on 999 permutations, what is the p-value of our statistic

    >>> round(mbi.p_z_sim, 3)
    0.001


    """  # noqa: E501

    def __init__(self, x, y, w, transformation="r", permutations=PERMUTATIONS):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        zy = (y - y.mean()) / y.std(ddof=1)
        zx = (x - x.mean()) / x.std(ddof=1)
        self.y = y
        self.x = x
        self.zx = zx
        self.zy = zy
        n = x.shape[0]
        self.den = n - 1.0  # zx'zx = zy'zy = n-1
        w = _transform(w, transformation)
        self.w = w
        self.I = self.__calc(zy)  # noqa: E741
        if permutations:
            nrp = np.random.permutation
            sim = [self.__calc(nrp(zy)) for i in range(permutations)]
            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            if self.z_sim > 0:
                self.p_z_sim = stats.norm.sf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

    def __calc(self, zy):
        wzy = _slag(self.w, zy)
        self.num = (self.zx * wzy).sum()
        return self.num / self.den

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.I

    @classmethod
    def by_col(
        cls,
        df,
        x,
        y=None,
        w=None,
        inplace=False,
        pvalue="sim",
        outvals=None,
        **stat_kws,
    ):
        """
        Function to compute a Moran_BV statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        X : list of strings
            column name or list of column names to use as X values to compute
            the bivariate statistic. If no Y is provided, pairwise comparisons
            among these variates are used instead.
        Y : list of strings
            column name or list of column names to use as Y values to compute
            the bivariate statistic. if no Y is provided, pariwise comparisons
            among the X variates are used instead.
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named
            'column_moran_local'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran_BV statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran_BV statistic
        **stat_kws : keyword arguments
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran_BV statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
        return _bivariate_handler(
            df,
            x,
            y=y,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            swapname=cls.__name__.lower(),
            stat=cls,
            **stat_kws,
        )

    def plot_scatter(
        self,
        ax=None,
        scatter_kwds=None,
        fitline_kwds=None,
    ):
        """
        Plot a Moran scatterplot with optional coloring for significant points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        scatter_kwds : dict, optional
            Additional keyword arguments for scatter plot, by default None.
        fitline_kwds : dict, optional
            Additional keyword arguments for fit line, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.
        """
        return _scatterplot(
            self,
            crit_value=None,
            bivariate=True,
            ax=ax,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )

    def plot_simulation(self, ax=None, legend=False, fitline_kwds=None, **kwargs):
        """
        Global Moran's I simulated reference distribution.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        legend : bool, optional
            Plot a legend, by default False
        fitline_kwds : dict, optional
            Additional keyword arguments for vertical Moran fit line, by default None.
        **kwargs : keyword arguments, optional
            Additional keyword arguments for KDE plot passed to ``seaborn.kdeplot``,
            by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.

        Notes
        -----
        This requires optional dependencies ``matplotlib`` and ``seaborn``.
        """
        return _simulation_plot(
            self,
            ax=ax,
            legend=legend,
            bivariate=True,
            fitline_kwds=fitline_kwds,
            **kwargs,
        )


def Moran_BV_matrix(variables, w, permutations=0, varnames=None):  # noqa: N802
    """
    Bivariate Moran Matrix

    Calculates bivariate Moran between all pairs of a set of variables.

    Parameters
    ----------
    variables    : array or pandas.DataFrame
                   sequence of variables to be assessed
    w            : W | Graph
                   spatial weights instance as W or Graph aligned with variables
    permutations : int
                   number of permutations
    varnames     : list, optional if variables is an array
                   Strings for variable names. Will add an
                   attribute to `Moran_BV` objects in results needed for plotting
                   in `splot` or `.plot()`. Default =None.
                   Note: If variables is a `pandas.DataFrame` varnames
                   will automatically be generated
    Returns
    -------
    results      : dictionary
                   (i,  j) is the key for the pair of variables, values are
                   the Moran_BV objects.

    Examples
    --------

    open dbf

    >>> import libpysal
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))

    pull of selected variables from dbf and create numpy arrays for each

    >>> varnames = ['SIDR74',  'SIDR79',  'NWR74',  'NWR79']
    >>> vars = [np.array(f.by_col[var]) for var in varnames]

    create a contiguity matrix from an external gal file

    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()

    create an instance of Moran_BV_matrix

    >>> from esda.moran import Moran_BV_matrix
    >>> res = Moran_BV_matrix(vars,  w,  varnames = varnames)

    check values

    >>> round(res[(0,  1)].I,7)
    0.1936261
    >>> round(res[(3,  0)].I,7)
    0.3770138

    """
    try:
        # check if pandas is installed
        import pandas

        if isinstance(variables, pandas.DataFrame):
            # if yes use variables as df and convert to numpy_array
            varnames = pandas.Index.tolist(variables.columns)
            variables_n = []
            for var in varnames:
                variables_n.append(variables[str(var)].values)
        else:
            variables_n = variables
    except ImportError:
        variables_n = variables

    results = _Moran_BV_Matrix_array(
        variables=variables_n, w=w, permutations=permutations, varnames=varnames
    )
    return results


def _Moran_BV_Matrix_array(variables, w, permutations=0, varnames=None):  # noqa: N802
    """
    Base calculation for MORAN_BV_Matrix
    """

    k = len(variables)
    if varnames is None:
        varnames = [f"x{i}" for i in range(k)]

    rk = list(range(0, k - 1))
    results = {}
    for i in rk:
        for j in range(i + 1, k):
            y1 = variables[i]
            y2 = variables[j]
            results[i, j] = Moran_BV(y1, y2, w, permutations=permutations)
            results[j, i] = Moran_BV(y2, y1, w, permutations=permutations)
            results[i, j].varnames = {"x": varnames[i], "y": varnames[j]}
            results[j, i].varnames = {"x": varnames[j], "y": varnames[i]}
    return results


def plot_moran_facet(
    moran_matrix,
    figsize=(16, 12),
    scatter_bv_kwds=None,
    fitline_bv_kwds=None,
    scatter_glob_kwds=dict(color="#737373"),
    fitline_glob_kwds=None,
):
    """
    Moran Facet visualization.

    A matrix containing bivariate Moran plots between all pairs of variables present in
    the ``moran_matrix`` dictionary. On the diagonal contains global Moran plot.

    Parameters
    ----------
    moran_matrix : dict
        Dictionary of Moran_BV objects returned by Moran_BV_matrix
    figsize : tuple, optional
        Size of the figure. Default is (16,12)
    scatter_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        off-diagonal Moran_BV plots.
        Default =None.
    fitline_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        off-diagonal Moran_BV plots.
        Default =None.
    scatter_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        diagonal Moran plots.
        Default =None.
    fitline_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        diagonal Moran plots.
        Default =None.

    Returns
    -------
    ax : matplotlib Axes instance
        Axes in which the figure is plotted
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib must be installed to plot the simulation."
        ) from err

    nrows = int(np.sqrt(len(moran_matrix))) + 1
    ncols = nrows

    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)
    fig.suptitle("Moran Facet")

    for row in range(nrows):
        for col in range(ncols):
            if row == col:
                global_m = Moran(
                    moran_matrix[row, (row + 1) % 4].zy,
                    moran_matrix[row, (row + 1) % 4].w,
                )
                _scatterplot(
                    global_m,
                    crit_value=None,
                    ax=axarr[row, col],
                    scatter_kwds=scatter_glob_kwds,
                    fitline_kwds=fitline_glob_kwds,
                )
                axarr[row, col].set_facecolor("#d9d9d9")
            else:
                _scatterplot(
                    moran_matrix[row, col],
                    bivariate=True,
                    crit_value=None,
                    ax=axarr[row, col],
                    scatter_kwds=scatter_bv_kwds,
                    fitline_kwds=fitline_bv_kwds,
                )

            axarr[row, col].spines[["left", "right", "top", "bottom"]].set_visible(
                False
            )
            if row == nrows - 1:
                axarr[row, col].set_xlabel(
                    str(moran_matrix[(col + 1) % 4, col].varnames["x"]).format(col)
                )
                axarr[row, col].spines["bottom"].set_visible(True)
            else:
                axarr[row, col].set_xlabel("")

            if col == 0:
                axarr[row, col].set_ylabel(
                    (
                        "Spatial Lag of "
                        + str(moran_matrix[row, (row + 1) % 4].varnames["y"])
                    ).format(row)
                )
                axarr[row, col].spines["left"].set_visible(True)
            else:
                axarr[row, col].set_ylabel("")

            axarr[row, col].set_title("")

    plt.tight_layout()

    return axarr


class Moran_Rate(Moran):  # noqa: N801
    """
    Adjusted Moran's I Global Autocorrelation Statistic for Rate
    Variables :cite:`Assuncao1999`

    Parameters
    ----------

    e               : array
                      an event variable measured across n spatial units
    b               : array
                      a population-at-risk variable measured across n spatial
                      units
    w               : W | Graph
                      spatial weights instance as W or Graph aligned with e and b
    adjusted        : boolean
                      whether or not Moran's I needs to be adjusted for rate
                      variable
    transformation  : {'R', 'B', 'D', 'U', 'V'}
                      weights transformation, default is row-standardized "r".
                      Other options include
                      "B": binary,
                      "D": doubly-standardized,
                      "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
                      "V": variance-stabilizing.
    two_tailed      : boolean
                      If True (default), analytical p-values for Moran's I are
                      two-tailed, otherwise they are one tailed.
    permutations    : int
                      number of random permutations for calculation of pseudo
                      p_values

    Attributes
    ----------
    y            : array
                   rate variable computed from parameters e and b
                   if adjusted is True, y is standardized rates
                   otherwise, y is raw rates
    z            : array
                   zero-mean, unit standard deviation normalized y
    w            : W | Graph
                   original w object
    permutations : int
                   number of permutations
    I            : float
                   value of Moran's I
    EI           : float
                   expected value under normality assumption
    VI_norm      : float
                   variance of I under normality assumption
    seI_norm     : float
                   standard deviation of I under normality assumption
    z_norm       : float
                   z-value of I under normality assumption
    p_norm       : float
                   p-value of I under normality assumption
    VI_rand      : float
                   variance of I under randomization assumption
    seI_rand     : float
                   standard deviation of I under randomization assumption
    z_rand       : float
                   z-value of I under randomization assumption
    p_rand       : float
                   p-value of I under randomization assumption
    two_tailed   : boolean
                   If True, p_norm and p_rand are two-tailed p-values,
                   otherwise they are one-tailed.
    sim          : array
                   (if permutations>0)
                   vector of I values for permuted samples
    p_sim        : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed I is extreme if it is
                   either extremely greater or extremely lower than the values
                   obtained from permutaitons
    EI_sim       : float
                   (if permutations>0)
                   average value of I from permutations
    VI_sim       : float
                   (if permutations>0)
                   variance of I from permutations
    seI_sim      : float
                   (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float
                   (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float
                   (if permutations>0)
                   p-value based on standard normal approximation from

    Examples
    --------
    >>> import libpysal
    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> e = np.array(f.by_col('SID79'))
    >>> b = np.array(f.by_col('BIR79'))
    >>> from esda.moran import Moran_Rate
    >>> mi = Moran_Rate(e, b,  w, two_tailed=False)
    >>> "%6.4f" % mi.I
    '0.1662'
    >>> "%6.4f" % mi.p_norm
    '0.0042'
    """  # noqa: E501

    def __init__(
        self,
        e,
        b,
        w,
        adjusted=True,
        transformation="r",
        permutations=PERMUTATIONS,
        two_tailed=True,
    ):
        e = np.asarray(e).flatten()
        b = np.asarray(b).flatten()
        y = assuncao_rate(e, b) if adjusted else e * 1.0 / b
        Moran.__init__(
            self,
            y,
            w,
            transformation=transformation,
            permutations=permutations,
            two_tailed=two_tailed,
        )

    @classmethod
    def by_col(
        cls,
        df,
        events,
        populations,
        w=None,
        inplace=False,
        pvalue="sim",
        outvals=None,
        swapname="",
        **stat_kws,
    ):
        """
        Function to compute a Moran_Rate statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        events : string or list of strings
            one or more names where events are stored
        populations : string or list of strings
            one or more names where the populations corresponding to the
            events are stored. If one population column is provided, it is
            used for all event columns. If more than one population column
            is provided but there is not a population for every event
            column, an exception will be raised.
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named
            'column_moran_rate'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran_Rate statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran_Rate statistic
        **stat_kws : keyword arguments
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran_Rate statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
        if not inplace:
            new = df.copy()
            cls.by_col(
                new,
                events,
                populations,
                w=w,
                inplace=True,
                pvalue=pvalue,
                outvals=outvals,
                swapname=swapname,
                **stat_kws,
            )
            return new
        if isinstance(events, str):
            events = [events]
        if isinstance(populations, str):
            populations = [populations]
        if len(populations) < len(events):
            populations = populations * len(events)
        if len(events) != len(populations):
            raise ValueError(
                "There is not a one-to-one matching between events and populations!"
                f"\nEvents: {events}\nPopulations: {populations}"
            )
        adjusted = stat_kws.pop("adjusted", True)

        if isinstance(adjusted, bool):
            adjusted = [adjusted] * len(events)
        if swapname == "":
            swapname = cls.__name__.lower()

        rates = [
            assuncao_rate(df[e], df[pop]) if adj else df[e].astype(float) / df[pop]
            for e, pop, adj in zip(events, populations, adjusted, strict=True)
        ]
        names = ["-".join((e, p)) for e, p in zip(events, populations, strict=True)]
        out_df = df.copy()
        rate_df = out_df.from_dict(
            dict(zip(names, rates, strict=True))
        )  # trick to avoid importing pandas
        stat_df = _univariate_handler(
            rate_df,
            names,
            w=w,
            inplace=False,
            pvalue=pvalue,
            outvals=outvals,
            swapname=swapname,
            stat=Moran,  # how would this get done w/super?
            **stat_kws,
        )
        for col in stat_df.columns:
            df[col] = stat_df[col]


# -----------------------------------------------------------------------------#
#                            Local Statistics                                 #
# -----------------------------------------------------------------------------#


class Moran_Local:  # noqa: N801
    """Local Moran Statistics.


    Parameters
    ----------
    y : array
        (n,1), attribute array
    w : W | Graph
        spatial weights instance as W or Graph aligned with y
    transformation : {'R', 'B', 'D', 'U', 'V'}
         weights transformation,  default is row-standardized "r".
         Other options include
         "B": binary,
         "D": doubly-standardized,
         "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
         "V": variance-stabilizing.
    permutations : int
        number of random permutations for calculation of pseudo
        p_values
    geoda_quads : boolean
        (default=False)
        If True use GeoDa scheme: HH=1, LL=2, LH=3, HL=4
        If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    keep_simulations : Boolean
        (default=True)
        If True, the entire matrix of replications under the null
        is stored in memory and accessible; otherwise, replications
        are not saved
    seed : None/int
           Seed to ensure reproducibility of conditional randomizations.
           Must be set here, and not outside of the function, since numba
           does not correctly interpret external seeds nor
           numpy.random.RandomState instances.
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Attributes
    ----------

    y : array
        original variable
    w : W | Graph
        original w object
    z : array
        zero-mean, unit standard deviation normalized y
    permutations : int
        number of random permutations for calculation of pseudo p_values
    Is : array
        local Moran's I values
    q : array
        (if permutations>0)
        values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    sim : array (permutations by n)
        (if permutations>0)
        I values for permuted samples
    p_sim : array
        (if permutations>0)
        p-values based on permutations (one-sided)
        null: spatial randomness
        alternative: the observed Ii is further away or extreme
        from the median of simulated values. It is either extremely
        high or extremely low in the distribution of simulated Is.
    EI_sim : array
        (if permutations>0)
        average values of local Is from permutations
    VI_sim : array
        (if permutations>0)
        variance of Is from permutations
    EI : array
        analytical expectation of Is under total permutation,
        from :cite:`Anselin1995`. Is the same at each site,
        and equal to the expectation of I itself when
        transformation='r'. We recommend using EI_sim, not EI,
        for analysis. This EI is only provided for reproducibility.
    VI : array
        analytical variance of Is under total permutation,
        from :cite:`Anselin1995`. Varies according only to
        cardinality. We recommend using VI_sim, not VI, for
        analysis. This VI is only provided for reproducibility.
    EIc : array
        analytical expectation of Is under conditional permutation,
        from :cite:`sokal1998local`. Varies strongly by site, since it
        conditions on z_i. We recommend using EI_sim, not EIc,
        for analysis. This EIc is only provided for reproducibility.
    VIc : array
        analytical variance of Is under conditional permutation,
        from :cite:`sokal1998local`. Varies strongly by site, since
        it conditions on z_i. We recommend using VI_sim, not VIc,
        for analysis. This VIc is only provided for reproducibility.
    seI_sim : array
        (if permutations>0)
        standard deviations of Is under permutations.
    z_sim : arrray
        (if permutations>0)
        standardized Is based on permutations
    p_z_sim : array
        (if permutations>0)
        p-values based on standard normal approximation from
        permutations (one-sided)
        for two-sided tests, these values should be multiplied by 2
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    keep_simulations : Boolean
        (default=True)
        If True, the entire matrix of replications under the null
        is stored in memory and accessible; otherwise, replications
        are not saved
    seed : None/int
        Seed to ensure reproducibility of conditional randomizations.
        Must be set here, and not outside of the function, since numba does
        not correctly interpret external seeds nor numpy.random.RandomState instances.

    Notes
    -----

    For technical details see :cite:`Anselin95`.


    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = libpysal.io.open(libpysal.examples.get_path("desmith.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("desmith.txt"))
    >>> y = np.array(f.by_col['z'])
    >>> from esda.moran import Moran_Local
    >>> lm = Moran_Local(y, w, transformation = "r", permutations = 99)
    >>> lm.q
    array([4, 4, 4, 2, 3, 3, 1, 4, 3, 3])
    >>> lm.p_z_sim[0]
    0.24669152541631179
    >>> lm = Moran_Local(y, w, transformation = "r", permutations = 99, \
                            geoda_quads=True)
    >>> lm.q
    array([4, 4, 4, 3, 2, 2, 1, 4, 2, 2])

    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures.
    """  # noqa: E501

    def __init__(
        self,
        y,
        w,
        transformation="r",
        permutations=PERMUTATIONS,
        geoda_quads=False,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,  # noqa: ARG002
    ):
        y = np.asarray(y).flatten()
        self.y = y
        n = len(y)
        self.n = n
        self.n_1 = n - 1
        z = y - y.mean()
        # setting for floating point noise
        orig_settings = np.seterr()
        np.seterr(all="ignore")
        sy = y.std()
        z /= sy
        np.seterr(**orig_settings)
        self.z = z
        w = _transform(w, transformation)
        self.w = w
        self.permutations = permutations
        self.den = (z * z).sum()
        self.Is = self.__calc(self.w, self.z)
        self.geoda_quads = geoda_quads
        quads = [1, 2, 3, 4]
        if geoda_quads:
            quads = [1, 3, 2, 4]
        self.quads = quads
        self.__quads()
        self.__moments()
        if permutations:
            self.p_sim, self.rlisas = _crand_plus(
                z,
                w,
                self.Is,
                permutations,
                keep_simulations,
                n_jobs=n_jobs,
                stat_func=_moran_local_crand,
                seed=seed,
            )
            self.sim = np.transpose(self.rlisas)
            if keep_simulations:
                sim = np.transpose(self.rlisas)
                above = sim >= self.Is
                larger = above.sum(0)
                low_extreme = (self.permutations - larger) < larger
                larger[low_extreme] = self.permutations - larger[low_extreme]
                self.p_sim = (larger + 1.0) / (permutations + 1.0)
                self.sim = sim
                self.EI_sim = self.sim.mean(axis=0)
                self.seI_sim = self.sim.std(axis=0)
                self.VI_sim = self.seI_sim * self.seI_sim
                self.z_sim = (self.Is - self.EI_sim) / self.seI_sim
                self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))
            else:
                self.sim = self.rlisas = None
                self.EI_sim = np.nan
                self.seI_sim = np.nan
                self.VI_sim = np.nan
                self.z_sim = np.nan
                self.p_z_sim = np.nan

    def __calc(self, w, z):
        zl = _slag(w, z)
        return self.n_1 * self.z * zl / self.den

    def __quads(self):
        zl = _slag(self.w, self.z)
        zp = self.z > 0
        lp = zl > 0
        pp = zp * lp
        np = (1 - zp) * lp
        nn = (1 - zp) * (1 - lp)
        pn = zp * (1 - lp)

        q0, q1, q2, q3 = self.quads
        self.q = (q0 * pp) + (q1 * np) + (q2 * nn) + (q3 * pn)

    def __moments(self):
        W = self.w.sparse
        z = self.z
        simplefilter("always", sparse.SparseEfficiencyWarning)
        n = self.n
        m2 = (z * z).sum() / n
        wi = np.asarray(W.sum(axis=1)).flatten()
        wi2 = np.asarray(W.multiply(W).sum(axis=1)).flatten()

        # ---------------------------------------------------------
        # Conditional randomization null, Sokal 1998, Eqs. A7 & A8
        # assume that division is as written, so that
        # a - b / (n - 1) means a - (b / (n-1))
        # ---------------------------------------------------------
        expectation = -(z**2 * wi) / ((n - 1) * m2)
        var_term1 = (z / m2) ** 2
        var_term2 = n / (n - 2)
        var_term3 = wi2 - (wi**2 / (n - 1))
        var_term4 = m2 - (z**2 / (n - 1))
        variance = var_term1 * var_term2 * var_term3 * var_term4

        self.EIc = expectation
        self.VIc = variance

        # ---------------------------------------------------------
        # Total randomization null, Sokal 1998, Eqs. A3 & A4*
        # ---------------------------------------------------------
        m4 = z**4 / n
        b2 = m4 / m2**2

        expectation = -wi / (n - 1)

        # assume that "avoiding identical subscripts" in :cite:`Anselin1995`
        # includes i==h and i==k, we can use the form due to
        # :cite:`sokal1998local` below.

        # wikh = _wikh_fast(W)
        # variance_anselin = (wi2 * (n - b2)/(n-1)
        #        + 2*wikh*(2*b2 - n) / ((n-1)*(n-2))
        #                    - wi**2/(n-1)**2)
        self.EI = expectation
        n1 = n - 1
        self.VI = wi2 * (n - b2) / n1
        self.VI += (wi**2 - wi2) * (2 * b2 - n) / (n1 * (n - 2))
        self.VI -= (-wi / n1) ** 2

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics."""
        return self.Is

    @classmethod
    def by_col(
        cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a Moran_Local statistic on a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        cols : string or list of string
            name or list of names of columns to use to compute the statistic
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named
            'column_moran_local'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran_Local statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran_Local statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran_Local statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
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

    def get_cluster_labels(self, crit_value=0.05):
        """Return LISA cluster labels for each observation.

        Parameters
        ----------
        crit_value : float, optional
            crititical significance value for statistical inference, by default 0.05

        Returns
        -------
        numpy.array
            an array of cluster labels aligned with the input data used to conduct the
            local Moran analysis
        """
        return _get_cluster_labels(self, crit_value)

    def explore(self, gdf, crit_value=0.05, **kwargs):
        """Create interactive map of LISA indicators

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        kwargs : dict, optional
            additional keyword arguments passed to the geopandas `explore` method

        Returns
        -------
        Folium.Map
            interactive map with LISA clusters
        """
        gdf = gdf.copy()
        gdf["Moran Cluster"] = self.get_cluster_labels(crit_value)
        return _viz_local_moran(self, gdf, crit_value, "explore", **kwargs)

    def plot(self, gdf, crit_value=0.05, **kwargs):
        """Create static map of LISA indicators

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        kwargs : dict, optional
            additional keyword arguments passed to the geopandas `explore` method

        Returns
        -------
        ax
            matplotlib axis
        """
        gdf = gdf.copy()
        gdf["Moran Cluster"] = self.get_cluster_labels(crit_value)
        return _viz_local_moran(self, gdf, crit_value, "plot", **kwargs)

    def plot_scatter(
        self,
        crit_value=0.05,
        ax=None,
        scatter_kwds=None,
        fitline_kwds=None,
    ):
        """
        Plot a Moran scatterplot with optional coloring for significant points.

        Parameters
        ----------
        crit_value : float, optional
            Critical value to determine statistical significance, by default 0.05.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        scatter_kwds : dict, optional
            Additional keyword arguments for scatter plot, by default None.
        fitline_kwds : dict, optional
            Additional keyword arguments for fit line, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.
        """
        return _scatterplot(
            self,
            crit_value=crit_value,
            ax=ax,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )

    def plot_combination(
        self,
        gdf,
        attribute,
        crit_value=0.05,
        region_column=None,
        mask=None,
        mask_color="#636363",
        quadrant=None,
        legend=True,
        scheme="Quantiles",
        cmap="YlGnBu",
        figsize=(15, 4),
        scatter_kwds=None,
        fitline_kwds=None,
        legend_kwds=None,
    ):
        """
        Produce three-plot visualisation of Moran Scatteprlot, LISA cluster
        and Choropleth maps, with Local Moran region and quadrant masking

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        attribute : str
            Column name of attribute which should be depicted in Choropleth map.
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        region_column: string, optional
            Column name containing mask region of interest, by default None
        mask: str, float, int, optional
            Identifier or name of the region to highlight, by default None
            Use the same dtype to specifiy as in original dataset.
        mask_color: str, optional
            Color of mask, by default '#636363'.
        quadrant : int, optional
            Quadrant 1-4 in scatterplot masking values in LISA cluster and
            Choropleth maps, by default None
        figsize: tuple, optional
            W, h of figure, by default (15,4)
        legend: boolean, optional
            If True, legend for maps will be depicted, by default True
        scheme: str, optional
            Name of mapclassify classifier to be used, by default 'Quantiles'
        cmap: str, optional
            Name of matplotlib colormap used for plotting the Choropleth.
            By default 'YlGnBu'.
        scatter_kwds : keyword arguments, optional
            Keywords used for creating and designing the scatter points, by default
            None.
        fitline_kwds : keyword arguments, optional
            Keywords used for creating and designing the moran fitline
            in the scatterplot, by default None.
        legend_kwds : dict
            Keyword arguments passed to geopandas.GeodataFrame.plot ``legend_kwds``
            allowing repositioning of the legend in LISA cluster plot and choropleth.

        Returns
        -------
        axs : array of Matplotlib axes
        """
        return _plot_combination(
            self,
            gdf,
            attribute,
            crit_value=crit_value,
            region_column=region_column,
            mask=mask,
            mask_color=mask_color,
            quadrant=quadrant,
            legend=legend,
            scheme=scheme,
            cmap=cmap,
            figsize=figsize,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
            legend_kwds=legend_kwds,
        )


class Moran_Local_BV:  # noqa: N801
    """Bivariate Local Moran Statistics.


    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        (n,1), wy will be on y axis
    w : W | Graph
        spatial weights instance as W or Graph aligned with y
    transformation : {'R', 'B', 'D', 'U', 'V'}
        weights transformation,  default is row-standardized "r".
        Other options include
        "B": binary,
        "D": doubly-standardized,
        "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
        "V": variance-stabilizing.
    permutations   : int
        number of random permutations for calculation of pseudo
        p_values
    geoda_quads    : boolean
        (default=False)
        If True use GeoDa scheme: HH=1, LL=2, LH=3, HL=4
        If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4
    njobs : int
        number of workers to use to compute the local statistic.
    keep_simulations : Boolean
        (default=True)
        If True, the entire matrix of replications under the null
        is stored in memory and accessible; otherwise, replications
        are not saved
    seed : None/int
        Seed to ensure reproducibility of conditional randomizations.
        Must be set here, and not outside of the function, since numba
        does not correctly interpret external seeds nor
        numpy.random.RandomState instances.
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Attributes
    ----------

    zx : array
        original x variable standardized by mean and std
    zy : array
        original y variable standardized by mean and std
    w : W | Graph
        original w object
    permutations : int
        number of random permutations for calculation of pseudo p_values
    Is : float
        value of Moran's I
    q : array
        (if permutations>0)
        values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    sim : array
        (if permutations>0) vector of I values for permuted samples
    p_sim : array
        (if permutations>0)
        p-value based on permutations (one-sided)
        null: spatial randomness
        alternative: the observed Ii is further away or extreme
        from the median of simulated values. It is either extremelyi
        high or extremely low in the distribution of simulated Is.
    EI_sim : array
        (if permutations>0)
        average values of local Is from permutations
    VI_sim : array
        (if permutations>0)
        variance of Is from permutations
    seI_sim: array
        (if permutations>0)
        standard deviations of Is under permutations.
    z_sim  : arrray
        (if permutations>0)
        standardized Is based on permutations
    p_z_sim: array
        (if permutations>0)
        p-values based on standard normal approximation from
        permutations (one-sided)
        for two-sided tests, these values should be multiplied by 2

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> x = np.array(f.by_col['SIDR79'])
    >>> y = np.array(f.by_col['SIDR74'])
    >>> from esda.moran import Moran_Local_BV
    >>> lm =Moran_Local_BV(x, y, w, transformation = "r", \
                               permutations = 99)
    >>> lm.q[:10]
    array([3, 4, 3, 4, 2, 1, 4, 4, 2, 4])
    >>> lm = Moran_Local_BV(x, y, w, transformation = "r", \
                               permutations = 99, geoda_quads=True)
    >>> lm.q[:10]
    array([2, 4, 2, 4, 3, 1, 4, 4, 3, 4])

    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures.
    """  # noqa: E501

    def __init__(
        self,
        x,
        y,
        w,
        transformation="r",
        permutations=PERMUTATIONS,
        geoda_quads=False,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,  # noqa: ARG002
    ):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        self.y = y
        self.x = x
        n = len(y)
        assert len(y) == len(x), "x and y must have the same shape!"
        self.n = n
        self.n_1 = n - 1
        zx = x - x.mean()
        zy = y - y.mean()
        # setting for floating point noise
        orig_settings = np.seterr()
        np.seterr(all="ignore")
        sx = x.std()
        zx /= sx
        sy = y.std()
        zy /= sy
        np.seterr(**orig_settings)
        self.zx = zx
        self.zy = zy
        w = _transform(w, transformation)
        self.w = w
        self.permutations = permutations
        self.den = (zx * zx).sum()
        self.Is = self.__calc()
        self.geoda_quads = geoda_quads
        quads = [1, 2, 3, 4]
        if geoda_quads:
            quads = [1, 3, 2, 4]
        self.quads = quads
        self.__quads()
        if permutations:
            self.p_sim, self.rlisas = _crand_plus(
                np.column_stack((zx, zy)),
                w,
                self.Is,
                permutations,
                keep_simulations,
                n_jobs=n_jobs,
                stat_func=_moran_local_bv_crand,
                seed=seed,
            )
            self.sim = np.transpose(self.rlisas)
            if keep_simulations:
                sim = np.transpose(self.rlisas)
                above = sim >= self.Is
                larger = above.sum(0)
                low_extreme = (self.permutations - larger) < larger
                larger[low_extreme] = self.permutations - larger[low_extreme]
                self.p_sim = (larger + 1.0) / (permutations + 1.0)
                self.sim = sim
                self.EI_sim = sim.mean(axis=0)
                self.seI_sim = sim.std(axis=0)
                self.VI_sim = self.seI_sim * self.seI_sim
                self.z_sim = (self.Is - self.EI_sim) / self.seI_sim
                self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))

    def __calc(self):
        zly = _slag(self.w, self.zy)
        return self.n_1 * self.zx * zly / self.den

    def __quads(self):
        zl = _slag(self.w, self.zy)
        zp = self.zx > 0
        lp = zl > 0
        pp = zp * lp
        np = (1 - zp) * lp
        nn = (1 - zp) * (1 - lp)
        pn = zp * (1 - lp)

        q0, q1, q2, q3 = self.quads
        self.q = (q0 * pp) + (q1 * np) + (q2 * nn) + (q3 * pn)

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics."""
        return self.Is

    @classmethod
    def by_col(
        cls,
        df,
        x,
        y=None,
        w=None,
        inplace=False,
        pvalue="sim",
        outvals=None,
        **stat_kws,
    ):
        """
        Function to compute a Moran_Local_BV statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        X : list of strings
            column name or list of column names to use as X values to compute
            the bivariate statistic. If no Y is provided, pairwise comparisons
            among these variates are used instead.
        Y : list of strings
            column name or list of column names to use as Y values to compute
            the bivariate statistic. if no Y is provided, pariwise comparisons
            among the X variates are used instead.
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named
            'column_moran_local_bv'
        pvalue  : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran_Local_BV statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran_Local_BV statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran_Local_BV statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
        return _bivariate_handler(
            df,
            x,
            y=y,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            swapname=cls.__name__.lower(),
            stat=cls,
            **stat_kws,
        )

    def get_cluster_labels(self, crit_value=0.05):
        """Return LISA cluster labels for each observation.

        Parameters
        ----------
        crit_value : float, optional
            crititical significance value for statistical inference, by default 0.05

        Returns
        -------
        numpy.array
            an array of cluster labels aligned with the input data used to conduct the
            local Moran analysis
        """
        return _get_cluster_labels(self, crit_value)

    def explore(self, gdf, crit_value=0.05, **kwargs):
        """Create interactive map of LISA indicators

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        kwargs : dict, optional
            additional keyword arguments passed to the geopandas `explore` method

        Returns
        -------
        Folium.Map
            interactive map with LISA clusters
        """
        gdf = gdf.copy()
        gdf["Moran Cluster"] = self.get_cluster_labels(crit_value)
        return _viz_local_moran(self, gdf, crit_value, "explore", **kwargs)

    def plot(self, gdf, crit_value=0.05, **kwargs):
        """Create static map of LISA indicators

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        kwargs : dict, optional
            additional keyword arguments passed to the geopandas `explore` method

        Returns
        -------
        ax
            matplotlib axis
        """
        gdf = gdf.copy()
        gdf["Moran Cluster"] = self.get_cluster_labels(crit_value)
        return _viz_local_moran(self, gdf, crit_value, "plot", **kwargs)

    def plot_scatter(
        self,
        crit_value=0.05,
        ax=None,
        scatter_kwds=None,
        fitline_kwds=None,
    ):
        """
        Plot a Moran scatterplot with optional coloring for significant points.

        Parameters
        ----------
        crit_value : float, optional
            Critical value to determine statistical significance, by default 0.05.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot, by default None.
        scatter_kwds : dict, optional
            Additional keyword arguments for scatter plot, by default None.
        fitline_kwds : dict, optional
            Additional keyword arguments for fit line, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the Moran scatterplot.
        """
        return _scatterplot(
            self,
            crit_value=crit_value,
            bivariate=True,
            ax=ax,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )

    def plot_combination(
        self,
        gdf,
        attribute,
        crit_value=0.05,
        region_column=None,
        mask=None,
        mask_color="#636363",
        quadrant=None,
        legend=True,
        scheme="Quantiles",
        cmap="YlGnBu",
        figsize=(15, 4),
        scatter_kwds=None,
        fitline_kwds=None,
        legend_kwds=None,
    ):
        """
        Produce three-plot visualisation of Moran Scatteprlot, LISA cluster
        and Choropleth maps, with Local Moran region and quadrant masking

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe used to conduct the local Moran analysis
        attribute : str
            Column name of attribute which should be depicted in Choropleth map.
        crit_value : float, optional
            critical value to determine statistical significance, by default 0.05
        region_column: string, optional
            Column name containing mask region of interest, by default None
        mask: str, float, int, optional
            Identifier or name of the region to highlight, by default None
            Use the same dtype to specifiy as in original dataset.
        mask_color: str, optional
            Color of mask, by default '#636363'.
        quadrant : int, optional
            Quadrant 1-4 in scatterplot masking values in LISA cluster and
            Choropleth maps, by default None
        figsize: tuple, optional
            W, h of figure, by default (15,4)
        legend: boolean, optional
            If True, legend for maps will be depicted, by default True
        scheme: str, optional
            Name of mapclassify classifier to be used, by default 'Quantiles'
        cmap: str, optional
            Name of matplotlib colormap used for plotting the Choropleth.
            By default 'YlGnBu'.
        scatter_kwds : keyword arguments, optional
            Keywords used for creating and designing the scatter points, by default
            None.
        fitline_kwds : keyword arguments, optional
            Keywords used for creating and designing the moran fitline
            in the scatterplot, by default None.
        legend_kwds : dict
            Keyword arguments passed to geopandas.GeodataFrame.plot ``legend_kwds``
            allowing repositioning of the legend in LISA cluster plot and choropleth.

        Returns
        -------
        axs : array of Matplotlib axes
        """
        return _plot_combination(
            self,
            gdf,
            attribute,
            crit_value=crit_value,
            region_column=region_column,
            mask=mask,
            mask_color=mask_color,
            quadrant=quadrant,
            legend=legend,
            scheme=scheme,
            cmap=cmap,
            figsize=figsize,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
            legend_kwds=legend_kwds,
        )


class Moran_Local_Rate(Moran_Local):  # noqa: N801
    """
    Adjusted Local Moran Statistics for Rate Variables :cite:`Assuncao1999`.

    Parameters
    ----------
    e : array
        (n,1), an event variable across n spatial units
    b : array
        (n,1), a population-at-risk variable across n spatial units
    w : W | Graph
        spatial weights instance as W or Graph aligned with y
    adjusted : boolean
        whether or not local Moran statistics need to be adjusted for
        rate variable
    transformation : {'R', 'B', 'D', 'U', 'V'}
        weights transformation,  default is row-standardized "r".
        Other options include
        "B": binary,
        "D": doubly-standardized,
        "O": restore original transformation (applicable only if ``w`` is  passed as ``W``),
        "V": variance-stabilizing.
    permutations : int
        number of random permutations for calculation of pseudo
        p_values
    geoda_quads : boolean
         (default=False)
         If True use GeoDa scheme: HH=1, LL=2, LH=3, HL=4
         If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4
    njobs : int
        number of workers to use to compute the local statistic.
    keep_simulations : Boolean
        (default=True)
        If True, the entire matrix of replications under the null
        is stored in memory and accessible; otherwise, replications
        are not saved
    seed : None/int
        Seed to ensure reproducibility of conditional randomizations.
        Must be set here, and not outside of the function, since numba does not
        correctly interpret external seeds nor numpy.random.RandomState instances.
    island_weight : float
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Attributes
    ----------
    y : array
        rate variables computed from parameters e and b
        if adjusted is True, y is standardized rates
        otherwise, y is raw rates
    z : array
        zero-mean, unit standard deviation normalized y
    w : W | Graph
        original w object
    permutations : int
        number of random permutations for calculation of pseudo
        p_values
    Is : float
        value of Local Moran's Ii
    q : array
        (if permutations>0)
        values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    sim : array
        (if permutations>0)
        vector of I values for permuted samples
    p_sim : array
        (if permutations>0)
        p-value based on permutations (one-sided)
        null: spatial randomness
        alternative: the observed Ii is further away or extreme
        from the median of simulated Iis. It is either extremely
        high or extremely low in the distribution of simulated Is
    EI_sim : float
        (if permutations>0)
        average value of I from permutations
    VI_sim : float
        (if permutations>0)
        variance of I from permutations
    seI_sim : float
        (if permutations>0)
        standard deviation of I under permutations.
    z_sim : float
        (if permutations>0)
        standardized I based on permutations
    p_z_sim : float
        (if permutations>0)
        p-value based on standard normal approximation from
        permutations (one-sided)
        for two-sided tests, these values should be multiplied by 2

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> e = np.array(f.by_col('SID79'))
    >>> b = np.array(f.by_col('BIR79'))
    >>> from esda.moran import Moran_Local_Rate
    >>> lm = Moran_Local_Rate(e, b, w, transformation="r", permutations=99)
    >>> lm.q[:10]
    array([2, 4, 3, 1, 2, 1, 1, 4, 2, 4])
    >>> lm = Moran_Local_Rate(
    ...     e, b, w, transformation = "r", permutations=99, geoda_quads=True
    )
    >>> lm.q[:10]
    array([3, 4, 2, 1, 3, 1, 1, 4, 3, 4])

    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures

    """  # noqa: E501

    def __init__(
        self,
        e,
        b,
        w,
        adjusted=True,
        transformation="r",
        permutations=PERMUTATIONS,
        geoda_quads=False,
        n_jobs=1,
        keep_simulations=True,
        seed=None,
        island_weight=0,  # noqa: ARG002
    ):
        e = np.asarray(e).flatten()
        b = np.asarray(b).flatten()
        y = assuncao_rate(e, b) if adjusted else e * 1.0 / b
        Moran_Local.__init__(
            self,
            y,
            w,
            transformation=transformation,
            permutations=permutations,
            geoda_quads=geoda_quads,
            n_jobs=n_jobs,
            keep_simulations=keep_simulations,
            seed=seed,
        )

    @classmethod
    def by_col(
        cls,
        df,
        events,
        populations,
        w=None,
        inplace=False,
        pvalue="sim",
        outvals=None,
        swapname="",
        **stat_kws,
    ):
        """
        Function to compute a Moran_Local_Rate statistic on a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            a pandas dataframe with a geometry column
        events : string or  list of strings
            one or more names where events are stored
        populations : string or list of strings
            one or more names where the populations corresponding to the
            events are stored. If one population column is provided, it is
            used for all event columns. If more than one population column
            is provided but there is not a population for every event
            column, an exception will be raised.
        w : W | Graph
            spatial weights instance as W or Graph aligned with the dataframe. If not
            provided, this is searched for in the dataframe's metadata
        inplace : bool
            a boolean denoting whether to operate on the dataframe
            inplace or to return a series contaning the results of
            the computation. If operating inplace, the derived columns
            will be named 'column_moran_local_rate'
        pvalue : string
            a string denoting which pvalue should be returned. Refer to the
            the Moran_Local_Rate statistic's documentation for available p-values
        outvals : list of strings
            list of arbitrary attributes to return as columns from the
            Moran_Local_Rate statistic
        **stat_kws : dict
            options to pass to the underlying statistic. For this, see the
            documentation for the Moran_Local_Rate statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe
        in memory. Otherwise, returns a copy of the dataframe with
        the relevant columns attached.

        """
        if not inplace:
            new = df.copy()
            cls.by_col(
                new,
                events,
                populations,
                w=w,
                inplace=True,
                pvalue=pvalue,
                outvals=outvals,
                swapname=swapname,
                **stat_kws,
            )
            return new
        if isinstance(events, str):
            events = [events]
        if isinstance(populations, str):
            populations = [populations]
        if len(populations) < len(events):
            populations = populations * len(events)
        if len(events) != len(populations):
            raise ValueError(
                "There is not a one-to-one matching between events and populations!"
                f"\nEvents: {events}\nPopulations: {populations}"
            )
        adjusted = stat_kws.pop("adjusted", True)

        if isinstance(adjusted, bool):
            adjusted = [adjusted] * len(events)
        if swapname == "":
            swapname = cls.__name__.lower()

        rates = [
            assuncao_rate(df[e], df[pop]) if adj else df[e].astype(float) / df[pop]
            for e, pop, adj in zip(events, populations, adjusted, strict=True)
        ]
        names = ["-".join((e, p)) for e, p in zip(events, populations, strict=True)]
        out_df = df.copy()
        rate_df = out_df.from_dict(
            dict(zip(names, rates, strict=True))
        )  # trick to avoid importing pandas
        _univariate_handler(
            rate_df,
            names,
            w=w,
            inplace=True,
            pvalue=pvalue,
            outvals=outvals,
            swapname=swapname,
            stat=Moran_Local,  # how would this get done w/super?
            **stat_kws,
        )
        for col in rate_df.columns:
            df[col] = rate_df[col]


def _viz_local_moran(moran_local, gdf, crit_value, method, **kwargs):
    """Common helper for local Moran's I vizualization

    Parameters
    ----------
    moran_local : esda.Moran_Local
        a fitted local Moran class from the PySAL esda module
    gdf : geopandas.GeoDataFrame
        geodataframe used to create the Moran_Local class
    crit_value : float, optional
        critical value for determining statistical significance, by default 0.05
    method : str {"explore", "plot"}
        GeoDataFrame method to be used
    kwargs : dict, optional
        additional keyword arguments are passed directly
        to the plotting method, by default None

    Returns
    -------
    m | ax
        either folium.Map or maptlotlib.Axes
    """

    try:
        from matplotlib import colors
    except ImportError as err:
        raise ImportError(
            "matplotlib library must be installed to use the vizualization feature"
        ) from err

    gdf = gdf.copy()
    gdf["Moran Cluster"] = moran_local.get_cluster_labels(crit_value)
    gdf["p-value"] = moran_local.p_sim

    x = gdf["Moran Cluster"].values
    y = np.unique(x)
    colors5_mpl = {
        "High-High": "#d7191c",
        "Low-High": "#89cff0",
        "Low-Low": "#2c7bb6",
        "High-Low": "#fdae61",
        "Insignificant": "lightgrey",
    }
    colors5 = [colors5_mpl[i] for i in y]  # for mpl
    hmap = colors.ListedColormap(colors5)
    if "cmap" not in kwargs:
        kwargs["cmap"] = hmap

    return getattr(gdf[["Moran Cluster", "p-value", "geometry"]], method)(
        "Moran Cluster", **kwargs
    )


def _get_cluster_labels(moran_local, crit_value):
    gdf = pd.DataFrame()
    gdf["q"] = moran_local.q
    gdf["p_sim"] = moran_local.p_sim
    gdf["Moran Cluster"] = "Insignificant"

    gdf.loc[(gdf["p_sim"] < crit_value) & (gdf["q"] == 1), "Moran Cluster"] = (
        "High-High"
    )
    gdf.loc[(gdf["p_sim"] < crit_value) & (gdf["q"] == 2), "Moran Cluster"] = "Low-High"
    gdf.loc[(gdf["p_sim"] < crit_value) & (gdf["q"] == 3), "Moran Cluster"] = "Low-Low"
    gdf.loc[(gdf["p_sim"] < crit_value) & (gdf["q"] == 4), "Moran Cluster"] = "High-Low"

    return gdf["Moran Cluster"].values


def _scatterplot(
    moran,
    crit_value=0.05,
    bivariate=False,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
):
    """Generates a Moran Local or Global Scatterplot.

    Parameters
    ----------
    moran : Moran object
        An instance of a Moran or Moran_Local object.
    crit_value : float, optional
        The critical value for significance. Default is 0.05.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, a new figure and axes are created.
    scatter_kwds : dict, optional
        Additional keyword arguments to pass to the scatter plot.
    fitline_kwds : dict, optional
        Additional keyword arguments to pass to the fit line plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the Moran Scatterplot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """

    try:
        from matplotlib import pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib library must be installed to use the scatterplot feature"
        ) from err

    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    if crit_value is not None:
        labels = _get_cluster_labels(moran, crit_value)
        # TODO: allow customization of colors in here and in plot and explore
        # TODO: in a way to keep them easily synced
        colors5_mpl = {
            "High-High": "#d7191c",
            "Low-High": "#89cff0",
            "Low-Low": "#2c7bb6",
            "High-Low": "#fdae61",
            "Insignificant": "lightgrey",
        }
        colors5 = [colors5_mpl[i] for i in labels]  # for mpl

    # define customization
    scatter_kwds.setdefault("alpha", 0.6)
    fitline_kwds.setdefault("alpha", 0.9)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_title("Moran Scatterplot")

    if bivariate:
        x = moran.zx
        lag = lag_spatial(moran.w, moran.zy)
        ax.set_xlabel("Attribute X")
        ax.set_ylabel("Spatial Lag of Y")
    else:
        x = moran.z
        lag = lag_spatial(moran.w, moran.z)
        ax.set_xlabel("Attribute")
        ax.set_ylabel("Spatial Lag")

    fit = stats.linregress(
        x,
        lag,
    )
    # v- and hlines
    ax.axvline(0, alpha=0.5, color="k", linestyle="--")
    ax.axhline(0, alpha=0.5, color="k", linestyle="--")
    if crit_value is not None:
        fitline_kwds.setdefault("color", "k")
        scatter_kwds.setdefault("c", colors5)
        ax.plot(x, fit.intercept + fit.slope * x, **fitline_kwds)
        ax.scatter(x, lag, **scatter_kwds)
    else:
        scatter_kwds.setdefault("color", "#bababa")
        fitline_kwds.setdefault("color", "#d6604d")
        ax.plot(x, fit.intercept + fit.slope * x, **fitline_kwds)
        ax.scatter(x, lag, **scatter_kwds)

    ax.set_aspect("equal")

    return ax


def _simulation_plot(
    moran, ax=None, legend=False, bivariate=False, fitline_kwds=None, **kwargs
):
    try:
        import seaborn as sns
        from matplotlib import pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib and seaborn must be installed to plot the simulation."
        ) from err
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    if ax is None:
        _, ax = plt.subplots()

    # plot distribution
    shade = kwargs.pop("shade", True)
    color = kwargs.pop("color", "#bababa")
    sns.kdeplot(
        moran.sim,
        fill=shade,
        color=color,
        ax=ax,
        label="Distribution of simulated Is",
        **kwargs,
    )

    exp = moran.EI_sim if bivariate else moran.EI

    # customize plot
    fitline_kwds.setdefault("color", "#d6604d")
    ax.vlines(moran.I, 0, 1, **fitline_kwds, label="Moran's I")
    ax.vlines(exp, 0, 1, label="Expected I")
    ax.set_title("Reference Distribution")
    ax.set_xlabel(f"Moran's I: {moran.I:.2f}")

    if legend:
        ax.legend()
    return ax


def _plot_combination(
    moran_loc,
    gdf,
    attribute,
    crit_value=0.05,
    region_column=None,
    mask=None,
    mask_color="#636363",
    quadrant=None,
    legend=True,
    scheme="Quantiles",
    cmap="YlGnBu",
    figsize=(15, 4),
    scatter_kwds=None,
    fitline_kwds=None,
    legend_kwds=None,
):
    """
    Produce three-plot visualisation of Moran Scatteprlot, LISA cluster
    and Choropleth maps, with Local Moran region and quadrant masking

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local or Moran_Local_BV instance
        Values of Moran's Local Autocorrelation Statistic
    gdf : geopandas dataframe
        The Dataframe containing information to plot the two maps.
    attribute : str
        Column name of attribute which should be depicted in Choropleth map.
    p : float, optional
        The p-value threshold for significance. Points and polygons will
        be colored by significance. Default = 0.05.
    region_column: string, optional
        Column name containing mask region of interest. Default = None
    mask: str, float, int, optional
        Identifier or name of the region to highlight. Default = None
        Use the same dtype to specifiy as in original dataset.
    mask_color: str, optional
        Color of mask. Default = '#636363'
    quadrant : int, optional
        Quadrant 1-4 in scatterplot masking values in LISA cluster and
        Choropleth maps. Default = None
    figsize: tuple, optional
        W, h of figure. Default = (15,4)
    legend: boolean, optional
        If True, legend for maps will be depicted. Default = True
    scheme: str, optional
        Name of PySAL classifier to be used. Default = 'Quantiles'
    cmap: str, optional
        Name of matplotlib colormap used for plotting the Choropleth.
        Default = 'YlGnBu'
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        in the scatterplot. Default =None.
    legend_kwds : dict
        Keyword arguments passed to geopandas.GeodataFrame.plot ``legend_kwds`` allowing
        repositioning of the legend in LISA cluster plot and choropleth.

    Returns
    -------
    axs : array of Matplotlib axes
    """
    try:
        from matplotlib import patches
        from matplotlib import pyplot as plt

    except ImportError as err:
        raise ImportError(
            "matplotlib library must be installed to use the scatterplot feature"
        ) from err

    _, axs = plt.subplots(
        1, 3, figsize=figsize, subplot_kw={"aspect": "equal", "adjustable": "datalim"}
    )
    # Moran Scatterplot
    moran_loc.plot_scatter(
        crit_value=crit_value,
        ax=axs[0],
        scatter_kwds=scatter_kwds,
        fitline_kwds=fitline_kwds,
    )

    # Lisa cluster map
    moran_loc.plot(
        gdf,
        crit_value=crit_value,
        ax=axs[1],
        legend=legend,
        legend_kwds=legend_kwds,
    )

    # Choropleth for attribute
    gdf.plot(
        column=attribute,
        scheme=scheme,
        cmap=cmap,
        legend=legend,
        legend_kwds=legend_kwds,
        ax=axs[2],
        alpha=1,
    )
    axs[2].set_axis_off()
    axs[2].set_aspect("equal")

    # MASKING QUADRANT VALUES
    if quadrant is not None:
        # Quadrant masking in Scatterplot
        mask_angles = {1: 0, 2: 90, 3: 180, 4: 270}  # rectangle angles
        # We don't want to change the axis data limits, so use the current ones
        xmin, xmax = axs[0].get_xlim()
        ymin, ymax = axs[0].get_ylim()
        # We are rotating, so we start from 0 degrees and
        # figured out the right dimensions for the rectangles for other angles
        mask_width = {1: abs(xmax), 2: abs(ymax), 3: abs(xmin), 4: abs(ymin)}
        mask_height = {1: abs(ymax), 2: abs(xmin), 3: abs(ymin), 4: abs(xmax)}
        axs[0].add_patch(
            patches.Rectangle(
                (0, 0),
                width=mask_width[quadrant],
                height=mask_height[quadrant],
                angle=mask_angles[quadrant],
                color="#E5E5E5",
                zorder=-1,
                alpha=0.8,
            )
        )
        # quadrant selection in maps
        non_quadrant = ~(moran_loc.q == quadrant)
        mask_quadrant = gdf[non_quadrant]
        df_quadrant = gdf.iloc[~non_quadrant]
        union2 = df_quadrant.dissolve().boundary

        # LISA Cluster mask and cluster boundary
        mask_quadrant.plot(
            scheme=scheme,
            color="white",
            ax=axs[1],
            alpha=0.7,
            zorder=1,
        )
        union2.plot(linewidth=1, ax=axs[1], color="#E5E5E5")

        # CHOROPLETH MASK
        mask_quadrant.plot(
            scheme=scheme,
            color="white",
            ax=axs[2],
            alpha=0.7,
            zorder=1,
        )
        union2.plot(linewidth=1, ax=axs[2], color="#E5E5E5")

    # REGION MASKING
    if region_column is not None:
        # masking inside axs[0] or Moran Scatterplot
        # enforce the same dtype of list and mask
        if not isinstance(mask[0], type(gdf[region_column].iloc[0])):
            warn(
                "Values in `mask` are not the same dtype as"
                + " values in `region_column`. Converting `mask` values"
                + " to dtype of first observation in region_column.",
                stacklevel=3,
            )
            data_type = type(gdf[region_column][0].item())
            mask = list(map(data_type, mask))

        ix = gdf[region_column].isin(mask)

        if not ix.any():
            raise ValueError(
                f"Specified values {mask} in `mask` not in `region_column`"
            )

        df_mask = gdf[ix]
        x_mask = moran_loc.z[ix]
        y_mask = lag_spatial(moran_loc.w, moran_loc.z)[ix]
        axs[0].plot(
            x_mask,
            y_mask,
            color=mask_color,
            marker="o",
            markersize=14,
            alpha=0.8,
            linestyle="None",
            zorder=-1,
        )

        # masking inside axs[1] or Lisa cluster map
        union = df_mask.dissolve().boundary
        union.plot(linewidth=2, ax=axs[1], color=mask_color)

        # masking inside axs[2] or Chloropleth
        union.plot(linewidth=2, ax=axs[2], color=mask_color)

    axs[0].spines[["right", "top"]].set_visible(False)
    axs[1].set_axis_off()

    return axs


# --------------------------------------------------------------
# Conditional Randomization Moment Estimators
# --------------------------------------------------------------


def _wikh_fast(W, sokal_correction=False):
    """
    This computes the outer product of weights for each observation.

    .. math::

        w_{i(kh)} = \\sum_{k \neq i}^n \\sum_{h \neq i}^n w_ik * w_hk

    If the :cite:`sokal1998local` version is used, then we also have h \neq k
    Since this version introduces a simplification in the expression
    where this function is called, the defaults should always return
    the version in the original :cite:`Anselin1995 paper`.

    Arguments
    ---------
    W : scipy sparse matrix
        a sparse matrix describing the spatial relationships
        between observations.
    sokal_correction : bool
        Whether to avoid self-neighbors in the summation of weights.
        If False (default), then the outer product of all weights
        for observation i are used, regardless if they are of the form
        w_hh or w_kk.

    Returns
    -------
    (n,) length numpy.ndarray containing the result.
    """
    return _wikh_numba(
        W.shape[0], *W.nonzero(), W.data, sokal_correction=sokal_correction
    )


@_njit(fastmath=True)
def _wikh_numba(n, row, col, data, sokal_correction=False):
    """
    This is a fast implementation of the wi(kh) function from
    :cite:`Anselin1995`.

    This uses numpy to compute the outer product of each observation's
    weights, after removing the w_ii entry. Then, the sum of the outer
    product is taken. If the sokal correction is requested, the trace
    of the outer product matrix is removed from the result.
    """
    result = np.empty((n,), dtype=data.dtype)
    ixs = np.arange(n)
    for i in ixs:
        # all weights that are not the self weight
        row_no_i = data[(row == i) & (col != i)]
        # compute the pairwise product
        pairwise_product = np.outer(row_no_i, row_no_i)
        # get the sum overall (wik*wih)
        result[i] = pairwise_product.sum()
        if sokal_correction:
            # minus the diagonal (wik*wih when k==h)
            result[i] -= np.trace(pairwise_product)
    return result / 2


def _wikh_slow(W, sokal_correction=False):
    """
    This is a slow implementation of the wi(kh) function from
    :cite:`Anselin1995`

    This does three nested for-loops over n, doing the literal operations
    stated by the expression.
    """
    W = W.toarray()
    (n, n) = W.shape
    result = np.empty((n,))
    # for each observation
    for i in range(n):
        acc = 0
        # we need the product wik
        for k in range(n):
            # excluding wii * wih
            if i == k:
                continue
            # and wij
            for h in range(n):
                # excluding wik * wii
                if i == h:
                    continue
                if sokal_correction and h == k:
                    # excluding wih * wih
                    continue
                acc += W[i, k] * W[i, h]
        result[i] = acc
    return result / 2


# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------


@_njit(fastmath=True)
def _moran_local_bv_crand(i, z, permuted_ids, weights_i, scaling):
    self_weight = weights_i[0]
    other_weights = weights_i[1:]
    zx = z[:, 0]
    zy = z[:, 1]
    zyi, zyrand = _prepare_univariate(i, zy, permuted_ids, other_weights)
    return zx[i] * (zyrand @ other_weights + self_weight * zyi) * scaling


@_njit(fastmath=True)
def _moran_local_crand(i, z, permuted_ids, weights_i, scaling):
    self_weight = weights_i[0]
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return zi * (zrand @ other_weights + self_weight * zi) * scaling
