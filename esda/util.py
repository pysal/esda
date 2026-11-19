import warnings

import numpy as np
from libpysal.weights import W


def _islands(w):
    """Return the ids of isolated observations for a ``W`` or ``Graph``."""
    if isinstance(w, W):
        return list(w.islands)
    return list(w.isolates)


def warn_if_disconnected(w, stat="The statistic"):
    """Warn when the spatial graph is not fully connected.

    Statistics in the Moran family (and several others) are only well defined
    on a fully connected graph. When ``w`` has isolates or more than one
    connected component the resulting value is not interpretable and can fall
    outside its usual bounds (e.g. Moran's :math:`I` outside ``[-1, 1]``). This
    mirrors the connectivity check ``libpysal``'s ``W`` performs on
    construction, but is run at the point of use so it also covers ``Graph``,
    which does not warn on its own (see ``pysal/libpysal#807``).

    Parameters
    ----------
    w : libpysal.weights.W or libpysal.graph.Graph
        The spatial weights or graph used by the statistic.
    stat : str, optional
        Name of the statistic, used in the warning message. Default is
        ``"The statistic"``.

    Notes
    -----
    The check is a no-op for a fully connected graph
    (``w.n_components == 1``). Both ``W`` and ``Graph`` count isolates as
    singleton components, so ``n_components > 1`` also captures the
    isolates-only case.
    """
    if w.n_components <= 1:
        return

    message = (
        "The spatial graph is not fully connected: "
        f"\n There are {w.n_components} disconnected components."
    )
    islands = _islands(w)
    n_islands = len(islands)
    if n_islands == 1:
        message += f"\n There is 1 island with id: {islands[0]}."
    elif n_islands > 1:
        ids = ", ".join(str(island) for island in islands)
        message += f"\n There are {n_islands} islands with ids: {ids}."
    message += (
        f"\n {stat} is not well defined for graphs that are not fully "
        "connected and the resulting value may fall outside its usual bounds."
    )
    warnings.warn(message, UserWarning, stacklevel=3)


def fdr(pvalues, alpha=0.05):
    """
    Calculate the p-value cut-off to control for
    the false discovery rate (FDR) for multiple testing.

    If by controlling for FDR, all of n null hypotheses
    are rejected, the conservative Bonferroni bound (alpha/n)
    is returned instead.

    Parameters
    ----------
    pvalues     : array
                  (n, ), p values for n multiple tests.
    alpha       : float, optional
                  Significance level. Default is 0.05.

    Returns
    -------
                : float
                  Adjusted criterion for rejecting the null hypothesis.
                  If by controlling for FDR, all of n null hypotheses
                  are rejected, the conservative Bonferroni bound (alpha/n)
                  is returned.

    Notes
    -----
    For technical details see :cite:`Benjamini:2001` and :cite:`Castro:2006tz`.

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
    >>> y = np.array(f.by_col['HR8893'])
    >>> from esda import Moran_Local
    >>> from esda import fdr
    >>> lm = Moran_Local(
    ...     y,
    ...     w,
    ...     transformation="r",
    ...     permutations=999,
    ...     seed=12345,
    ...     alternative='two-sided',
    ... )
    >>> fdr(lm.p_sim, 0.1)
    0.001282051282051282

    Return the conservative Bonferroni bound

    >>> fdr(lm.p_sim, 0.05)
    0.000641025641025641
    """

    n = len(pvalues)
    p_sort = np.sort(pvalues)[::-1]
    index = np.arange(n, 0, -1)
    p_fdr = index * alpha / n
    search = p_sort < p_fdr
    sig_all = np.where(search)[0]
    if len(sig_all) == 0:
        return alpha / n
    else:
        return p_fdr[sig_all[0]]
