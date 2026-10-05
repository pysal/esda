import warnings

import numpy as np

try:
    from numba import njit
except (ImportError, ModuleNotFoundError):
    from libpysal.common import jit as njit


def _resolve_alternative(alternative, stacklevel=3):
    """
    Returns the alternative hypothesis for a permutation p-value.

    ``None`` resolves to ``'directed'`` and emits a ``DeprecationWarning``.

    Parameters
    ----------
    alternative : None or str
        The alternative hypothesis requested by the caller.
    stacklevel : int
        The stack level for the warning. The default points at the caller of
        the function that calls this one.

    Returns
    -------
    str
        The alternative hypothesis to pass to ``calculate_significance``.
    """
    if alternative is None:
        warnings.warn(
            "The alternative hypothesis for permutation inference"
            " is changing in the next major release of esda. We recommend"
            " setting alternative='two-sided', which will generally"
            " double the p-value returned."
            " To retain the current behavior, set alternative='directed'.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return "directed"
    return alternative


def calculate_significance(test_stat, reference_distribution, alternative="two-sided"):
    """
    Calculate a pseudo p-value from a reference distribution.

    Pseudo-p values are calculated using the formula (M + 1) / (R + 1).
    Where R is the number of simulations and M is the number of times that the
    simulated value was equal to, or more extreme than the observed test statistic.
    The 'two-sided' alternative doubles this, 2 * (M + 1) / (R + 1), where M counts
    the smaller of the two tails, and caps the result at one.

    Parameters
    ----------
    test_stat : float or numpy.ndarray
        The observed test statistic, or a vector of observed test statistics
    reference_distribution : numpy.ndarray
        A numpy array containing simulated test statistics as a result
        of conditional permutation.
    alternative : string
        One of 'two-sided', 'lesser', 'greater', 'folded', or 'directed'.
        Indicates the alternative hypothesis.
        - 'two-sided': the observed test statistic is in either tail of
            the reference distribution. This is an un-directed alternative hypothesis.
        - 'folded': the observed test statistic is an extreme value of
            the reference distribution folded about its mean.
            This is an un-directed alternative hypothesis.
        - 'lesser': the observed test statistic is small relative to
            the reference distribution. This is a directed alternative hypothesis.
        - 'greater': the observed test statistic is large relative to
            the reference distribution. This is a directed alternative hypothesis.
        - 'directed': the observed test statistic is in either tail of the reference
            distribution, but the tail is selected depending on the test statistic.
            This is a directed alternative hypothesis, but the direction
            is chosen dependent on the data. This is not advised,
            and included solely to reproduce past results.

    Notes
    -----
    the directed p-value is half of the two-sided p-value, and corresponds to running
    the lesser and greater tests, then picking the smaller significance value.
    This is not advised, since the p-value will be uniformly too small.

    Both tails of the 'two-sided' p-value count the observed test statistic, so a
    statistic that ties with part of the reference distribution contributes to each
    tail. This matters for discrete statistics such as the local join counts, where
    the reference distribution puts mass on a handful of integers.

    Doubling puts a floor of 2 / (R + 1) on the 'two-sided' p-value. With the default
    999 permutations the smallest reportable two-sided p-value is 0.002, so a
    threshold below that never rejects. Raise ``permutations`` to go lower.
    """
    reference_distribution = np.atleast_2d(reference_distribution)
    n_samples, p_permutations = reference_distribution.shape
    test_stat = np.atleast_2d(test_stat).reshape(n_samples, -1)
    if alternative not in ("folded", "two-sided", "greater", "lesser", "directed"):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not"
            " one of the supported options: 'two-sided', 'greater', "
            "'lesser', 'directed', 'folded')"
        )
    result = _permutation_significance(
        test_stat, reference_distribution, alternative=alternative
    )
    if test_stat.size == 1:
        return result[0]
    else:
        return result


@njit(parallel=False, fastmath=False)
def _permutation_significance(
    test_stat, reference_distribution, alternative="two-sided"
):
    reference_distribution = np.atleast_2d(reference_distribution)
    n_samples, p_permutations = reference_distribution.shape
    if isinstance(test_stat, (int, float)):
        test_stat = np.ones((n_samples,)) * test_stat
    if alternative == "directed":
        larger = (reference_distribution >= test_stat).sum(axis=1)
        low_extreme = (p_permutations - larger) < larger
        larger[low_extreme] = p_permutations - larger[low_extreme]
        p_value = (larger + 1.0) / (p_permutations + 1.0)
    elif alternative == "lesser":
        p_value = (np.sum(reference_distribution <= test_stat, axis=1) + 1) / (
            p_permutations + 1
        )
    elif alternative == "greater":
        p_value = (np.sum(reference_distribution >= test_stat, axis=1) + 1) / (
            p_permutations + 1
        )
    elif alternative == "two-sided":
        # use the robust pseudo p-value rather than percentiles. Percentiles
        # are degenerate when the reference distribution is constant, which
        # makes the percentile-based count exceed p_permutations and yields a
        # p-value greater than one.
        greater = (reference_distribution >= test_stat).sum(axis=1)
        lesser = (reference_distribution <= test_stat).sum(axis=1)
        p_value = np.minimum(
            2 * (np.minimum(greater, lesser) + 1) / (p_permutations + 1), 1.0
        )
    elif alternative == "folded":
        means = np.empty((n_samples, 1)).astype(reference_distribution.dtype)
        for i in range(n_samples):
            means[i] = reference_distribution[i].mean()
        folded_test_stat = np.abs(test_stat - means)
        folded_reference_distribution = np.abs(reference_distribution - means)
        p_value = (
            (folded_reference_distribution >= folded_test_stat).sum(axis=1) + 1
        ) / (p_permutations + 1)
    else:
        p_value = np.ones((n_samples,)) * np.nan
    return p_value
