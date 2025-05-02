import numpy as np
import warnings

try:
    from numba import njit
except (ImportError, ModuleNotFoundError):
    from libpysal.common import jit as njit


def calculate_significance(test_stat, reference_distribution, alternative="two-sided"):
    """
    Calculate a pseudo p-value from a reference distribution.

    Pseudo-p values are calculated using the formula (M + 1) / (R + 1). Where R is the number of simulations
    and M is the number of times that the simulated value was equal to, or more extreme than the observed test statistic.

    Parameters
    ----------
    test_stat: float or numpy.ndarray
        The observed test statistic, or a vector of observed test statistics
    reference_distribution: numpy.ndarray
        A numpy array containing simulated test statistics as a result of conditional permutation.
    alternative: string
        One of 'two-sided', 'lesser', 'greater', 'folded', or 'directed'. Indicates the alternative hypothesis.
        - 'two-sided': the observed test statistic is in either tail of the reference distribution. This is an un-directed alternative hypothesis.
        - 'folded': the observed test statistic is an extreme value of the reference distribution folded about its mean. This is an un-directed alternative hypothesis.
        - 'lesser': the observed test statistic is small relative to the reference distribution. This is a directed alternative hypothesis.
        - 'greater': the observed test statistic is large relative to the reference distribution. This is a directed alternative hypothesis. 
        - 'directed': the observed test statistic is in either tail of the reference distribution, but the tail is selected depending on the test statistic. This is a directed alternative hypothesis, but the direction is chosen dependent on the data. This is not advised, and included solely to reproduce past results.

    Notes
    -----

    the directed p-value is half of the two-sided p-value, and corresponds to running the
    lesser and greater tests, then picking the smaller significance value. This is not advised, 
    since the p-value will be uniformly too small. 
    """
    reference_distribution = np.atleast_2d(reference_distribution)
    n_samples, p_permutations = reference_distribution.shape
    test_stat = np.atleast_2d(test_stat).reshape(n_samples, -1)
    if alternative not in (
        'folded',
        'two-sided',
        'greater',
        'lesser',
        'directed'
    ):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not"
            f" one of the supported options: 'two-sided', 'greater', 'lesser', 'directed', 'folded')"
            )
    return _permutation_significance(
        test_stat, 
        reference_distribution,
        alternative=alternative
        )

@njit(parallel=False, fastmath=False)
def _permutation_significance(test_stat, reference_distribution, alternative='two-sided'):
    reference_distribution = np.atleast_2d(reference_distribution)
    n_samples, p_permutations = reference_distribution.shape
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
        # find percentile p at which the test statistic sits
        # find "synthetic" test statistic at 1-p
        # count how many observations are outisde of (p, 1-p)
        # including the test statistic and its synthetic pair
        percentile = (reference_distribution <= test_stat).mean()*100
        lows, highs = numpy.percentile(
            reference_distribution, (percentile, 100-percentile), axis=1
        ).T
        n_outside = (reference_distribution <= lows[:,None]).sum(axis=1)
        n_outside += (reference_distribution >= highs[:,None]).sum(axis=1)
        p_value = (n_outside + 1) / (p_permutations + 1)
    elif alternative == "folded":
        means = reference_distribution.mean(axis=1, keepdims=True)
        test_stat = np.abs(test_stat - means)
        reference_distribution = np.abs(reference_distribution - means)
        p_value = ((reference_distribution >= test_stat).sum(axis=1) + 1) / (
            p_permutations + 1
        )
    else:
        p_value = np.ones((n_samples, 1))*np.nan
    return p_value


