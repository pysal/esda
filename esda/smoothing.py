"""
Apply smoothing to rate computation

[Longer Description]

Author(s):
    Myunghwa Hwang mhwang4@gmail.com
    David Folch dfolch@asu.edu
    Luc Anselin luc.anselin@asu.edu
    Serge Rey srey@asu.edu

"""

__author__ = (
    "Myunghwa Hwang <mhwang4@gmail.com>, "
    "David Folch <dfolch@asu.edu>, "
    "Luc Anselin <luc.anselin@asu.edu>, "
    "Serge Rey <srey@asu.edu"
)

import warnings
from functools import reduce, wraps

import numpy as np
from libpysal.weights.distance import Kernel
from libpysal.weights.spatial_lag import lag_spatial as slag
from scipy.spatial import KDTree
from scipy.stats import chi2, gamma, norm, poisson

__all__ = [
    "Excess_Risk",
    "Empirical_Bayes",
    "Spatial_Empirical_Bayes",
    "Spatial_Rate",
    "Kernel_Smoother",
    "Age_Adjusted_Smoother",
    "Disk_Smoother",
    "Spatial_Median_Rate",
    "Spatial_Filtering",
    "flatten",
    "weighted_median",
    "sum_by_n",
    "crude_age_standardization",
    "direct_age_standardization",
    "indirect_age_standardization",
    "standardized_mortality_ratio",
    "choynowski",
    "assuncao_rate",
]


def smoothing_removal_warning(obj):
    @wraps(obj)
    def wrapper(*args, **kwargs):
        warnings.warn(
            (
                f"The '{__name__}' module is deprecated and '{obj.__name__}()' "
                "will be removed in future release of esda."
            ),
            FutureWarning,
            stacklevel=1,
        )
        return obj(*args, **kwargs)

    return wrapper


@smoothing_removal_warning
def flatten(l, unique=True):  # noqa: E741 - Ambiguous variable name: `l`
    """flatten a list of lists

    Parameters
    ----------
    l          : list
                 of lists
    unique     : boolean
                 whether or not only unique items are wanted (default=True)

    Returns
    -------
    list
        of single items

    Examples
    --------
    Creating a sample list whose elements are lists of integers

    >>> l = [[1, 2], [3, 4, ], [5, 6]]

    Applying flatten function

    >>> flatten(l)
    [1, 2, 3, 4, 5, 6]
    """
    l = reduce(lambda x, y: x + y, l)  # noqa: E741 - Ambiguous variable name: `l`
    if not unique:
        return list(l)
    return list(set(l))


@smoothing_removal_warning
def weighted_median(d, w):
    """A utility function to find a median of d based on w

    Parameters
    ----------
    d          : array
                 (n, 1), variable for which median will be found
    w          : array
                 (n, 1), variable on which d's median will be decided

    Notes
    -----
    d and w are arranged in the same order

    Returns
    -------
    float
        median of d

    Examples
    --------
    Creating an array including five integers.
    We will get the median of these integers.

    >>> d = np.array([5,4,3,1,2])

    Creating another array including weight values for the above integers.
    The median of d will be decided with a consideration to these weight
    values.

    >>> w = np.array([10, 22, 9, 2, 5])

    Applying weighted_median function

    >>> weighted_median(d, w)
    np.int64(4)
    """
    dtype = [("w", f"{w.dtype}"), ("v", f"{d.dtype}")]
    d_w = np.array(list(zip(w, d, strict=True)), dtype=dtype)
    d_w.sort(order="v")
    reordered_w = d_w["w"].cumsum()
    cumsum_threshold = reordered_w[-1] * 1.0 / 2
    median_inx = (reordered_w >= cumsum_threshold).nonzero()[0][0]
    if reordered_w[median_inx] == cumsum_threshold and len(d) - 1 > median_inx:
        return np.sort(d)[median_inx : median_inx + 2].mean()
    return np.sort(d)[median_inx]


@smoothing_removal_warning
def sum_by_n(d, w, n):
    """A utility function to summarize a data array into n values
       after weighting the array with another weight array w

    Parameters
    ----------
    d          : array
                 (t, 1), numerical values
    w          : array
                 (t, 1), numerical values for weighting
    n          : integer
                 the number of groups
                 t = c*n (c is a constant)

    Returns
    -------
               : array
                 (n, 1), an array with summarized values

    Examples
    --------
    Creating an array including four integers.
    We will compute weighted means for every two elements.

    >>> d = np.array([10, 9, 20, 30])

    Here is another array with the weight values for d's elements.

    >>> w = np.array([0.5, 0.1, 0.3, 0.8])

    We specify the number of groups for which the weighted mean is computed.

    >>> n = 2

    Applying sum_by_n function

    >>> sum_by_n(d, w, n)
    array([ 5.9, 30. ])
    """
    t = len(d)
    h = t // n  # must be floor!
    d = d * w
    return np.array([sum(d[i : i + h]) for i in range(0, t, h)])


@smoothing_removal_warning
def crude_age_standardization(e, b, n):
    """A utility function to compute rate through crude age standardization

    Parameters
    ----------
    e : array
       (n*h, 1), event variable measured for each age group across n spatial units
    b : array
       (n*h, 1), population at risk variable measured for each age
       group across n spatial units
    n : integer
       the number of spatial units

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array
                 (n, 1), age standardized rate

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Specifying the number of regions.

    >>> n = 2

    Applying crude_age_standardization function to e and b

    >>> crude_age_standardization(e, b, n)
    array([0.2375    , 0.26666667])
    """
    r = e * 1.0 / b
    b_by_n = sum_by_n(b, 1.0, n)
    age_weight = b * 1.0 / b_by_n.repeat(len(e) // n)
    return sum_by_n(r, age_weight, n)


@smoothing_removal_warning
def direct_age_standardization(e, b, s, n, alpha=0.05):
    """A utility function to compute rate through direct age standardization

    Parameters
    ----------
    e : array
        (n*h, 1), event variable measured for each age group across n spatial units
    b : array
        (n*h, 1), population at risk variable measured for each
        age group across n spatial units
    s : array
        (n*h, 1), standard population for each age group across n spatial units
    n : integer
        the number of spatial units
    alpha : float
            significance level for confidence interval

    Notes
    -----
    e, b, and s are arranged in the same order

    Returns
    -------
    list
        a list of n tuples; a tuple has a rate and its lower and upper limits
        age standardized rates and confidence intervals

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([1000, 1000, 1100, 900, 1000, 900, 1100, 900])

    For direct age standardization, we also need the data for standard population.
    Standard population is a reference population-at-risk (e.g., population
    distribution for the U.S.) whose age distribution can be used as a benchmarking
    point for comparing age distributions across regions (e.g., population
    distribution for Arizona and California). Another array including standard
    population is created.

    >>> s = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying direct_age_standardization function to e and b

    >>> a, b = [i[0] for i in direct_age_standardization(e, b, s, n)]
    >>> round(a, 4)
    np.float64(0.0237)
    >>> round(b, 4)
    np.float64(0.0267)
    """
    age_weight = (1.0 / b) * (s * 1.0 / sum_by_n(s, 1.0, n).repeat(len(s) // n))
    adjusted_r = sum_by_n(e, age_weight, n)
    var_estimate = sum_by_n(e, np.square(age_weight), n)
    g_a = np.square(adjusted_r) / var_estimate
    g_b = var_estimate / adjusted_r
    _b = len(b)
    _rb = range(0, _b, _b // n)
    k = [age_weight[i : i + _b // n].max() for i in _rb]
    g_a_k = np.square(adjusted_r + k) / (var_estimate + np.square(k))
    g_b_k = (var_estimate + np.square(k)) / (adjusted_r + k)
    res = []
    for i in range(len(adjusted_r)):
        if adjusted_r[i] == 0:
            upper = 0.5 * chi2.ppf(1 - 0.5 * alpha)
            lower = 0.0
        else:
            lower = gamma.ppf(0.5 * alpha, g_a[i], scale=g_b[i])
            upper = gamma.ppf(1 - 0.5 * alpha, g_a_k[i], scale=g_b_k[i])
        res.append((adjusted_r[i], lower, upper))
    return res


@smoothing_removal_warning
def indirect_age_standardization(e, b, s_e, s_b, n, alpha=0.05):
    """A utility function to compute rate through indirect age standardization

    Parameters
    ----------
    e : array
        (n*h, 1), event variable measured for each age group across n spatial units
    b : array
        (n*h, 1), population at risk variable measured for each age
        group across n spatial units
    s_e : array
        (n*h, 1), event variable measured for each age group across
        n spatial units in a standard population
    s_b : array
        (n*h, 1), population variable measured for each age group across
        n spatial units in a standard population
    n : integer
        the number of spatial units
    alpha : float
        significance level for confidence interval

    Notes
    -----
    e, b, s_e, and s_b are arranged in the same order

    Returns
    -------
    list
        a list of n tuples; a tuple has a rate and its lower and upper limits
        age standardized rate

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    For indirect age standardization, we also need the data for standard population
    and event. Standard population is a reference population-at-risk (e.g.,
    population distribution for the U.S.) whose age distribution can be used as a
    benchmarking point for comparing age distributions across regions (e.g.,
    popoulation distribution for Arizona and California). When the same concept is
    applied to the event variable, we call it standard event (e.g., the number of
    cancer patients in the U.S.). Two additional arrays including standard population
    and event are created.

    >>> s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
    >>> s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> [i[0] for i in indirect_age_standardization(e, b, s_e, s_b, n)]
    [np.float64(0.23723821989528798), np.float64(0.2610803324099723)]
    """
    smr = standardized_mortality_ratio(e, b, s_e, s_b, n)
    s_r_all = sum(s_e * 1.0) / sum(s_b * 1.0)
    adjusted_r = s_r_all * smr

    e_by_n = sum_by_n(e, 1.0, n)
    log_smr = np.log(smr)
    log_smr_sd = 1.0 / np.sqrt(e_by_n)
    norm_thres = norm.ppf(1 - 0.5 * alpha)
    log_smr_lower = log_smr - norm_thres * log_smr_sd
    log_smr_upper = log_smr + norm_thres * log_smr_sd
    smr_lower = np.exp(log_smr_lower) * s_r_all
    smr_upper = np.exp(log_smr_upper) * s_r_all
    res = list(zip(adjusted_r, smr_lower, smr_upper, strict=True))
    return res


@smoothing_removal_warning
def standardized_mortality_ratio(e, b, s_e, s_b, n):
    """A utility function to compute standardized mortality ratio (SMR).

    Parameters
    ----------
    e : array
        (n*h, 1), event variable measured for each age group across n spatial units
    b : array
        (n*h, 1), population at risk variable measured for each age
        group across n spatial units
    s_e : array
        (n*h, 1), event variable measured for each age group across
        n spatial units in a standard population
    s_b : array
        (n*h, 1), population variable measured for each age group
        across n spatial units in a standard population
    n : integer
        the number of spatial units

    Notes
    -----
    e, b, s_e, and s_b are arranged in the same order

    Returns
    -------
    array
        (nx1)

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    To compute standardized mortality ratio (SMR),
    we need two additional arrays for standard population and event.
    Creating s_e and s_b for standard event and population, respectively.

    >>> s_e = np.array([100, 45, 120, 100, 50, 30, 200, 80])
    >>> s_b = np.array([1000, 900, 1000, 900, 1000, 900, 1000, 900])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> a, b = standardized_mortality_ratio(e, b, s_e, s_b, n)
    >>> round(a, 4)
    np.float64(2.4869)
    >>> round(b, 4)
    np.float64(2.7368)
    """
    s_r = s_e * 1.0 / s_b
    e_by_n = sum_by_n(e, 1.0, n)
    expected = sum_by_n(b, s_r, n)
    smr = e_by_n * 1.0 / expected
    return smr


@smoothing_removal_warning
def choynowski(e, b, n, threshold=None):
    """Choynowski map probabilities [Choynowski1959]_ .

    Parameters
    ----------
    e : array(n*h, 1)
        event variable measured for each age group across n spatial units
    b : array(n*h, 1)
        population at risk variable measured for each age group across n spatial units
    n : integer
        the number of spatial units
    threshold : float
        Returns zero for any p-value greater than threshold

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array (nx1)

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 2 regions in each of which 4 age groups are available.
    The first 4 values are event values for 4 age groups in the region 1,
    and the next 4 values are for 4 age groups in the region 2.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same two regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Specifying the number of regions.

    >>> n = 2

    Applying indirect_age_standardization function to e and b

    >>> a,b = choynowski(e, b, n)
    >>> round(a, 3)
    np.float64(0.304)
    >>> round(b, 3)
    np.float64(0.294)
    """
    e_by_n = sum_by_n(e, 1.0, n)
    b_by_n = sum_by_n(b, 1.0, n)
    r_by_n = sum(e_by_n) * 1.0 / sum(b_by_n)
    expected = r_by_n * b_by_n
    p = []
    for index, i in enumerate(e_by_n):
        if i <= expected[index]:
            p.append(poisson.cdf(i, expected[index]))
        else:
            p.append(1 - poisson.cdf(i - 1, expected[index]))
    if threshold:
        p = [i if i < threshold else 0.0 for i in p]
    return np.array(p)


@smoothing_removal_warning
def assuncao_rate(e, b):
    """The standardized rates where the mean and stadard deviation used for
    the standardization are those of Empirical Bayes rate estimates
    The standardized rates resulting from this function are used to compute
    Moran's I corrected for rate variables [Choynowski1959]_ .

    Parameters
    ----------
    e          : array(n, 1)
                 event variable measured at n spatial units
    b          : array(n, 1)
                 population at risk variable measured at n spatial units

    Notes
    -----
    e and b are arranged in the same order

    Returns
    -------
               : array (nx1)

    Examples
    --------
    Creating an array of an event variable (e.g., the number of cancer patients)
    for 8 regions.

    >>> e = np.array([30, 25, 25, 15, 33, 21, 30, 20])

    Creating another array of a population-at-risk variable (e.g., total population)
    for the same 8 regions.
    The order for entering values is the same as the case of e.

    >>> b = np.array([100, 100, 110, 90, 100, 90, 110, 90])

    Computing the rates

    >>> assuncao_rate(e, b)[:4]
    array([ 1.03843594, -0.04099089, -0.56250375, -1.73061861])
    """

    y = e * 1.0 / b
    e_sum, b_sum = sum(e), sum(b)
    ebi_b = e_sum * 1.0 / b_sum
    s2 = sum(b * ((y - ebi_b) ** 2)) / b_sum
    ebi_a = s2 - ebi_b / (float(b_sum) / len(e))
    ebi_v_raw = ebi_a + ebi_b / b
    ebi_v = np.where(ebi_v_raw < 0, ebi_b / b, ebi_v_raw)

    return (y - ebi_b) / np.sqrt(ebi_v)


@smoothing_removal_warning
class Excess_Risk:
    """Excess Risk

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units

    Attributes
    ----------
    r           : array (n, 1)
                  execess risk values

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl = libpysal.io.open(libpysal.examples.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number
    of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating an instance of Excess_Risk class using stl_e and stl_b

    >>> er = Excess_Risk(stl_e, stl_b)

    Extracting the excess risk values through
    the property r of the Excess_Risk instance, er

    >>> er.r[:10]
    array([[0.20665681],
           [0.43613787],
           [0.42078261],
           [0.22066928],
           [0.57981596],
           [0.35301709],
           [0.56407549],
           [0.17020994],
           [0.3052372 ],
           [0.25821905]])
    """

    def __init__(self, e, b):
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        r_mean = e.sum() * 1.0 / b.sum()
        self.r = e * 1.0 / (b * r_mean)


@smoothing_removal_warning
class Empirical_Bayes:
    """Aspatial Empirical Bayes Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from Empirical Bayes Smoothing

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes
    the number of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating an instance of Empirical_Bayes class using stl_e and stl_b

    >>> eb = Empirical_Bayes(stl_e, stl_b)

    Extracting the risk values through the property
    r of the Empirical_Bayes instance, eb

    >>> eb.r[:10]
    array([[2.36718950e-05],
           [4.54539167e-05],
           [4.78114019e-05],
           [2.76907146e-05],
           [6.58989323e-05],
           [3.66494122e-05],
           [5.79952721e-05],
           [2.03064590e-05],
           [3.31152999e-05],
           [3.02748380e-05]])
    """

    def __init__(self, e, b):
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        e_sum, b_sum = e.sum() * 1.0, b.sum() * 1.0
        r_mean = e_sum / b_sum
        rate = e * 1.0 / b
        r_variat = rate - r_mean
        r_var_left = (b * r_variat * r_variat).sum() * 1.0 / b_sum
        r_var_right = r_mean * 1.0 / b.mean()
        r_var = r_var_left - r_var_right
        weight = r_var / (r_var + r_mean / b)
        self.r = weight * rate + (1.0 - weight) * r_mean


@smoothing_removal_warning
class Spatial_Empirical_Bayes:
    """Spatial Empirical Bayes Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from Empirical Bayes Smoothing

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number
    of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = libpysal.io.open(stl_ex.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of observations
    in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Creating an instance of Spatial_Empirical_Bayes class
    using stl_e, stl_b, and stl_w

    >>> from esda.smoothing import Spatial_Empirical_Bayes
    >>> s_eb = Spatial_Empirical_Bayes(stl_e, stl_b, stl_w)

    Extracting the risk values through the property r of s_eb

    >>> s_eb.r[:10]
    array([[4.01485749e-05],
           [3.62437513e-05],
           [4.93034844e-05],
           [5.09387329e-05],
           [3.72735210e-05],
           [3.69333797e-05],
           [5.40245456e-05],
           [2.99806055e-05],
           [3.73034109e-05],
           [3.47270722e-05]])
    """

    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError(
                "The `id_order` of `w` must be set to "
                "align with the order of `e` and `b`."
            )
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        r_mean = Spatial_Rate(e, b, w).r
        rate = e * 1.0 / b
        r_var_left = np.ones_like(e) * 1.0
        ngh_num = np.ones_like(e)
        bi = slag(w, b) + b
        for i, idv in enumerate(w.id_order):
            ngh = list(w[idv].keys()) + [idv]
            nghi = [w.id2i[k] for k in ngh]
            ngh_num[i] = len(nghi)
            v = sum(np.square(rate[nghi] - r_mean[i]) * b[nghi])
            r_var_left[i] = v
        r_var_left = r_var_left / bi
        r_var_right = r_mean / (bi / ngh_num)
        r_var = r_var_left - r_var_right
        r_var[r_var < 0] = 0.0
        self.r = r_mean + (rate - r_mean) * (r_var / (r_var + (r_mean / b)))


@smoothing_removal_warning
class Spatial_Rate:
    """Spatial Rate Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial rate smoothing

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the
    number of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = libpysal.io.open(stl_ex.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number
    of observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Creating an instance of Spatial_Rate class using stl_e, stl_b, and stl_w

    >>> from esda.smoothing import Spatial_Rate
    >>> sr = Spatial_Rate(stl_e,stl_b,stl_w)

    Extracting the risk values through the property r of sr

    >>> sr.r[:10]
    array([[4.59326407e-05],
           [3.62437513e-05],
           [4.98677081e-05],
           [5.09387329e-05],
           [3.72735210e-05],
           [4.01073093e-05],
           [3.79372794e-05],
           [3.27019246e-05],
           [4.26204928e-05],
           [3.47270722e-05]])
    """

    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError(
                "w id_order must be set to align with the order of e and b."
            )
        else:
            e = np.asarray(e).reshape(-1, 1)
            b = np.asarray(b).reshape(-1, 1)
            w.transform = "b"
            w_e, w_b = slag(w, e), slag(w, b)
            self.r = (e + w_e) / (b + w_b)
            w.transform = "o"


@smoothing_removal_warning
class Kernel_Smoother:
    """Kernal smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : Kernel weights instance

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial rate smoothing

    Examples
    --------
    Creating an array including event values for 6 regions

    >>> e = np.array([10, 1, 3, 4, 2, 5])

    Creating another array including population-at-risk values for the 6 regions

    >>> b = np.array([100, 15, 20, 20, 80, 90])

    Creating a list containing geographic coordinates of the 6 regions' centroids

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a kernel-based spatial weights instance by using the above points

    >>> kw=Kernel(points)

    Ensuring that the elements in the kernel-based weights are ordered
    by the given sequential numbers from 0 to 5

    >>> if not kw.id_order_set: kw.id_order = range(0,len(points))

    Applying kernel smoothing to e and b

    >>> kr = Kernel_Smoother(e, b, kw)

    Extracting the smoothed rates through the
    property r of the Kernel_Smoother instance

    >>> kr.r
    array([[0.10543301],
           [0.0858573 ],
           [0.08256196],
           [0.09884584],
           [0.04756872],
           [0.04845298]])
    """

    def __init__(self, e, b, w):
        if not isinstance(w, Kernel):
            raise TypeError("w must be an instance of Kernel weights.")
        if not w.id_order_set:
            raise ValueError(
                "w id_order must be set to align with the order of e and b."
            )
        else:
            e = np.asarray(e).reshape(-1, 1)
            b = np.asarray(b).reshape(-1, 1)
            w_e, w_b = slag(w, e), slag(w, b)
            self.r = w_e / w_b


@smoothing_removal_warning
class Age_Adjusted_Smoother:
    """Age-adjusted rate smoothing

    Parameters
    ----------
    e : array (n*h, 1)
        event variable measured for each age group across n spatial units
    b : array (n*h, 1)
        population at risk variable measured for each age group across n spatial units
    w : spatial weights instance
    s : array (n*h, 1)
        standard population for each age group across n spatial units

    Attributes
    ----------
    r : array (n, 1)
        rate values from spatial rate smoothing

    Notes
    -----
    Weights used to smooth age-specific events
    and populations are simple binary weights

    Examples
    --------
    Creating an array including 12 values for the 6 regions with 2 age groups

    >>> e = np.array([10, 8, 1, 4, 3, 5, 4, 3, 2, 1, 5, 3])

    Creating another array including 12 population-at-risk values for the 6 regions

    >>> b = np.array([100, 90, 15, 30, 25, 20, 30, 20, 80, 80, 90, 60])

    For age adjustment, we need another array of values containing standard population
    s includes standard population data for the 6 regions

    >>> s = np.array([98, 88, 15, 29, 20, 23, 33, 25, 76, 80, 89, 66])

    Creating a list containing geographic coordinates of the 6 regions' centroids

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a kernel-based spatial weights instance by using the above points

    >>> kw=Kernel(points)

    Ensuring that the elements in the kernel-based weights are ordered
    by the given sequential numbers from 0 to 5

    >>> if not kw.id_order_set: kw.id_order = range(0,len(points))

    Applying age-adjusted smoothing to e and b

    >>> ar = Age_Adjusted_Smoother(e, b, kw, s)

    Extracting the smoothed rates through the property r of
    the Age_Adjusted_Smoother instance

    >>> ar.r
    array([0.10519625, 0.08494318, 0.06440072, 0.06898604, 0.06952076,
           0.05020968])
    """

    def __init__(self, e, b, w, s, alpha=0.05):
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        s = np.asarray(s).flatten()
        t = len(e)
        h = t // w.n
        w.transform = "b"
        e_n, b_n = [], []
        for i in range(h):
            e_n.append(slag(w, e[i::h]).tolist())
            b_n.append(slag(w, b[i::h]).tolist())
        e_n = np.array(e_n).reshape((1, t), order="F")[0]
        b_n = np.array(b_n).reshape((1, t), order="F")[0]
        e_n = e_n.reshape(s.shape)
        b_n = b_n.reshape(s.shape)
        r = direct_age_standardization(e_n, b_n, s, w.n, alpha=alpha)
        self.r = np.array([i[0] for i in r])
        w.transform = "o"


@smoothing_removal_warning
class Disk_Smoother:
    """Locally weighted averages or disk smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights matrix

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from disk smoothing

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal

    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the
    number of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = libpysal.io.open(stl_ex.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of
    observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Applying disk smoothing to stl_e and stl_b

    >>> sr = Disk_Smoother(stl_e,stl_b,stl_w)

    Extracting the risk values through the property r of s_eb

    >>> sr.r[:10]
    array([[4.56502262e-05],
           [3.44027685e-05],
           [3.38280487e-05],
           [4.78530468e-05],
           [3.12278573e-05],
           [2.22596997e-05],
           [2.67074856e-05],
           [2.36924573e-05],
           [3.48801587e-05],
           [3.09511832e-05]])
    """

    def __init__(self, e, b, w):
        if not w.id_order_set:
            raise ValueError(
                "w id_order must be set to align with the order of e and b."
            )
        else:
            e = np.asarray(e).reshape(-1, 1)
            b = np.asarray(b).reshape(-1, 1)
            r = e * 1.0 / b
            weight_sum = []
            for i in w.id_order:
                weight_sum.append(sum(w.weights[i]))
            self.r = slag(w, r) / np.array(weight_sum).reshape(-1, 1)


@smoothing_removal_warning
class Spatial_Median_Rate:
    """Spatial Median Rate Smoothing

    Parameters
    ----------
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    w           : spatial weights instance
    aw          : array (n, 1)
                  auxiliary weight variable measured across n spatial units
    iteration   : integer
                  the number of interations

    Attributes
    ----------
    r           : array (n, 1)
                  rate values from spatial median rate smoothing
    w           : spatial weights instance
    aw          : array (n, 1)
                  auxiliary weight variable measured across n spatial units

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    The 11th and 14th columns in stl_hom.csv includes the number of
    homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Creating a spatial weights instance by reading in stl.gal file.

    >>> stl_w = libpysal.io.open(stl_ex.get_path('stl.gal'), 'r').read()

    Ensuring that the elements in the spatial weights instance are ordered
    by the given sequential numbers from 1 to the number of
    observations in stl_hom.csv

    >>> if not stl_w.id_order_set: stl_w.id_order = range(1,len(stl) + 1)

    Computing spatial median rates without iteration

    >>> smr0 = Spatial_Median_Rate(stl_e,stl_b,stl_w)

    Extracting the computed rates through the property
    r of the Spatial_Median_Rate instance

    >>> smr0.r[:10]
    array([3.96047383e-05, 3.55386859e-05, 3.28308921e-05, 4.30731238e-05,
           3.12453969e-05, 1.97300409e-05, 3.10159267e-05, 2.19279204e-05,
           2.93763432e-05, 2.93763432e-05])

    Recomputing spatial median rates with 5 iterations

    >>> smr1 = Spatial_Median_Rate(stl_e,stl_b,stl_w,iteration=5)

    Extracting the computed rates through the property r
    of the Spatial_Median_Rate instance

    >>> smr1.r[:10]
    array([3.11293620e-05, 2.95956330e-05, 3.11293620e-05, 3.10159267e-05,
           2.98436066e-05, 2.76406686e-05, 3.10159267e-05, 2.94788171e-05,
           2.99460806e-05, 2.96981070e-05])

    Computing spatial median rates by using the base variable as auxilliary weights
    without iteration

    >>> smr2 = Spatial_Median_Rate(stl_e,stl_b,stl_w,aw=stl_b)

    Extracting the computed rates through the property
    r of the Spatial_Median_Rate instance

    >>> smr2.r[:10]
    array([5.77412020e-05, 4.46449551e-05, 5.77412020e-05, 5.77412020e-05,
           4.46449551e-05, 3.61363528e-05, 3.61363528e-05, 4.46449551e-05,
           5.77412020e-05, 4.03987355e-05])

    Recomputing spatial median rates by using the base variable as auxilliary weights
    with 5 iterations

    >>> smr3 = Spatial_Median_Rate(stl_e,stl_b,stl_w,aw=stl_b,iteration=5)

    Extracting the computed rates through the property
    r of the Spatial_Median_Rate instance

    >>> smr3.r[:10]
    array([3.61363528e-05, 4.46449551e-05, 3.61363528e-05, 3.61363528e-05,
           4.46449551e-05, 3.61363528e-05, 3.61363528e-05, 4.46449551e-05,
           3.61363528e-05, 4.46449551e-05])
    """

    def __init__(self, e, b, w, aw=None, iteration=1):
        if not w.id_order_set:
            raise ValueError(
                "w id_order must be set to align with the order of e and b."
            )
        e = np.asarray(e).flatten()
        b = np.asarray(b).flatten()
        self.r = e * 1.0 / b
        self.aw, self.w = aw, w
        while iteration:
            self.__search_median()
            iteration -= 1

    def __search_median(self):
        r, aw, w = self.r, self.aw, self.w
        new_r = []
        if self.aw is None:
            for i, _id in enumerate(w.id_order):
                r_disk = np.append(r[i], r[w.neighbor_offsets[_id]])
                new_r.append(np.median(r_disk))
        else:
            for i, _id in enumerate(w.id_order):
                id_d = [i] + list(w.neighbor_offsets[_id])
                aw_d, r_d = aw[id_d], r[id_d]
                new_r.append(weighted_median(r_d, aw_d))
        self.r = np.asarray(new_r).reshape(r.shape)


@smoothing_removal_warning
class Spatial_Filtering:
    """Spatial Filtering

    Parameters
    ----------
    bbox        : a list of two lists where each list is a pair of coordinates
                  a bounding box for the entire n spatial units
    data        : array (n, 2)
                  x, y coordinates
    e           : array (n, 1)
                  event variable measured across n spatial units
    b           : array (n, 1)
                  population at risk variable measured across n spatial units
    x_grid      : integer
                  the number of cells on x axis
    y_grid      : integer
                  the number of cells on y axis
    r           : float
                  fixed radius of a moving window
    pop         : integer
                  population threshold to create adaptive moving windows

    Attributes
    ----------
    grid        : array (x_grid*y_grid, 2)
                  x, y coordinates for grid points
    r           : array (x_grid*y_grid, 1)
                  rate values for grid points

    Notes
    -----
    No tool is provided to find an optimal value for r or pop.

    Examples
    --------
    Reading data in stl_hom.csv into stl to extract values
    for event and population-at-risk variables

    >>> import libpysal
    >>> stl_ex = libpysal.examples.load_example('stl')
    >>> stl = libpysal.io.open(stl_ex.get_path('stl_hom.csv'), 'r')

    Reading the stl data in the WKT format so that
    we can easily extract polygon centroids

    >>> from libpysal.io.util.wkt import WKTParser
    >>> fromWKT = WKTParser()
    >>> stl.cast('WKT',fromWKT)

    Extracting polygon centroids through iteration

    >>> d = np.array([i.centroid for i in stl[:,0]])

    Specifying the bounding box for the stl_hom data.
    The bbox should includes two points for the left-bottom and the right-top corners

    >>> bbox = [[-92.700676, 36.881809], [-87.916573, 40.3295669]]

    The 11th and 14th columns in stl_hom.csv includes the number
    of homocides and population. Creating two arrays from these columns.

    >>> stl_e, stl_b = np.array(stl[:,10]), np.array(stl[:,13])

    Applying spatial filtering by using a 10*10 mesh grid and a moving window
    with 2 radius

    >>> sf_0 = Spatial_Filtering(bbox,d,stl_e,stl_b,10,10,r=2)

    Extracting the resulting rates through the property
    r of the Spatial_Filtering instance

    >>> sf_0.r[:10]
    array([4.23561763e-05, 4.45290850e-05, 4.56456221e-05, 4.49133384e-05,
           4.39671835e-05, 4.44903042e-05, 4.19845497e-05, 4.11936548e-05,
           3.93463504e-05, 4.04376345e-05])

    Applying another spatial filtering by allowing the moving window to grow until
    600000 people are found in the window

    >>> sf = Spatial_Filtering(bbox,d,stl_e,stl_b,10,10,pop=600000)

    Checking the size of the reulting array including the rates

    >>> sf.r.shape
    (100,)

    Extracting the resulting rates through the property
    r of the Spatial_Filtering instance

    >>> sf.r[:10]
    array([3.73728738e-05, 4.04456300e-05, 4.04456300e-05, 3.81035327e-05,
           4.54831940e-05, 4.54831940e-05, 3.75658628e-05, 3.75658628e-05,
           3.75658628e-05, 3.75658628e-05])
    """

    def __init__(self, bbox, data, e, b, x_grid, y_grid, r=None, pop=None):
        e = np.asarray(e).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        data_tree = KDTree(data)
        x_range = bbox[1][0] - bbox[0][0]
        y_range = bbox[1][1] - bbox[0][1]
        x, y = np.mgrid[
            bbox[0][0] : bbox[1][0] : float(x_range) / x_grid,
            bbox[0][1] : bbox[1][1] : float(y_range) / y_grid,
        ]
        self.grid = list(zip(x.ravel(), y.ravel(), strict=True))
        self.r = []
        if r is None and pop is None:
            raise ValueError("Either r or pop should not be None.")
        if r is not None:
            pnts_in_disk = data_tree.query_ball_point(self.grid, r=r)
            for i in pnts_in_disk:
                r = e[i].sum() * 1.0 / b[i].sum()
                self.r.append(r)
        if pop is not None:
            half_nearest_pnts = data_tree.query(self.grid, k=len(e))[1]
            for i in half_nearest_pnts:
                e_n, b_n = e[i].cumsum(), b[i].cumsum()
                b_n_filter = b_n <= pop
                e_n_f, b_n_f = e_n[b_n_filter], b_n[b_n_filter]
                if len(e_n_f) == 0:
                    e_n_f = e_n[[0]]
                    b_n_f = b_n[[0]]
                self.r.append(e_n_f[-1] * 1.0 / b_n_f[-1])
        self.r = np.array(self.r)
