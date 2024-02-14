import numpy as np

def calculate_significance(test_stat, reference_distribution, method='two-sided'):
    """
    Calculate a pseudo p-value from a reference distribution. 

    Pseudo-p values are calculated using the formula (M + 1) / (R + 1). Where R is the number of simulations and M is the number of times that the simulated value was equal to, or more extreme than the observed test statistic. 

    Simulated test statistics are generated through a process of conditional permutation. Conditional permutation holds fixed the value of Xi and values of neighbors are randomly sampled from X removing Xi simulating spatial randomness. This process is repeated R times to generate a reference distribution from which the pseudo-p value is calculated.

    Parameters
    ----------
    test_stat:
        The observed test statistic
    reference_distribution:
        A one-dimensional numpy array containing simulated test statistics as a result of conditional permutation. 
    method: 
        One of 'two-sided', 'lesser', or 'greater'. Indicates the alternative hypothesis.
        - 'two-sided': the observed test-statistic is more-extreme than expected under the assumption of complete spatial randomness.
        - 'lesser': the observed test-statistic is less than the expected value under the assumption of complete spatial randomness.
        - 'greater': the observed test-statistic is greater than the exepcted value under the assumption of complete spatial randomness. 

    """
    if method == 'two-sided':
        p_value = 2 * (np.sum(reference_distribution >= np.abs(test_stat)) / (len(reference_distribution) + 1))
    elif method == 'lesser':
        p_value = (np.sum(reference_distribution >= test_stat) + 1) / (len(reference_distribution) + 1)
    elif method == 'greater':
        p_value = (np.sum(reference_distribution <= test_stat) + 1) / (len(reference_distribution) + 1)
    else:
        raise ValueError(f"Unknown method {method}")
    return p_value