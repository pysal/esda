import numpy as np
from scipy import stats


def calculate_significance(test_stat, reference_distribution, method="two-sided"):
    """
    Calculate a pseudo p-value from a reference distribution.

    Pseudo-p values are calculated using the formula (M + 1) / (R + 1). Where R is the number of simulations and M is the number of times that the simulated value was equal to, or more extreme than the observed test statistic.

    Simulated test statistics are generated through a process of conditional permutation. Conditional permutation holds fixed the value of Xi and values of neighbors are randomly sampled from X removing Xi simulating spatial randomness. This process is repeated R times to generate a reference distribution from which the pseudo-p value is calculated.

    Parameters
    ----------
    test_stat:
        The observed test statistic, or a vector of observed test statistics
    reference_distribution:
        A numpy array containing simulated test statistics as a result of conditional permutation.
    method:
        One of 'two-sided', 'lesser', or 'greater'. Indicates the alternative hypothesis.
        - 'two-sided': the observed test-statistic is more-extreme than expected under the assumption of complete spatial randomness.
        - 'lesser': the observed test-statistic is less than the expected value under the assumption of complete spatial randomness.
        - 'greater': the observed test-statistic is greater than the exepcted value under the assumption of complete spatial randomness.
        - 'directed': run both lesser and greater tests, then pick the smaller p-value.
    """
    reference_distribution = np.atleast_2d(reference_distribution)
    n_samples, p_permutations = reference_distribution.shape
    test_stat = np.atleast_2d(test_stat).reshape(n_samples, -1)
    if method == "directed":
        larger = (reference_distribution >= test_stat).sum(axis=1)
        low_extreme = (p_permutations - larger) < larger
        larger[low_extreme] = p_permutations - larger[low_extreme]
        p_value = (larger + 1.0) / (p_permutations + 1.0)
    elif method == "lesser":
        p_value = (np.sum(reference_distribution >= test_stat, axis=1) + 1) / (
            p_permutations + 1
        )
    elif method == "greater":
        p_value = (np.sum(reference_distribution <= test_stat, axis=1) + 1) / (
            p_permutations + 1
        )
    elif method == "two-sided":
        percentile = (reference_distribution < test_stat).mean(axis=1)
        bounds = np.column_stack((1 - percentile, percentile)) * 100
        bounds.sort(axis=1)
        lows, highs = np.row_stack(
            [
                stats.scoreatpercentile(r, per=p)
                for r, p in zip(reference_distribution, bounds)
            ]
        ).T
        n_outside = (reference_distribution < lows[:, None]).sum(axis=1)
        n_outside += (reference_distribution > highs[:, None]).sum(axis=1) + 1
        p_value = (n_outside + 1) / (p_permutations + 1)
    elif method == "folded":
        means = reference_distribution.mean(axis=1, keepdims=True)
        test_stat = np.abs(test_stat - means)
        reference_distribution = np.abs(reference_distribution - means)
        p_value = ((reference_distribution >= test_stat).sum(axis=1) + 1) / (
            p_permutations + 1
        )
    else:
        raise ValueError(
            f"Unknown p-value method: {method}. Generally, 'two-sided' is a good default!"
        )
    return p_value


if __name__ == "__main__":
    import numpy
    import esda
    import pandas
    from libpysal.weights import Voronoi
    from esda.significance import calculate_significance

    coordinates = numpy.random.random(size=(2000, 2))
    x = numpy.random.normal(size=(2000,))
    w = Voronoi(coordinates, clip="bbox")
    w.transform = "r"
    stat = esda.Moran_Local(x, w)

    ts = calculate_significance(stat.Is, stat.rlisas, method="two-sided")
    di = calculate_significance(stat.Is, stat.rlisas, method="directed")
    lt = calculate_significance(stat.Is, stat.rlisas, method="lesser")
    gt = calculate_significance(stat.Is, stat.rlisas, method="greater")
    fo = calculate_significance(stat.Is, stat.rlisas, method="folded")

    numpy.testing.assert_array_equal(
        numpy.minimum(lt, gt), di
    )  # di is just the minimum of the two tests

    print(
        f"directed * 2 is the same as two-sided {(di*2 == ts).mean()*100}% of the time"
    )

    print(
        pandas.DataFrame(
            numpy.column_stack((ts, di, fo, lt, gt)),
            columns=["two-sided", "directed", "folded", "lt", "gt"],
        ).corr()
    )
