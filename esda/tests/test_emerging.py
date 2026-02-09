"""
Tests for Emerging Hot Spot Analysis
"""

import numpy as np
import pytest
from libpysal.weights import lat2W

from esda.emerging import EmergingHotSpot, _bh_fdr, _mann_kendall


def test_mann_kendall_increasing():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, p = _mann_kendall(series)
    assert z > 0
    assert p < 0.05


def test_mann_kendall_decreasing():
    series = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    z, p = _mann_kendall(series)
    assert z < 0
    assert p < 0.05


def test_mann_kendall_no_trend():
    series = np.array([1.0, 1.0, 1.0, 1.0])
    z, p = _mann_kendall(series)
    assert abs(z) < 0.1
    assert p > 0.05


def test_mann_kendall_short_series():
    series = np.array([1.0, 2.0])
    z, p = _mann_kendall(series)
    assert np.isnan(z)
    assert np.isnan(p)


def test_bh_fdr_all_significant():
    pvals = np.array([0.001, 0.002, 0.003, 0.004])
    adjusted = _bh_fdr(pvals, alpha=0.05)
    assert np.all(adjusted)


def test_bh_fdr_none_significant():
    pvals = np.array([0.1, 0.2, 0.3, 0.4])
    adjusted = _bh_fdr(pvals, alpha=0.05)
    assert not np.any(adjusted)


def test_bh_fdr_partial_significant():
    pvals = np.array([0.001, 0.01, 0.05, 0.1])
    adjusted = _bh_fdr(pvals, alpha=0.05)
    assert np.sum(adjusted) > 0
    assert np.sum(adjusted) < len(pvals)


def test_emerging_hotspot_basic():
    np.random.seed(42)
    w = lat2W(5, 5)
    n_time = 5
    n_obs = 25

    y = np.random.randn(n_time, n_obs)
    y[:, 12] += 2.0

    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=42, n_jobs=1)

    assert ehsa.z_scores.shape == (n_time, n_obs)
    assert ehsa.p_values.shape == (n_time, n_obs)
    assert ehsa.mk_z.shape == (n_obs,)
    assert ehsa.mk_p.shape == (n_obs,)
    assert ehsa.classification.shape == (n_obs,)

    assert np.all((ehsa.p_values >= 0) & (ehsa.p_values <= 1))
    assert np.all((ehsa.classification >= 0) & (ehsa.classification <= 8))


def test_emerging_hotspot_intensifying_pattern():
    np.random.seed(123)
    w = lat2W(5, 5)
    n_time = 6
    n_obs = 25

    y = np.random.randn(n_time, n_obs) * 0.5

    center_idx = 12
    for t in range(n_time):
        y[t, center_idx] += 1.0 + t * 0.5

    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=123, n_jobs=1)

    center_class = ehsa.classification[center_idx]
    assert 0 <= center_class <= 8
    assert ehsa.z_scores.shape == (n_time, n_obs)


def test_emerging_hotspot_with_nans():
    np.random.seed(456)
    w = lat2W(4, 4)
    n_time = 4
    n_obs = 16

    y = np.random.randn(n_time, n_obs)
    y[1, 5] = np.nan
    y[:, 10] = np.nan

    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=456, n_jobs=1)

    assert ehsa.classification[10] == 8
    assert not np.isnan(ehsa.classification[5])


def test_emerging_hotspot_time_axis():
    np.random.seed(789)
    w = lat2W(3, 3)
    n_time = 4
    n_obs = 9

    y_time_first = np.random.randn(n_time, n_obs)
    y_obs_first = y_time_first.T

    times = np.arange(n_time)

    ehsa1 = EmergingHotSpot(
        y_time_first, w, times, permutations=9, seed=789, time_axis=0, n_jobs=1
    )

    ehsa2 = EmergingHotSpot(
        y_obs_first, w, times, permutations=9, seed=789, time_axis=1, n_jobs=1
    )

    np.testing.assert_array_almost_equal(ehsa1.z_scores, ehsa2.z_scores)
    np.testing.assert_array_almost_equal(ehsa1.classification, ehsa2.classification)


def test_emerging_hotspot_pattern_names():
    np.random.seed(101)
    w = lat2W(3, 3)
    n_time = 4
    n_obs = 9

    y = np.random.randn(n_time, n_obs)
    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=101, n_jobs=1)

    names = ehsa.get_pattern_names()
    assert len(names) == n_obs
    assert all(isinstance(name, str) for name in names)
    assert all(name in EmergingHotSpot.PATTERN_NAMES.values() for name in names)


def test_emerging_hotspot_invalid_times():
    w = lat2W(3, 3)
    y = np.random.randn(5, 9)
    times = np.arange(3)

    with pytest.raises(ValueError, match="times length"):
        EmergingHotSpot(y, w, times)


def test_emerging_hotspot_persistent_pattern():
    np.random.seed(202)
    w = lat2W(4, 4)
    n_time = 8
    n_obs = 16

    y = np.random.randn(n_time, n_obs) * 0.3

    hot_idx = 7
    for t in range(n_time):
        y[t, hot_idx] += 2.0 + np.random.randn() * 0.1

    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=202, n_jobs=1)

    hot_class = ehsa.classification[hot_idx]
    assert 0 <= hot_class <= 8
    assert ehsa.mk_p[hot_idx] <= 1.0


def test_emerging_hotspot_diminishing_pattern():
    np.random.seed(303)
    w = lat2W(4, 4)
    n_time = 6
    n_obs = 16

    y = np.random.randn(n_time, n_obs) * 0.4

    dim_idx = 8
    for t in range(n_time):
        y[t, dim_idx] += 3.0 - t * 0.6

    times = np.arange(n_time)

    ehsa = EmergingHotSpot(y, w, times, permutations=9, seed=303, n_jobs=1)

    dim_class = ehsa.classification[dim_idx]
    assert 0 <= dim_class <= 8
    assert ehsa.mk_z.shape == (n_obs,)


def test_emerging_hotspot_reproducibility():
    np.random.seed(404)
    w = lat2W(3, 3)
    n_time = 4
    n_obs = 9

    y = np.random.randn(n_time, n_obs)
    times = np.arange(n_time)

    ehsa1 = EmergingHotSpot(y, w, times, permutations=9, seed=505, n_jobs=1)

    ehsa2 = EmergingHotSpot(y, w, times, permutations=9, seed=505, n_jobs=1)

    np.testing.assert_array_almost_equal(ehsa1.z_scores, ehsa2.z_scores)
    np.testing.assert_array_equal(ehsa1.classification, ehsa2.classification)
