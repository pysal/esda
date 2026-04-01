"""
Spatio-temporal hot spot analysis using Getis-Ord Gi* and Mann-Kendall trend test
"""

__author__ = "samay2504 <samay.m2504@gmail.com>"
__all__ = ["EmergingHotSpot"]

import numpy as np
from scipy import stats as sp_stats

from .getisord import G_Local


def _mann_kendall(series):
    n = len(series)
    if n < 3:
        return np.nan, np.nan

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(series[j] - series[i])

    unique_vals = len(np.unique(series))
    if unique_vals == n:
        var_s = n * (n - 1) * (2 * n + 5) / 18
    else:
        _, tp = np.unique(series, return_counts=True)
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p


class EmergingHotSpot:
    """
    Spatio-temporal hot spot analysis using Getis-Ord Gi* and Mann-Kendall trend test.

    Computes local Gi* statistics across time slices and applies nonparametric
    Mann-Kendall trend test to detect temporal patterns in spatial clustering.
    This implementation provides statistical building blocks without imposing
    categorical interpretations.

    Parameters
    ----------
    y : np.ndarray
        Data array of shape (n_time, n_obs) or (n_obs, n_time)
    w : W | Graph
        Spatial weights for observations
    times : array-like
        Time identifiers matching y dimensions
    permutations : int, optional
        Number of permutations for Gi* p-values (default: 999)
    seed : int, optional
        Random seed for reproducibility (default: None)
    time_axis : int, optional
        Axis representing time dimension: 0 for (n_time, n_obs), 1 for (n_obs, n_time)
        Default is 0
    n_jobs : int, optional
        Number of parallel jobs for permutation testing (default: 1)
    alpha : float, optional
        Significance level for trend detection (default: 0.05)

    Attributes
    ----------
    z_scores : np.ndarray
        Gi* z-scores of shape (n_time, n_obs)
    p_values : np.ndarray
        Gi* p-values of shape (n_time, n_obs)
    mk_z : np.ndarray
        Mann-Kendall z-statistic per observation (n_obs,)
    mk_p : np.ndarray
        Mann-Kendall p-values per observation (n_obs,)
    trend_direction : np.ndarray
        Trend direction per observation: +1 (increasing), 0 (no trend), -1 (decreasing)

    Examples
    --------
    >>> import numpy as np
    >>> from libpysal.weights import lat2W
    >>> from esda.emerging import EmergingHotSpot

    Create synthetic spatio-temporal data

    >>> np.random.seed(42)
    >>> w = lat2W(5, 5)
    >>> n_time, n_obs = 5, 25
    >>> y = np.random.randn(n_time, n_obs) * 0.5
    >>> y[:, 12] += np.arange(n_time)
    >>> times = np.arange(n_time)

    Compute spatio-temporal statistics

    >>> ehsa = EmergingHotSpot(y, w, times, permutations=99, seed=42, n_jobs=1)
    >>> ehsa.z_scores.shape
    (5, 25)
    >>> ehsa.trend_direction.shape
    (25,)
    """

    def __init__(
        self,
        y: np.ndarray,
        w,
        times: np.ndarray,
        permutations: int = 999,
        seed: int | None = None,
        time_axis: int = 0,
        n_jobs: int = 1,
        alpha: float = 0.05,
    ):
        self.y = np.asarray(y)
        self.w = w
        self.times = np.asarray(times)
        self.permutations = permutations
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha = alpha

        if time_axis == 1:
            self.y = self.y.T

        self.n_time, self.n_obs = self.y.shape

        if len(self.times) != self.n_time:
            raise ValueError(
                f"times length {len(self.times)} "
                f"must match time dimension {self.n_time}"
            )

        self._compute_gi_star()
        self._compute_mann_kendall()
        self._compute_trend_direction()

    def _compute_gi_star(self):
        self.z_scores = np.zeros((self.n_time, self.n_obs))
        self.p_values = np.ones((self.n_time, self.n_obs))

        for t in range(self.n_time):
            y_t = self.y[t, :]

            valid_mask = ~np.isnan(y_t)
            if np.sum(valid_mask) < 3:
                continue

            gi_star = G_Local(
                y_t,
                self.w,
                transform="B",
                permutations=self.permutations,
                star=True,
                seed=self.seed,
                n_jobs=self.n_jobs,
                keep_simulations=False,
            )
            self.z_scores[t, :] = gi_star.Zs
            self.p_values[t, :] = gi_star.p_sim

    def _compute_mann_kendall(self):
        self.mk_z = np.zeros(self.n_obs)
        self.mk_p = np.ones(self.n_obs)

        for i in range(self.n_obs):
            series = self.z_scores[:, i]
            valid_mask = ~np.isnan(series)

            if np.sum(valid_mask) < 3:
                self.mk_z[i] = np.nan
                self.mk_p[i] = np.nan
                continue

            z, p = _mann_kendall(series[valid_mask])
            self.mk_z[i] = z
            self.mk_p[i] = p

    def _compute_trend_direction(self):
        self.trend_direction = np.zeros(self.n_obs, dtype=int)

        for i in range(self.n_obs):
            if np.isnan(self.mk_p[i]) or self.mk_p[i] >= self.alpha:
                self.trend_direction[i] = 0
            elif self.mk_z[i] > 0:
                self.trend_direction[i] = 1
            else:
                self.trend_direction[i] = -1
