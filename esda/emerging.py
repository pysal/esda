"""
Emerging Hot Spot Analysis for spatio-temporal clustering detection
"""

__author__ = "samay2504 <samay.m2504@gmail.com>"
__all__ = ["EmergingHotSpot"]

import numpy as np
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator

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


def _bh_fdr(pvalues, alpha=0.05):
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]

    threshold_line = alpha * (np.arange(n) + 1) / n
    significant = sorted_p <= threshold_line

    if np.any(significant):
        max_idx = np.where(significant)[0][-1]
        adjusted = np.zeros(n, dtype=bool)
        adjusted[sorted_idx[: max_idx + 1]] = True
        return adjusted
    return np.zeros(n, dtype=bool)


def _classify_pattern(z_series, p_series, mk_z, mk_p, alpha=0.05):
    significant = p_series < alpha
    n_sig = np.sum(significant)
    n_total = len(z_series)

    if n_sig == 0:
        return 8

    recent_sig = significant[-min(3, n_total) :]
    past_sig = significant[: max(1, n_total - 3)]

    is_recent = np.any(recent_sig)
    is_past = np.any(past_sig)

    if n_sig == 1:
        if is_recent and not is_past:
            return 0
        elif is_past and not is_recent:
            return 7
        else:
            return 5

    if mk_p < alpha:
        if mk_z > 0 and np.all(recent_sig):
            return 1
        elif mk_z < 0:
            return 3

    if n_sig >= 0.8 * n_total and mk_p >= alpha:
        return 2

    consecutive_count = 0
    max_consecutive = 0
    for sig in significant:
        if sig:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else:
            consecutive_count = 0

    if max_consecutive >= 3:
        return 6

    changes = 0
    for i in range(len(significant) - 1):
        if significant[i] != significant[i + 1]:
            changes += 1

    if changes >= 0.4 * n_total:
        return 4

    return 5


class EmergingHotSpot(BaseEstimator):
    """
    Emerging Hot Spot Analysis using Getis-Ord Gi* and Mann-Kendall trend test.

    Detects spatio-temporal clustering patterns by computing Gi* statistics across
    time slices and applying trend analysis with classification into ESRI-compatible
    pattern types.

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
    classification : np.ndarray
        Pattern classification codes per observation (n_obs,)

    Notes
    -----
    Classification codes (sign of z-scores indicates hot spot vs cold spot):
    0 = new, 1 = intensifying, 2 = persistent, 3 = diminishing,
    4 = oscillating, 5 = sporadic, 6 = consecutive, 7 = historical, 8 = not significant

    Examples
    --------
    >>> import numpy as np
    >>> from libpysal.weights import lat2W
    >>> from esda.emerging import EmergingHotSpot

    Create synthetic spatio-temporal data with an emerging hot spot

    >>> np.random.seed(42)
    >>> w = lat2W(5, 5)
    >>> n_time, n_obs = 5, 25
    >>> y = np.random.randn(n_time, n_obs) * 0.5
    >>> y[:, 12] += np.arange(n_time)  # intensifying pattern at location 12
    >>> times = np.arange(n_time)

    Run Emerging Hot Spot Analysis

    >>> ehsa = EmergingHotSpot(y, w, times, permutations=99, seed=42, n_jobs=1)
    >>> ehsa.z_scores.shape
    (5, 25)
    >>> ehsa.classification.shape
    (25,)

    Get pattern names

    >>> patterns = ehsa.get_pattern_names()
    >>> patterns[12]  # doctest: +SKIP
    'intensifying'

    Check for significant hot spots

    >>> significant = ehsa.classification < 8
    >>> np.sum(significant)  # doctest: +SKIP
    3
    """

    PATTERN_NAMES = {
        0: "new",
        1: "intensifying",
        2: "persistent",
        3: "diminishing",
        4: "oscillating",
        5: "sporadic",
        6: "consecutive",
        7: "historical",
        8: "not_significant",
    }

    def __init__(
        self,
        y: np.ndarray,
        w,
        times: np.ndarray,
        permutations: int = 999,
        seed: int | None = None,
        time_axis: int = 0,
        n_jobs: int = 1,
    ):
        self.y = np.asarray(y)
        self.w = w
        self.times = np.asarray(times)
        self.permutations = permutations
        self.seed = seed
        self.n_jobs = n_jobs

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
        self._classify_patterns()

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

    def _classify_patterns(self, alpha=0.05):
        self.classification = np.zeros(self.n_obs, dtype=int)

        valid_p = self.mk_p[~np.isnan(self.mk_p)]
        if len(valid_p) > 0:
            significant_mask = _bh_fdr(valid_p, alpha=alpha)
            valid_idx = np.where(~np.isnan(self.mk_p))[0]
            adjusted_sig = np.zeros(self.n_obs, dtype=bool)
            adjusted_sig[valid_idx] = significant_mask
        else:
            adjusted_sig = np.zeros(self.n_obs, dtype=bool)

        for i in range(self.n_obs):
            if np.isnan(self.mk_p[i]):
                self.classification[i] = 8
            else:
                mk_p_corrected = self.mk_p[i] if adjusted_sig[i] else 1.0
                pattern = _classify_pattern(
                    self.z_scores[:, i],
                    self.p_values[:, i],
                    self.mk_z[i],
                    mk_p_corrected,
                    alpha=alpha,
                )
                self.classification[i] = pattern

    def get_pattern_names(self):
        """
        Get human-readable pattern names for classifications.

        Returns
        -------
        np.ndarray
            Array of pattern name strings
        """
        return np.array([self.PATTERN_NAMES[c] for c in self.classification])
