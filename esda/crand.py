"""
Centralised conditional randomisation engine. Numba accelerated.
"""

import os
import warnings

import numpy as np

from .significance import _permutation_significance

try:
    from numba import boolean, njit, prange
except (ImportError, ModuleNotFoundError):
    from libpysal.common import jit as njit

    prange = range
    boolean = bool


__all__ = ["crand"]

# Integer codes for alternative hypotheses — avoids string handling inside
# Numba parallel loops where unicode support is inconsistent across backends.
_ALT_DIRECTED = 0
_ALT_TWO_SIDED = 1
_ALT_GREATER = 2
_ALT_LESSER = 3
_ALT_FOLDED = 4

_ALTERNATIVE_CODES = {
    "directed": _ALT_DIRECTED,
    "two-sided": _ALT_TWO_SIDED,
    "greater": _ALT_GREATER,
    "lesser": _ALT_LESSER,
    "folded": _ALT_FOLDED,
}

#######################################################################
#                   Utilities for all functions                       #
#######################################################################


@njit(fastmath=True)
def vec_permutations(max_card: int, n: int, k_replications: int, seed: int):
    """
    Generate `max_card` permuted IDs, sampled from `n` without replacement,
    `k_replications` times
    ...

    Parameters
    ----------
    max_card : int
        Number of permuted IDs to generate per sample
    n : int
        Size of the sample to sample IDs from
    k_replications : int
        Number of samples of permuted IDs to perform
    seed : int
        Seed to ensure reproducibility of conditional randomizations

    Returns
    -------
    result : ndarray
        (k_replications, max_card) array with permuted IDs
    """
    np.random.seed(seed)
    result = np.empty((k_replications, max_card), dtype=np.int64)
    for k in prange(k_replications):
        result[k] = np.random.choice(n - 1, size=max_card, replace=False)
    return result


@njit(fastmath=True)
def _p_sim_one(observed_i, rstats, alt_code):
    """Compute pseudo-p-value for a single observation's permutation distribution.

    Parameters
    ----------
    observed_i : float
        Observed statistic for observation i.
    rstats : ndarray
        (permutations,) simulated statistics under the null.
    alt_code : int
        One of the _ALT_* integer codes.

    Returns
    -------
    float32
        Pseudo-p-value.
    """
    # Scalar loops instead of array ops: Numba optimises these better than
    # vectorised expressions inside a parallel region.
    p = rstats.shape[0]
    if alt_code == _ALT_DIRECTED:
        larger = np.int64(0)
        for v in rstats:
            if v >= observed_i:
                larger += 1
        # pick the smaller tail
        if (p - larger) < larger:
            larger = p - larger
        return np.float32((larger + 1.0) / (p + 1.0))
    elif alt_code == _ALT_GREATER:
        larger = np.int64(0)
        for v in rstats:
            if v >= observed_i:
                larger += 1
        return np.float32((larger + 1.0) / (p + 1.0))
    elif alt_code == _ALT_LESSER:
        lesser = np.int64(0)
        for v in rstats:
            if v <= observed_i:
                lesser += 1
        return np.float32((lesser + 1.0) / (p + 1.0))
    elif alt_code == _ALT_TWO_SIDED:
        # mirror _permutation_significance: find the symmetric tail cutoffs
        # then count observations outside both tails
        n_below = np.int64(0)
        for v in rstats:
            if v <= observed_i:
                n_below += 1
        pct = n_below / p * 100.0
        p_low = min(pct, 100.0 - pct)
        low = np.percentile(rstats, p_low)
        high = np.percentile(rstats, 100.0 - p_low)
        n_outside = np.int64(0)
        for v in rstats:
            if v <= low or v >= high:
                n_outside += 1
        return np.float32((n_outside + 1.0) / (p + 1.0))
    else:  # FOLDED
        mean = rstats.mean()
        folded_obs = abs(observed_i - mean)
        n_extreme = np.int64(0)
        for v in rstats:
            if abs(v - mean) >= folded_obs:
                n_extreme += 1
        return np.float32((n_extreme + 1.0) / (p + 1.0))


@njit(parallel=True, fastmath=True)
def _compute_prange(
    z,
    observed,
    cardinalities,
    self_weights,
    other_weights,
    w_indptr,
    permuted_ids,
    scaling,
    keep,
    stat_func,
    island_weight,
    alt_code,
):
    """Conditional randomisation using Numba prange for observation-level parallelism.

    Uses the same pre-generated ``permuted_ids`` array as the serial path so
    results are numerically identical to ``compute_chunk``.  Parallelism comes
    from ``prange`` over observations rather than joblib chunks, eliminating
    Python-level dispatch overhead and allowing Numba to schedule threads
    directly.

    Parameters
    ----------
    z : ndarray
        (n,) standardised observed values.
    observed : ndarray
        (n,) observed statistics.
    cardinalities : ndarray
        (n,) neighbour counts.
    self_weights : ndarray
        (n,) self-weights (usually zero).
    other_weights : ndarray
        Flat neighbour-weight buffer (CSR data array, self-weights removed).
    w_indptr : ndarray
        (n+1,) cumulative sum of cardinalities; offsets into ``other_weights``.
    permuted_ids : ndarray
        (permutations, max_cardinality) array from ``vec_permutations``.
    scaling : float
        Scaling factor passed to ``stat_func``.
    keep : bool
        If True, store all simulated statistics in the returned array.
    stat_func : callable
        Numba-compiled statistic function with signature
        ``(i, z, permuted_ids, weights_i, scaling) -> (permutations,) array``.
    island_weight : float
        Weight assigned to the synthetic neighbour of isolated observations.
    alt_code : int
        One of the _ALT_* integer codes.

    Returns
    -------
    p_sims : ndarray
        (n,) float32 pseudo-p-values.
    rlocals : ndarray
        (n, permutations) simulated statistics if ``keep`` is True;
        otherwise a (1, 1) placeholder.
    """
    n = z.shape[0]
    p_sims = np.zeros(n, dtype=np.float32)
    rlocals = np.empty((n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    for i in prange(n):
        cardinality = cardinalities[i]
        if cardinality == 0:
            # island: give it one synthetic zero-weight neighbour
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            # weights_i[0] = self-weight; weights_i[1:] = neighbour weights
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            weights_i[1:] = other_weights[w_indptr[i] : w_indptr[i + 1]]

        rstats = stat_func(i, z, permuted_ids, weights_i, scaling)
        p_sims[i] = _p_sim_one(observed[i], rstats, alt_code)
        if keep:
            rlocals[i] = rstats

    return p_sims, rlocals


def crand(
    z,
    w,
    observed,
    permutations,
    keep,
    n_jobs,
    stat_func,
    scaling=None,
    seed=None,
    island_weight=0,
    alternative=None,
):
    """
    Conduct conditional randomization of a given input using the provided
    statistic function. Numba accelerated.
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    w : libpysal.weights.W
        Spatial weights object
    observed : ndarray
        (N,) array with observed values
    permutations : int
        Number of permutations for conditional randomisation
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
     seed : None/int
        Seed to ensure reproducibility of conditional randomizations

    Returns
    -------
    p_sim : ndarray
        (N,) array with pseudo p-values from conditional permutation
    rlocals : ndarray
        If keep=True, (N, permutations) array with simulated values
        of stat_func under the null of spatial randomness; else, empty (1, 1) array
    """
    adj_matrix = w.sparse

    n = len(z)
    if z.ndim == 2:
        if z.shape[1] == 2:
            # assume that matrix is [X Y], and scaling is moran-like
            scaling = (
                (n - 1) / (z[:, 0] * z[:, 0]).sum() if (scaling is None) else scaling
            )
        elif z.shape[1] == 1:
            # assume that matrix is [X], and scaling is moran-like
            scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
        else:
            raise NotImplementedError(
                f"multivariable input is not yet supported in "
                f"conditional randomization. Recieved `z` of shape {z.shape}"
            )
    elif z.ndim == 1:
        scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
    else:
        raise NotImplementedError(
            f"multivariable input is not yet supported in "
            f"conditional randomization. Recieved `z` of shape {z.shape}"
        )

    if alternative is None:
        warnings.warn(
            "The alternative hypothesis for conditional randomization"
            " is changing in the next major release of esda. We recommend"
            " setting alternative='two-sided', which will generally"
            " double the p-value returned."
            " To retain the current behavior, set alternative='directed'."
            " We strongly recommend moving to alternative='two-sided'.",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: replace this with 'two-sided' by next major release
        alternative = "directed"
    if alternative not in ("two-sided", "greater", "lesser", "directed", "folded"):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not one of the"
            " supported options: 'two-sided', 'greater', 'lesser', 'directed', 'folded'"
        )

    # paralellise over permutations?
    if seed is None:
        seed = np.random.randint(12345, 12345000)

    # we need to be careful to shuffle only *other* sites, not
    # the self-site. This means we need to
    # extract the self-weight, if any
    self_weights = adj_matrix.diagonal()
    # force the self-site weight to zero
    with warnings.catch_warnings():
        # massive changes to sparsity incur a cost, but it's not
        # large for simply changing the diag
        warnings.simplefilter("ignore")
        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()
    # extract the weights from a now no-self-weighted adj_matrix
    other_weights = adj_matrix.data.astype(z.dtype)  # cast is forced by @ in numba
    # use the non-self weight as the cardinality, since
    # this is the set we have to randomize.
    # if there is a self-neighbor, we need to *not* shuffle the
    # self neighbor, since conditional randomization conditions on site i.
    cardinalities = np.array((adj_matrix != 0).sum(1)).flatten()

    # Cumulative offsets into other_weights for random access by observation.
    # (Cannot reuse adj_matrix.indptr directly — diagonal was stripped above.)
    w_indptr = np.concatenate(([0], np.cumsum(cardinalities))).astype(np.int64)

    # n_jobs is accepted for API compatibility but Numba manages its own thread
    # pool; set NUMBA_NUM_THREADS to control parallelism instead.
    if n_jobs != 1:
        warnings.warn(
            "n_jobs is ignored by the prange engine; "
            "set NUMBA_NUM_THREADS to control thread count.",
            stacklevel=2,
        )

    max_card = int(cardinalities.max()) if len(cardinalities) > 0 else 1
    permuted_ids = vec_permutations(max_card, n, permutations, seed)

    p_sims, rlocals = _compute_prange(
        z,
        observed,
        cardinalities,
        self_weights,
        other_weights,
        w_indptr,
        permuted_ids,
        scaling,
        keep,
        stat_func,
        island_weight,
        _ALTERNATIVE_CODES[alternative],
    )

    return p_sims, rlocals


@njit(parallel=False, fastmath=True)
def compute_chunk(
    chunk_start: int,
    z_chunk: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    keep: bool,
    stat_func,
    island_weight: float,
    alternative: str,
):
    """
    Compute conditional randomisation for a single chunk
    ...

    Parameters
    ----------
    chunk_start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float
        Scaling value to apply to every local statistic
    keep : bool
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Returns
    -------
    larger : ndarray
        (n_chunk,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (n_chunk, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    chunk_n = z_chunk.shape[0]
    n_samples = z.shape[0]
    p_permutations, k_max_card = permuted_ids.shape
    p_sims = np.zeros((chunk_n,), dtype=np.float32)
    rlocals = np.empty((chunk_n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    mask = np.ones((n_samples,), dtype=np.int8) == 1
    wloc = 0

    for i in range(chunk_n):
        cardinality = cardinalities[i]
        if cardinality == 0:  # deal with islands
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            # we need to fix the self-weight to the first position
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            # this chomps the next `cardinality` weights off of `weights`
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        mask[chunk_start + i] = False
        rstats = stat_func(chunk_start + i, z, permuted_ids, weights_i, scaling)
        p_sims[i] = _permutation_significance(
            observed[i], rstats, alternative=alternative
        ).item()
        if keep:
            rlocals[i] = rstats

    return p_sims, rlocals


#######################################################################
#                   Parallel Implementation                           #
#######################################################################


@njit(fastmath=True)
def build_weights_offsets(cardinalities: np.ndarray, n_chunks: int):
    """
    Utility function to construct offsets into the weights
    flat data array found in the W.sparse.data object
    ...

    Parameters
    ----------
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    n_chunks : int
        Number of chunks to split the weights into

    Returns
    -------
    boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk
    """
    boundary_points = np.zeros((n_chunks + 1,), dtype=np.int64)
    n = cardinalities.shape[0]
    chunk_size = np.int64(n / n_chunks) + 1
    start = 0
    for i in range(n_chunks):
        advance = cardinalities[start : start + chunk_size].sum()
        boundary_points[i + 1] = boundary_points[i] + advance
        start += chunk_size
    return boundary_points


@njit(fastmath=True)
def chunk_generator(
    n_jobs: int,
    starts: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    w_boundary_points: np.ndarray,
):
    """
    Construct chunks to iterate over within numba in parallel
    ...

    Parameters
    ----------
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    starts : ndarray
        (n_chunks+1,) array of positional starts for each chunk
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    weights : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    w_boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk

    Yields
    ------
    start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed_chunk : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities_chunk : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    weights_chunk : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    """
    chunk_size = starts[1] - starts[0]
    for i in range(n_jobs):
        start = starts[i]
        z_chunk = z[start : (start + chunk_size)]
        self_weights_chunk = self_weights[start : (start + chunk_size)]
        observed_chunk = observed[start : (start + chunk_size)]
        cardinalities_chunk = cardinalities[start : (start + chunk_size)]
        w_chunk = other_weights[w_boundary_points[i] : w_boundary_points[i + 1]]
        yield (
            start,
            z_chunk,
            z,
            observed_chunk,
            cardinalities_chunk,
            self_weights_chunk,
            w_chunk,
        )


def parallel_crand(
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    n_jobs: int,
    keep: bool,
    stat_func,
    island_weight,
    alternative: str = "directed",
):
    """
    Conduct conditional randomization in parallel using numba
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float64
        Scaling value to apply to every local statistic
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.
    Returns
    -------
    larger : ndarray
        (N,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (N, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    from joblib import Parallel, delayed, parallel_backend

    n = z.shape[0]
    w_boundary_points = build_weights_offsets(cardinalities, n_jobs)
    chunk_size = n // n_jobs + 1
    starts = np.arange(n_jobs + 1) * chunk_size
    # ------------------------------------------------------------------
    # Set up output holders
    rlocals = np.empty((n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    # ------------------------------------------------------------------
    # Joblib parallel loop by chunks

    # construct chunks using a generator
    chunks = chunk_generator(
        n_jobs,
        starts,
        z,
        observed,
        cardinalities,
        self_weights,
        other_weights,
        w_boundary_points,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk)(
                *pars,
                permuted_ids,
                scaling,
                keep,
                stat_func,
                island_weight,
                alternative,
            )
            for pars in chunks
        )

    p_sims, rlocals = zip(*worker_out, strict=False)
    p_sims = np.hstack(p_sims).squeeze()
    rlocals = np.row_stack(rlocals).squeeze()
    return p_sims, rlocals


#######################################################################
#                   Local statistical functions                       #
#######################################################################


@njit(fastmath=False)
def _prepare_univariate(i, z, permuted_ids, weights_i):
    mask = np.ones_like(z, dtype=boolean)
    mask[i] = False
    z_no_i = z[mask]
    cardinality = len(weights_i)
    flat_permutation_ids = permuted_ids[:, :cardinality].flatten()
    zrand = z_no_i[flat_permutation_ids].reshape(-1, cardinality)
    return z[i], zrand


@njit(fastmath=False)
def _prepare_bivariate(i, z, permuted_ids, weights_i):
    zx = z[:, 0]
    zy = z[:, 1]

    cardinality = len(weights_i)

    mask = np.ones_like(zx, dtype=boolean)
    mask[i] = False
    zx_no_i = zy[mask]
    zy_no_i = zx[mask]

    flat_permutation_indices = permuted_ids[:, :cardinality].flatten()

    zxrand = zx_no_i[flat_permutation_indices].reshape(-1, cardinality)
    zyrand = zy_no_i[flat_permutation_indices].reshape(-1, cardinality)

    return zx[i], zxrand, zy[i], zyrand


@njit(fastmath=True)
def local(i, z, permuted_ids, weights_i, scaling):  # noqa: ARG001
    raise NotImplementedError
    # returns (k_permutations,) array of random statistics for observation i
