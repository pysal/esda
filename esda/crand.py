"""
Centralised conditional randomisation engine. Numba accelerated.
"""

import os
import warnings

import numpy as np

try:
    from numba import boolean, jit, njit, prange
except (ImportError, ModuleNotFoundError):

    def jit(*dec_args, **dec_kwargs):
        """
        decorator mimicking numba.jit
        """

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function

    njit = jit

    prange = range
    boolean = bool


__all__ = ["crand"]

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
    other_weights = adj_matrix.data
    # use the non-self weight as the cardinality, since
    # this is the set we have to randomize.
    # if there is a self-neighbor, we need to *not* shuffle the
    # self neighbor, since conditional randomization conditions on site i.
    cardinalities = np.array((adj_matrix != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    permuted_ids = vec_permutations(max_card, n, permutations, seed)

    if n_jobs != 1:
        try:
            import joblib  # noqa F401
        except (ModuleNotFoundError, ImportError):
            warnings.warn(
                f"Parallel processing is requested (n_jobs={n_jobs}),"
                f" but joblib cannot be imported. n_jobs will be set"
                f" to 1.",
                stacklevel=2,
            )
            n_jobs = 1

    if n_jobs == 1:
        larger, rlocals = compute_chunk(
            0,  # chunk start
            z,  # chunked z, for serial this is the entire data
            z,  # all z, for serial this is also the entire data
            observed,  # observed statistics
            cardinalities,  # cardinalities conforming to chunked z
            self_weights,  # n-length vector containing the self-weights.
            other_weights,  # flat weights buffer
            permuted_ids,  # permuted ids
            scaling,  # scaling applied to all statistics
            keep,  # whether or not to keep the local statistics
            stat_func,
            island_weight,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs > len(z):
            n_jobs = len(z)
        # Parallel implementation
        larger, rlocals = parallel_crand(
            z,
            observed,
            cardinalities,
            self_weights,
            other_weights,
            permuted_ids,
            scaling,
            n_jobs,
            keep,
            stat_func,
            island_weight,
        )

    low_extreme = (permutations - larger) < larger
    larger[low_extreme] = permutations - larger[low_extreme]
    p_sim = (larger + 1.0) / (permutations + 1.0)

    return p_sim, rlocals


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
    n = z.shape[0]
    larger = np.zeros((chunk_n,), dtype=np.int64)
    if keep:
        rlocals = np.empty((chunk_n, permuted_ids.shape[0]))
    else:
        rlocals = np.empty((1, 1))

    mask = np.ones((n,), dtype=np.int8) == 1
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
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]  # noqa E203
        wloc += cardinality
        mask[chunk_start + i] = False
        rstats = stat_func(chunk_start + i, z, permuted_ids, weights_i, scaling)
        if keep:
            rlocals[i] = rstats
        larger[i] = np.sum(rstats >= observed[i])
    return larger, rlocals


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
        advance = cardinalities[start : start + chunk_size].sum()  # noqa E203
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
        z_chunk = z[start : (start + chunk_size)]  # noqa E203
        self_weights_chunk = self_weights[start : (start + chunk_size)]  # noqa E203
        observed_chunk = observed[start : (start + chunk_size)]  # noqa E203
        cardinalities_chunk = cardinalities[start : (start + chunk_size)]  # noqa E203
        w_chunk = other_weights[
            w_boundary_points[i] : w_boundary_points[i + 1]  # noqa E203
        ]
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
    larger = np.zeros((n,), dtype=np.int64)
    if keep:
        rlocals = np.empty((n, permuted_ids.shape[0]))
    else:
        rlocals = np.empty((1, 1))
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
                *pars, permuted_ids, scaling, keep, stat_func, island_weight
            )
            for pars in chunks
        )
    larger, rlocals = zip(*worker_out)
    larger = np.hstack(larger).squeeze()
    rlocals = np.row_stack(rlocals).squeeze()
    return larger, rlocals


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
def local(i, z, permuted_ids, weights_i, scaling):
    raise NotImplementedError
    # returns (k_permutations,) array of random statistics for observation i
