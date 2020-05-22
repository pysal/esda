# --------------------------------------------------------------------#
#                  Performance Optimisations                          #
# --------------------------------------------------------------------#

import os
import numpy as np
from numba import njit, jit, prange, boolean
from joblib import Parallel, delayed, parallel_backend
import tempfile

__all__ = ["crand"]

#######################################################################
#                   Utilities for all functions                       #
#######################################################################


@njit(fastmath=True)
def vec_permutations(max_card: int, n: int, k_replications: int):
    result = np.empty((k_replications, max_card), dtype=np.int64)
    for k in prange(k_replications):
        result[k] = np.random.choice(n - 1, size=max_card, replace=False)
    return result


def crand(z, w, observed, permutations, keep, n_jobs, stat_func):
    """
    Conduct conditional randomization of a given input using the provided simulation function.
    """
    cardinalities = np.array((w.sparse != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    n = len(z)
    if z.ndim == 2:
        if z.shape[1] == 2:
            # assume that matrix is [X Y]
            scaling = (n - 1) / (z[:,0] * z[:,0]).sum()
        elif z.shape[1] == 1:
            scaling = (n - 1) / (z * z).sum()
        else:
            raise NotImplementedError(f'multivariable input is not yet supported in '
                                      f'conditional randomization. Recieved `z` of shape {z.shape}')
    elif z.ndim == 1:
        scaling = (n - 1) / (z * z).sum()
    else:
        raise NotImplementedError(f'multivariable input is not yet supported in '
                                  f'conditional randomization. Recieved `z` of shape {z.shape}')

    # paralellise over permutations?
    permuted_ids = vec_permutations(max_card, n, permutations)

    if n_jobs == 1:
        larger, rlocals = compute_chunk(
            0,  # chunk start
            z,  # chunked z, for serial this is the entire data
            z,  # all z, for serial this is also the entire data
            observed,  # observed statistics
            cardinalities,  # cardinalities conforming to chunked z
            w.sparse.data,  # flat weights buffer
            permuted_ids,  # permuted ids
            scaling,  # scaling applied to all statistics
            keep,  # whether or not to keep the local statistics
            stat_func,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        # Parallel implementation
        larger, rlocals = parallel_crand(
            z,
            observed,
            cardinalities,
            w.sparse.data,
            permuted_ids,
            scaling,
            n_jobs,
            keep,
            stat_func,
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
    weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    keep: bool,
    stat_func,
):
    """
    Arguments
    ---------
    chunk_start :   int
        the starting index the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray 
        (n_chunk,) array containing the chunk of data to process.
    z   :   numpy.ndarray
        (n_observations,1) array containing all of the rest of the data
    observed    :   numpy.ndarray
        (n_chunk,) array observed local statistics within the chunk
    cardinalities   :   numpy.ndarray
        (n_chunk,) array containing the cardinalities pertaining to each element. 
    weights :   numpy.ndarray
        array containing the weights within the chunk. This is as long as the
        sum of cardinalities within this chunk.
    permuted_ids    :   numpy.ndarray
        (n_observations,n_permutations) array containing the permutation 
        ids to use to construct random realizations of the statistic
    scaling :   float
        scaling value to apply to every local statistic
    keep : bool
        whether or not to keep the simulated statistics
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
        ### this chomps the first `cardinality` weights off of `weights`
        weights_i = weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        z_chunk_i = z_chunk[i]
        mask[chunk_start + i] = False
        z_no_i = z[
            mask,
        ]
        # ------
        # flat_permuted_ids = permuted_ids[:, :cardinality].flatten()
        # rstats = z_no_i[flat_permuted_ids].reshape(-1, cardinality).dot(weights_i)
        # mask[chunk_start + i] = True
        # rstats *= z_chunk_i * scaling
        # ------
        rstats = stat_func(chunk_start + i, z, permuted_ids, weights_i, scaling)
        if keep:
            rlocals[i,] = rstats
        larger[i] = np.sum(rstats >= observed[i])
    return larger, rlocals


#######################################################################
#                   Parallel Implementation                           #
#######################################################################


@njit(fastmath=True)
def build_weights_offsets(cardinalities: np.ndarray, n_chunks: int):
    """
    This is a utility function to construct offsets into the weights
    flat data array found in the W.sparse.data object.
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
    weights: np.ndarray,
    w_boundary_points: np.ndarray,
):
    """
    Construct chunks to iterate over in numba
    """
    chunk_size = starts[1] - starts[0]
    for i in range(n_jobs):
        start = starts[i]
        z_chunk = z[start : (start + chunk_size)]
        observed_chunk = observed[start : (start + chunk_size)]
        cardinalities_chunk = cardinalities[start : (start + chunk_size)]
        w_chunk = weights[w_boundary_points[i] : w_boundary_points[i + 1]]
        yield (
            start,
            z_chunk,
            z,
            observed_chunk,
            cardinalities_chunk,
            w_chunk,
        )


def parallel_crand(
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    n_jobs: int,
    keep: bool,
    stat_func,
):
    """
    conduct conditional randomization in parallel using numba
    """
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
        n_jobs, starts, z, observed, cardinalities, weights, w_boundary_points,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk)(*pars, permuted_ids, scaling, keep, stat_func)
            for pars in chunks
        )
    larger, rlocals = zip(*worker_out)
    larger = np.hstack(larger).flatten()
    rlocals = np.hstack(rlocals).flatten()
    return larger, rlocals


#######################################################################
#                   Local statistical functions                       #
#######################################################################


@njit
def _prepare_univariate(i, z, permuted_ids, weights_i):
    mask = np.ones_like(z, dtype=boolean)
    mask[i] = False
    z_no_i = z[mask]
    cardinality = len(weights_i)
    flat_permutation_ids = permuted_ids[:, :cardinality].flatten()
    zrand = z_no_i[flat_permutation_ids].reshape(-1, cardinality)
    return z[i], zrand


@njit
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
def local_moran(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return zi * (zrand @ weights_i) * scaling


@njit(fastmath=True)
def local_geary(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return np.power(zrand - zi, 2) @ weights_i * scaling


@njit(fastmath=True)
def local_gamma(i, z, permuted_ids, weights_i, scaling):
    zi, zrand = _prepare_univariate(i, z, permuted_ids, weights_i)
    return (zi * zrand) @ weights_i * scaling


@njit(fastmath=True)
def local_spatial_pearson(i, z, permuted_ids, weights_i, scaling):
    zxi, zxrand, zyi, zyrand = _prepare_bivariate(i, z, permuted_ids, weights_i)
    return (zyrand @ weights_i) * (zxrand @ weights_i) * scaling


@njit(fastmath=True)
def local_wartenburg(i, z, permuted_ids, weights_i, scaling):
    zx = z[:, 0]
    zy = z[:, 1]
    zyi, zyrand = _prepare_univariate(i, zy, permuted_ids, weights_i)
    return zx[i] * (zyrand @ weights_i) * scaling


@njit(fastmath=True)
def local_join_count():
    raise NotImplementedError


def local(i, z, permuted_ids, weights_i, scaling):
    raise NotImplementedError
    # returns (k_permuotations,) array of random statistics for observation i##
