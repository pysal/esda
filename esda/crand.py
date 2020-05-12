# --------------------------------------------------------------------#
#                  Performance Optimisations                          #
# --------------------------------------------------------------------#

import os
import numpy as np
from numba import njit, jit, prange 
from joblib import Parallel, delayed, parallel_backend

#######################################################################
#                   Utilities for all functions                       #
#######################################################################

@njit(fastmath=True)
def vec_permutations(max_card: int, n: int, k_replications: int):
    result = np.empty((k_replications, max_card), dtype=np.int64)
    for k in prange(k_replications):
        result[k] = np.random.choice(n - 1, size=max_card, replace=False)
    return result

def crand(z, w, observed, permutations, keep, n_jobs):
    """
    Conduct conditional randomization of a given input using the provided simulation function.
    """
    cardinalities = np.array((w.sparse != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    n = len(z)
    scaling = (n - 1) / (z * z).sum()

    # paralellise over permutations?
    permuted_ids = vec_permutations(max_card, n, permutations)

    if n_jobs == 1:
        larger, rlocals = serial_crand(
            0,
            z,
            z,
            observed,
            cardinalities,
            w.sparse.data,
            permuted_ids,
            scaling,
            max_card,
            keep,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        # Parallel implementation
        larger, rlocals = parallel_crand(
            z,
            Is,
            cardinalities,
            w.sparse.data,
            permuted_ids,
            scaling,
            max_card,
            n_jobs,
            keep,
        )

    low_extreme = (permutations - larger) < larger
    larger[low_extreme] = permutations - larger[low_extreme]
    p_sim = (larger + 1.0) / (permutations + 1.0)

    return p_sim, rlocals

#######################################################################
#                   Serial Implementation                             #
#######################################################################

@njit(parallel=False, fastmath=True)
def serial_crand(
    chunk_start: int,  # Obs. i the chunk starts in
    z_chunk: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    max_card: int,
    keep: bool,
):
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
        flat_permuted_ids = permuted_ids[:, :cardinality].flatten()
        rstats = z_no_i[flat_permuted_ids].reshape(-1, cardinality).dot(weights_i)
        # ------
        mask[chunk_start + i] = True
        rstats *= z_chunk_i * scaling
        if keep:
            rlocals[i,] = rstats
        larger[i] = np.sum(rstats >= observed[i])
    return larger, rlocals

#######################################################################
#                   Parallel Implementation                           #
#######################################################################

@njit(fastmath=True)
def build_weights_offsets(cardinalities: np.ndarray, 
                          n_chunks:int):
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
        n_jobs :int ,
        starts: np.ndarray,
        z : np.ndarray,
        observed: np.ndarray,
        cardinalities: np.ndarray,
        weights: np.ndarray,
        w_boundary_points: np.ndarray,
        permuted_ids: np.ndarray,
        scaling: float,
        max_card: int,
        keep: bool,
):
    """
    Construct chunks to iterate over in numba
    """
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
            permuted_ids,
            scaling,
            max_card,
            keep,
        )


def parallel_crand(
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    max_card: int,
    n_jobs: int,
    keep: bool,
):
    """
    conduct conditional randomization in parallel using numba
    """
    n = z.shape[0]
    w_boundary_points = chunk_weights(cardinalities, n_jobs)
    chunk_size = np.int64(n / n_jobs) + 1
    starts = np.zeros((n_jobs + 1,), dtype=np.int64)
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + chunk_size
    # ------------------------------------------------------------------
    # Set up output holders
    larger = np.zeros((n,), dtype=np.int64)
    if keep:
        rlocals = np.empty((n, permuted_ids.shape[0]))
    else:
        rlocals = np.empty((1, 1))
    # ------------------------------------------------------------------
    # Joblib parallel loop by chunks

    # prepare memory map for data shared across all cores
    tmp = tempfile.SpooledTemporaryFile()
    zmm = np.memmap(tmp, shape=z.shape)
    zmm[:] = z[:]

    # construct chunks using a generator
    chunks = chunk_generator(
        n_jobs,
        starts,
        zmm,
        observed,
        cardinalities,
        weights,
        w_boundary_points,
        permuted_ids,
        scaling,
        max_card,
        keep,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(serial_crand)(*pars) for pars in chunks
        )

    for i in range(len(worker_out)):
        larger_chunk, rlocals_chunk = worker_out[i]
        start = chunks[i][0]
        larger[start : start + chunk_size] = larger_chunk
        if keep:
            rlocals[start : start + chunk_size] = rlocals_chunk
    tmp.close()
    return larger, rlocals


def crand(z, w, observed, permutations, keep, n_jobs):
    """
    Conduct conditional randomization of a given input using the provided simulation function.
    """
    cardinalities = np.array((w.sparse != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    n = len(z)
    scaling = (n - 1) / (z * z).sum()

    # paralellise over permutations?
    permuted_ids = vec_permutations(max_card, n, permutations)

    if n_jobs == 1:
        larger, rlocals = serial_crand(
            0,
            z,
            z,
            observed,
            cardinalities,
            w.sparse.data,
            permuted_ids,
            scaling,
            max_card,
            keep,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        # Parallel implementation
        larger, rlocals = parallel_crand(
            z,
            Is,
            cardinalities,
            w.sparse.data,
            permuted_ids,
            scaling,
            max_card,
            n_jobs,
            keep,
        )

    low_extreme = (permutations - larger) < larger
    larger[low_extreme] = permutations - larger[low_extreme]
    p_sim = (larger + 1.0) / (permutations + 1.0)

    return p_sim, rlocals
