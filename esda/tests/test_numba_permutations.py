import time

import numpy as np
import pytest
from libpysal.weights import lat2W

try:
    from numba import njit  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from esda.crand import _wloc_offsets, vec_permutations
from esda.moran import Moran_Local

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


@pytest.fixture(scope="session", autouse=True)
def numba_warmup():
    """Pre-compile all Numba JIT kernels used in this module.

    Without this, each test that hits a new type specialisation (float32,
    G_Local) pays the full ~80 s cold-start penalty.  Warming here means
    every subsequent call loads from the on-disk cache (<2 ms).
    """
    if not HAS_NUMBA:
        return
    from esda.getisord import G_Local

    w = lat2W(5, 5)
    rng = np.random.default_rng(0)
    y64 = rng.standard_normal(25)
    y32 = y64.astype(np.float32)
    # Warm float64 Moran_Local (prange kernel + vec_permutations)
    Moran_Local(y64, w, permutations=9, seed=0, alternative="two-sided")
    # Warm float32 specialisation (separate Numba type signature)
    Moran_Local(y32, w, permutations=9, seed=0, alternative="two-sided")
    # Warm G_Local (different stat_func, triggers its own JIT compile)
    G_Local(np.abs(y64) + 0.1, w, permutations=9, seed=0)


@requires_numba
def test_vec_permutations_seed_reproducibility():
    a = vec_permutations(4, 20, 99, 7)
    b = vec_permutations(4, 20, 99, 7)
    np.testing.assert_array_equal(a, b)


@requires_numba
def test_vec_permutations_different_seeds_differ():
    a = vec_permutations(4, 20, 99, 7)
    b = vec_permutations(4, 20, 99, 8)
    assert not np.array_equal(a, b)


@requires_numba
def test_wloc_offsets_basic():
    cards = np.array([2, 3, 1], dtype=np.int64)
    offsets = _wloc_offsets(cards)
    np.testing.assert_array_equal(offsets, [0, 2, 5, 6])


@requires_numba
def test_wloc_offsets_zero_cardinality():
    cards = np.array([0, 3, 0], dtype=np.int64)
    offsets = _wloc_offsets(cards)
    np.testing.assert_array_equal(offsets, [0, 0, 3, 3])


@requires_numba
def test_moran_local_seed_reproducibility():
    w = lat2W(10, 10)
    y = np.random.default_rng(0).standard_normal(100)
    m1 = Moran_Local(y, w, permutations=99, seed=42, alternative="two-sided")
    m2 = Moran_Local(y, w, permutations=99, seed=42, alternative="two-sided")
    np.testing.assert_array_equal(m1.p_sim, m2.p_sim)
    np.testing.assert_array_equal(m1.Is, m2.Is)


@requires_numba
def test_moran_local_different_seeds_differ():
    w = lat2W(8, 8)
    y = np.random.default_rng(0).standard_normal(64)
    m1 = Moran_Local(y, w, permutations=199, seed=1, alternative="two-sided")
    m2 = Moran_Local(y, w, permutations=199, seed=2, alternative="two-sided")
    assert not np.array_equal(m1.p_sim, m2.p_sim)


@requires_numba
def test_moran_local_p_sim_range():
    w = lat2W(10, 10)
    y = np.random.default_rng(5).standard_normal(100)
    m = Moran_Local(y, w, permutations=99, seed=99, alternative="two-sided")
    assert m.p_sim.shape == (100,)
    assert np.all(m.p_sim >= 0.0)
    assert np.all(m.p_sim <= 1.0)


@requires_numba
def test_moran_local_island_handling():
    from libpysal.weights import W

    neighbors = {0: [1], 1: [0, 2], 2: [1], 3: [], 4: [5], 5: [4]}
    w = W(neighbors)
    y = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.0])
    m = Moran_Local(y, w, permutations=49, seed=0, alternative="two-sided")
    assert m.p_sim.shape == (6,)


@requires_numba
def test_moran_local_float32_float64_parity():
    w = lat2W(10, 10)
    rng = np.random.default_rng(11)
    y = rng.standard_normal(100)
    m32 = Moran_Local(
        y.astype(np.float32), w, permutations=99, seed=7, alternative="two-sided"
    )
    m64 = Moran_Local(
        y.astype(np.float64), w, permutations=99, seed=7, alternative="two-sided"
    )
    np.testing.assert_allclose(m32.Is, m64.Is, rtol=1e-4)


@requires_numba
def test_getis_ord_local_seed_reproducibility():
    from esda.getisord import G_Local

    w = lat2W(10, 10)
    y = np.abs(np.random.default_rng(3).standard_normal(100)) + 0.1
    g1 = G_Local(y, w, permutations=99, seed=42)
    g2 = G_Local(y, w, permutations=99, seed=42)
    np.testing.assert_array_equal(g1.p_sim, g2.p_sim)


@requires_numba
def test_njobs1_prange_equals_njobs2_joblib():
    """Numerical equivalence: the n_jobs=1 prange kernel and the n_jobs>1 joblib
    path must produce identical p_sims given the same seed.

    Architecture guarantee: both paths use the same ``permuted_ids`` array
    (determined solely by ``seed``).  The per-observation statistic is
    independent across observations, so chunking cannot affect results.
    This simultaneously confirms there is no nested-parallelism hazard:
    ``n_jobs=1`` uses ``compute_chunk_parallel`` (Numba thread pool) while
    ``n_jobs>1`` uses ``compute_chunk`` (``parallel=False``) inside loky workers
    started with ``inner_max_num_threads=1``.  The paths are mutually exclusive.
    """
    pytest.importorskip("joblib")
    w = lat2W(8, 8)
    y = np.random.default_rng(0).standard_normal(w.n)
    kw = dict(permutations=99, seed=42, alternative="two-sided")
    m1 = Moran_Local(y, w, n_jobs=1, **kw)
    m2 = Moran_Local(y, w, n_jobs=2, **kw)
    np.testing.assert_array_equal(m1.p_sim, m2.p_sim)


@requires_numba
def test_cache_true_determinism():
    """cache=True must not alter results: two calls with the same seed must
    return bit-for-bit identical p_sims.  The first call may write the JIT
    cache; the second call loads it.  Any divergence indicates a caching bug.
    """
    w = lat2W(8, 8)
    y = np.random.default_rng(1).standard_normal(w.n)
    kw = dict(permutations=99, seed=99, alternative="two-sided")
    m1 = Moran_Local(y, w, **kw)
    m2 = Moran_Local(y, w, **kw)
    np.testing.assert_array_equal(m1.p_sim, m2.p_sim)


@requires_numba
@pytest.mark.slow
def test_moran_local_parallel_speedup():
    w = lat2W(50, 50)
    y = np.random.default_rng(42).standard_normal(2500)

    _ = Moran_Local(y, w, permutations=9, seed=0)  # warm JIT cache

    t0 = time.perf_counter()
    Moran_Local(y, w, permutations=199, seed=0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"Expected <5s on warm JIT, got {elapsed:.2f}s"
