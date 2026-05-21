"""
Benchmark comparing the original crand engine (compute_chunk / parallel_crand)
against the new _compute_prange engine.

Usage:
    uv run --extra plus python benchmark_crand.py

The original engine is exercised by calling compute_chunk directly with a
pre-generated permuted_ids array.  The new engine is called via crand() which
now routes through _compute_prange.  Both paths use the same data, seed, and
number of permutations.

Results are printed as a table and saved to benchmark_crand_results.json.
"""

import json
import time
import warnings

import libpysal
import numpy as np

import esda
from esda.crand import (
    _compute_prange,
    _ALTERNATIVE_CODES,
    compute_chunk,
    vec_permutations,
)
from esda.moran import _moran_local_crand


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_inputs(n, seed=42):
    """Return (z, w, observed, cardinalities, self_weights, other_weights, w_indptr)."""
    rng = np.random.default_rng(seed)
    w = libpysal.weights.lat2W(int(n**0.5), int(n**0.5))
    w.transform = "r"
    y = rng.standard_normal(len(w.id_order))
    z = (y - y.mean()) / y.std()
    scaling = (len(z) - 1) / (z * z).sum()
    observed = np.array([
        scaling * z[i] * (z[list(w.neighbors[w.id_order[i]])] *
                          np.array(list(w.weights[w.id_order[i]]))).sum()
        for i in range(len(z))
    ])
    return z, w, observed, scaling


def _run_original(z, w, observed, scaling, permutations, seed):
    """Run the original compute_chunk path."""
    adj = w.sparse.copy()
    self_weights = adj.diagonal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj.setdiag(0)
        adj.eliminate_zeros()
    other_weights = adj.data.astype(z.dtype)
    cardinalities = np.array((adj != 0).sum(1)).flatten()
    max_card = int(cardinalities.max())
    permuted_ids = vec_permutations(max_card, len(z), permutations, seed)
    p_sims, rlocals = compute_chunk(
        0, z, z, observed, cardinalities, self_weights, other_weights,
        permuted_ids, scaling, False, _moran_local_crand, 0,
        alternative="directed",
    )
    return p_sims


def _run_prange(z, w, observed, scaling, permutations, seed):
    """Run the new _compute_prange path."""
    adj = w.sparse.copy()
    self_weights = adj.diagonal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj.setdiag(0)
        adj.eliminate_zeros()
    other_weights = adj.data.astype(z.dtype)
    cardinalities = np.array((adj != 0).sum(1)).flatten()
    w_indptr = np.concatenate(([0], np.cumsum(cardinalities))).astype(np.int64)
    max_card = int(cardinalities.max())
    permuted_ids = vec_permutations(max_card, len(z), permutations, seed)
    p_sims, rlocals = _compute_prange(
        z, observed, cardinalities, self_weights, other_weights,
        w_indptr, permuted_ids, scaling, False,
        _moran_local_crand, 0, _ALTERNATIVE_CODES["directed"],
    )
    return p_sims


def _time(fn, *args, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)  # best-of-n to reduce noise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

SIZES = [100, 400, 900, 2500, 10000, 40000]
PERMUTATIONS = 999
SEED = 12345
WARMUP_N = 100  # JIT warm-up size


def main():
    print("Warming up Numba JIT...")
    z, w, observed, scaling = _build_inputs(WARMUP_N)
    _run_original(z, w, observed, scaling, 99, SEED)
    _run_prange(z, w, observed, scaling, 99, SEED)
    print("Done.\n")

    print(f"{'n':>6}  {'original (s)':>13}  {'prange (s)':>11}  {'speedup':>8}")
    print("-" * 46)

    results = []
    for n in SIZES:
        z, w, observed, scaling = _build_inputs(n)

        t_orig = _time(_run_original, z, w, observed, scaling, PERMUTATIONS, SEED)
        t_prange = _time(_run_prange, z, w, observed, scaling, PERMUTATIONS, SEED)
        speedup = t_orig / t_prange

        print(f"{n:>6}  {t_orig:>13.3f}  {t_prange:>11.3f}  {speedup:>7.1f}x")
        results.append(
            dict(n=n, permutations=PERMUTATIONS, original_s=t_orig,
                 prange_s=t_prange, speedup=speedup)
        )

    out = "benchmark_crand_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
