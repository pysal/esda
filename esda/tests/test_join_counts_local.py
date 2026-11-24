"""Regression tests for join counts with islands and id ordering (GH531)."""

import numpy as np
import pytest
from libpysal import graph
from libpysal.weights import W
from libpysal.weights.util import lat2W

from ..join_counts import Join_Counts
from ..join_counts_local import Join_Counts_Local
from ..join_counts_local_bv import Join_Counts_Local_BV
from ..join_counts_local_mv import Join_Counts_Local_MV

SEED = 12345
LOCAL = [Join_Counts_Local, Join_Counts_Local_BV, Join_Counts_Local_MV]


def _w_with_island():
    """3x3 rook lattice (ids 0-8) plus an island with id 9."""
    neighbors = dict(lat2W(3, 3).neighbors)
    neighbors[9] = []
    return W(neighbors, silence_warnings=True)


def _w_unsorted():
    """Path c - a - b with an id order that is not sorted."""
    return W({"c": ["a"], "a": ["c", "b"], "b": ["a"]}, id_order=["c", "a", "b"])


parametrize_island = pytest.mark.parametrize(
    "w",
    [_w_with_island(), graph.Graph.from_W(_w_with_island())],
    ids=["W", "Graph"],
)
parametrize_unsorted = pytest.mark.parametrize(
    "w",
    [_w_unsorted(), graph.Graph.from_W(_w_unsorted())],
    ids=["W", "Graph"],
)

# the island (last observation) is 1 in both variables
Y = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=float)
X = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)


def _fit(cls, w, **kwargs):
    est = cls(connectivity=w, seed=SEED, **kwargs)
    if cls is Join_Counts_Local:
        return est.fit(Y, permutations=99)
    if cls is Join_Counts_Local_BV:
        return est.fit(X, Y, permutations=99)
    return est.fit([X, Y], permutations=99)


def _statistic(cls, w, drop_islands):
    if cls is Join_Counts_Local:
        return cls._statistic(Y, w, drop_islands)
    if cls is Join_Counts_Local_BV:
        return cls._statistic(X, Y, w, "CLC", drop_islands)
    return cls._statistic([X, Y], w, drop_islands)


@parametrize_island
@pytest.mark.parametrize("cls", LOCAL)
def test_local_islands(cls, w):
    res = _fit(cls, w)
    assert res.LJC.shape == (10,)
    assert res.p_sim.shape == (10,)
    # the island has no joins, hence LJC 0 and an undefined p-value
    assert res.LJC[-1] == 0
    assert np.isnan(res.p_sim[-1])


@parametrize_island
@pytest.mark.parametrize("cls", LOCAL)
def test_local_statistic_islands_kept_or_dropped(cls, w):
    dropped = _statistic(cls, w, True)
    kept = _statistic(cls, w, False)
    assert dropped.shape == (10,)
    np.testing.assert_array_equal(dropped, kept)


@parametrize_island
@pytest.mark.parametrize("drop_islands", [True, False])
@pytest.mark.parametrize("cls", LOCAL)
def test_local_drop_islands_deprecated(cls, w, drop_islands):
    with pytest.warns(FutureWarning, match="drop_islands"):
        res = _fit(cls, w, drop_islands=drop_islands)
    np.testing.assert_array_equal(res.LJC, _fit(cls, w).LJC)


@parametrize_unsorted
def test_local_unsorted_id_order(w):
    y = np.array([0, 1, 1], dtype=float)  # c=0, a=1, b=1
    res = Join_Counts_Local(connectivity=w).fit(y, permutations=0)
    np.testing.assert_array_equal(res.LJC, [0, 1, 1])


@parametrize_island
def test_global_islands(w):
    jc = Join_Counts(Y, w, permutations=0)
    # same counts as the lattice without the island
    ref = Join_Counts(Y[:9], lat2W(3, 3), permutations=0)
    assert (jc.bb, jc.ww, jc.bw) == (ref.bb, ref.ww, ref.bw)


@parametrize_island
@pytest.mark.parametrize("drop_islands", [True, False])
def test_global_drop_islands_deprecated(w, drop_islands):
    with pytest.warns(FutureWarning, match="drop_islands"):
        jc = Join_Counts(Y, w, permutations=0, drop_islands=drop_islands)
    ref = Join_Counts(Y, w, permutations=0)
    assert (jc.bb, jc.ww, jc.bw) == (ref.bb, ref.ww, ref.bw)
