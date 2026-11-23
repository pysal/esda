"""Regression tests for local join counts with islands and id ordering (GH531)."""

import numpy as np
import pytest
from libpysal import graph
from libpysal.weights import W
from libpysal.weights.util import lat2W

from ..join_counts_local import Join_Counts_Local
from ..join_counts_local_bv import Join_Counts_Local_BV
from ..join_counts_local_mv import Join_Counts_Local_MV

SEED = 12345


def _w_with_island():
    """3x3 rook lattice (ids 0-8) plus an island with id 9."""
    neighbors = dict(lat2W(3, 3).neighbors)
    neighbors[9] = []
    return W(neighbors, silence_warnings=True)


parametrize_island = pytest.mark.parametrize(
    "w",
    [_w_with_island(), graph.Graph.from_W(_w_with_island())],
    ids=["W", "Graph"],
)

Y = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=float)
X = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)


def _fit(cls, w, drop_islands):
    est = cls(connectivity=w, seed=SEED, drop_islands=drop_islands)
    if cls is Join_Counts_Local:
        return est.fit(Y, permutations=99)
    if cls is Join_Counts_Local_BV:
        return est.fit(X, Y, permutations=99)
    return est.fit([X, Y], permutations=99)


@parametrize_island
@pytest.mark.parametrize("drop_islands", [True, False])
@pytest.mark.parametrize(
    "cls", [Join_Counts_Local, Join_Counts_Local_BV, Join_Counts_Local_MV]
)
def test_islands(cls, w, drop_islands):
    res = _fit(cls, w, drop_islands)
    assert res.LJC.shape == (10,)
    assert res.p_sim.shape == (10,)
    # the island has no joins, hence LJC 0 and an undefined p-value
    assert res.LJC[-1] == 0
    assert np.isnan(res.p_sim[-1])


@parametrize_island
@pytest.mark.parametrize(
    "cls", [Join_Counts_Local, Join_Counts_Local_BV, Join_Counts_Local_MV]
)
def test_drop_islands_does_not_change_non_islands(cls, w):
    dropped = _fit(cls, w, True)
    kept = _fit(cls, w, False)
    np.testing.assert_array_equal(dropped.LJC, kept.LJC)


def test_unsorted_id_order():
    # path c - a - b with id_order deliberately not sorted
    w = W({"c": ["a"], "a": ["c", "b"], "b": ["a"]}, id_order=["c", "a", "b"])
    y = np.array([0, 1, 1], dtype=float)  # c=0, a=1, b=1
    res = Join_Counts_Local(connectivity=w, drop_islands=False).fit(y, permutations=0)
    np.testing.assert_array_equal(res.LJC, [0, 1, 1])
