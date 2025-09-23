import numpy as np
import pytest

from esda.crand import vec_permutations


def test_vec_permutations_basic():
    """Test vec_permutations with basic input."""
    result = vec_permutations(5, 24, 10, seed=12345)
    assert len(result) == 10, "Should create 10 permutations"
    assert all(isinstance(item, np.ndarray) for item in result), (
        "Each permutation should be an array"
    )
    expected = np.array(
        [
            [10, 19, 8, 17, 3],
            [21, 16, 8, 15, 10],
            [18, 4, 20, 3, 13],
            [4, 9, 20, 7, 15],
            [16, 15, 4, 21, 0],
            [9, 10, 8, 1, 11],
            [20, 11, 6, 15, 0],
            [10, 3, 11, 14, 16],
            [12, 6, 8, 7, 19],
            [18, 19, 4, 13, 15],
        ]
    )
    np.testing.assert_array_equal(result, expected)
