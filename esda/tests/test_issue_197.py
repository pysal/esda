import numpy as np
import pytest

from esda.crand import crand
from libpysal.weights import lat2W


def test_issue_197_crand_dtype_validation_float32():
    """
    Test for issue #197: crand should validate dtype compatibility before numba calls.

    The issue occurs when z (standardized values) has a different dtype than float64.
    Numba is strict about dtype compatibility and requires float64.

    This test verifies that crand raises a clear dtype error for float32 inputs.
    """
    w = lat2W(3, 3)
    n = w.n

    z_float32 = np.random.randn(n).astype(np.float32)
    observed = np.random.randn(w.n)

    assert z_float32.dtype == np.float32

    with pytest.raises(TypeError) as exc_info:
        crand(
            z_float32,
            w,
            observed,
            permutations=3,
            keep=False,
            n_jobs=1,
            stat_func=None,
        )

    error_msg = str(exc_info.value).lower()
    assert "dtype" in error_msg or "float64" in error_msg or "float32" in error_msg, (
        f"Expected a dtype-related error, but got: {exc_info.value}"
    )
