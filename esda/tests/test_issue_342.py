import warnings

import numpy as np
import pandas

from esda import adbscan


def test_issue_342_dtype_futurewarning_in_remap_lbls():
    """
    Test for issue #342: remap_lbls emits pandas FutureWarning on dtype mismatch.

    The warning occurs when solus contains int64 but the remap includes NaN (float64).
    This test should trigger the warning before the fix and pass after.
    """
    db = pandas.DataFrame({"X": [0, 0.1, 4, 6, 5], "Y": [0, 0.2, 5, 7, 5]})

    solus = pandas.DataFrame(
        {
            "rep-00": pandas.Series([0, 0, -1, 7, 7], dtype="int64"),
            "rep-01": pandas.Series([4, 4, -1, 6, 6], dtype="int64"),
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always", category=FutureWarning)
        lbls = adbscan.remap_lbls(solus, db, n_jobs=1)

        future_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, FutureWarning)
            and "incompatible dtype" in str(warning.message)
        ]

        assert len(future_warnings) == 0, (
            f"FutureWarning about incompatible dtype detected. "
            f"Message: {future_warnings[0].message if future_warnings else 'N/A'}"
        )

    assert lbls is not None
    assert lbls.shape == solus.shape
