"""
Utilities for handling missing values in spatial analysis.

This module provides utilities to manage missing (NaN) values in spatial data
before and after statistical analysis. Following best practices from spatial
statistics literature (Anselin 1995, Rey & Anselin 2010), missing data should
be handled transparently rather than silently propagated through computations.

References
----------
Anselin, L. (1995). Local indicators of spatial association-LISA.
    Geographical Analysis, 27(2), 93-115.
Rey, S. J., & Anselin, L. (2010). PySAL: A Python library of spatial analytical methods.
    Journal of Statistical Software, 42(2), 1-24.
Little, R. J. A., & Rubin, D. B. (2002). Statistical analysis with missing data.
    John Wiley & Sons.
"""

import warnings

import numpy as np
import pandas as pd
from libpysal.weights import W, w_subset


def identify_missing_observations(y, raise_error=False):
    """
    Identify observations with missing values in the data vector.

    Parameters
    ----------
    y : array-like
        Data vector to check for missing values.
    raise_error : bool, default False
        If True, raise ValueError when missing values are detected.
        If False, return mask array identifying missing observations.

    Returns
    -------
    mask : ndarray
        Boolean array where True indicates non-missing values.
        Only returned if raise_error=False.

    Raises
    ------
    ValueError
        If raise_error=True and missing values are detected in y.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 2.0, np.nan, 4.0])
    >>> mask = identify_missing_observations(y)
    >>> mask
    array([ True,  True, False,  True])
    """
    y_array = np.asarray(y)
    mask = ~np.isnan(y_array)

    if raise_error and not mask.all():
        n_missing = np.sum(~mask)
        raise ValueError(
            f"Input data contains {n_missing} missing value(s). "
            "Consider using mask_missing_observations() to remove them "
            "or use raise_error=False to get a mask of valid observations."
        )

    if not raise_error:
        return mask

    return None


def mask_missing_observations(y, w, return_indices=False):
    """
    Remove observations with missing values from data and weights.

    This function implements a standard approach in spatial statistics:
    exclude observations with missing attribute values before analysis,
    then restore missing indicators to results afterward.

    Parameters
    ----------
    y : array-like
        Data vector with potential missing values (NaN).
    w : libpysal.weights.W
        Spatial weights object aligned with y.
    return_indices : bool, default False
        If True, also return indices of non-missing observations.

    Returns
    -------
    y_clean : ndarray
        Data vector with missing values removed.
    w_clean : libpysal.weights.W
        Weights object subset to non-missing observations only.
    indices : ndarray, optional
        Indices of non-missing observations (only if return_indices=True).

    Raises
    ------
    ValueError
        If all observations are missing.
    ValueError
        If no observations remain after removing missing values.

    Notes
    -----
    This approach is equivalent to creating a weights matrix from a geography
    with "holes" (Anselin 1995). It assumes analysis should be performed only
    on observations with complete data.

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from esda.missing import mask_missing_observations
    >>> y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    >>> w = libpysal.weights.util.lattice_weights(shape=(5,1))
    >>> y_clean, w_clean, idx = mask_missing_observations(
    ...     y, w, return_indices=True
    ... )
    >>> y_clean
    array([1., 2., 4., 5.])
    >>> idx
    array([0, 1, 3, 4])
    """
    y_array = np.asarray(y)

    if y_array.size == 0:
        raise ValueError("Input data is empty.")

    mask = identify_missing_observations(y_array, raise_error=False)

    if not mask.any():
        raise ValueError("All observations have missing values.")

    n_removed = np.sum(~mask)
    if n_removed > 0:
        warnings.warn(
            f"Removing {n_removed} observation(s) with missing values. "
            f"Analysis will proceed with {np.sum(mask)} observations.",
            UserWarning,
            stacklevel=2,
        )

    y_clean = y_array[mask]
    indices = np.where(mask)[0]

    # Subset weights to non-missing observations
    w_clean = w_subset(w, indices)

    if return_indices:
        return y_clean, w_clean, indices

    return y_clean, w_clean


def restore_missing_observations(
    result_array, original_length, missing_indices, fill_value=np.nan
):
    """
    Restore missing value indicators to result arrays.

    After performing spatial analysis on a subset of observations,
    this function restores the original geometry by filling missing
    observations with a specified value (typically NaN).

    Parameters
    ----------
    result_array : array-like
        Result array from analysis performed on non-missing observations.
    original_length : int
        Length of the original data (including missing observations).
    missing_indices : array-like
        Indices of non-missing observations in original data.
    fill_value : float or str, default np.nan
        Value to use for missing observations in output.
        Can also be "Undefined" for categorical classification.

    Returns
    -------
    full_result : ndarray
        Result array with original length, with fill_value at missing positions.

    Raises
    ------
    ValueError
        If result_array length doesn't match missing_indices length.

    Examples
    --------
    >>> import numpy as np
    >>> from esda.missing import restore_missing_observations
    >>> result = np.array([0.5, 0.3, 0.7])
    >>> original_length = 5
    >>> missing_indices = np.array([0, 1, 3])
    >>> restored = restore_missing_observations(
    ...     result, original_length, missing_indices
    ... )
    >>> restored
    array([0.5, 0.3,  nan, 0.7,  nan])
    """
    result_array = np.asarray(result_array)
    missing_indices = np.asarray(missing_indices)

    if len(result_array) != len(missing_indices):
        raise ValueError(
            f"Length of result_array ({len(result_array)}) does not match "
            f"length of missing_indices ({len(missing_indices)})"
        )

    # Initialize output array with fill_value
    if isinstance(fill_value, str) and fill_value == "Undefined":
        full_result = np.full(original_length, "Undefined", dtype=object)
        full_result[missing_indices] = result_array
    else:
        full_result = np.full(original_length, fill_value, dtype=result_array.dtype)
        full_result[missing_indices] = result_array

    return full_result


def missing_data_summary(y):
    """
    Generate a diagnostic summary of missing data patterns.

    Provides information about the extent and distribution of missing values
    that may affect spatial analysis.

    Parameters
    ----------
    y : array-like
        Data vector to analyze.

    Returns
    -------
    summary : dict
        Dictionary with keys:
        - 'n_total': Total number of observations
        - 'n_missing': Number of missing observations
        - 'n_valid': Number of valid observations
        - 'pct_missing': Percentage of missing observations
        - 'complete': True if no missing values

    Examples
    --------
    >>> import numpy as np
    >>> from esda.missing import missing_data_summary
    >>> y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    >>> summary = missing_data_summary(y)
    >>> summary['pct_missing']
    20.0
    """
    y_array = np.asarray(y)
    mask = identify_missing_observations(y_array, raise_error=False)

    n_total = len(y_array)
    n_valid = np.sum(mask)
    n_missing = n_total - n_valid
    pct_missing = (n_missing / n_total) * 100.0

    summary = {
        "n_total": n_total,
        "n_missing": n_missing,
        "n_valid": n_valid,
        "pct_missing": pct_missing,
        "complete": bool(n_missing == 0),
    }

    return summary


__all__ = [
    "identify_missing_observations",
    "mask_missing_observations",
    "restore_missing_observations",
    "missing_data_summary",
]
