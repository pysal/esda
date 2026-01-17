# PR: Issue #215 — Missing Data Utilities and Tests

## Summary
This PR introduces utilities for handling missing (NaN) values in esda analyses and adds focused tests verifying the workflow. It also updates test helpers to use the current libpysal lattice API.

## Changes
- `esda/missing.py`
  - Add/confirm utilities: `identify_missing_observations()`, `mask_missing_observations()`, `restore_missing_observations()`, `missing_data_summary()`.
  - Ensure `missing_data_summary()` returns a native Python boolean for `complete`.
  - Sort imports per ruff/black conventions.
- `esda/tests/test_missing.py`
  - Replace deprecated `libpysal.weights.util.lattice_weights(shape=...)` with `libpysal.weights.lat2W(nrows, ncols)`.
  - Fix assertions to avoid `is True/False` with NumPy boolean types.
  - Wrap long line in array equality assertion.

## Rationale and References
Handling missing data explicitly improves transparency and reproducibility in spatial statistics.
- Anselin, L. (1995). Local indicators of spatial association (LISA). Geographical Analysis, 27(2), 93–115.
- Rey, S. J., & Anselin, L. (2010). PySAL: A Python library of spatial analytical methods. Journal of Statistical Software, 42(2), 1–24.

## Tests
Targeted tests for missing-data utilities:
- All tests in `esda/tests/test_missing.py` pass locally (19/19).

## Targeted CI Checks (modified files only)
- `pytest esda/tests/test_missing.py -v`
- `ruff check esda/missing.py esda/tests/test_missing.py`
- `black --check esda/missing.py esda/tests/test_missing.py`
- `mypy esda/missing.py --ignore-missing-imports`

Note: Full suite failures observed locally are unrelated (optional visualization deps and numeric underflow in mixture smoothing). The targeted checks for the modified files are clean.

## Compatibility
- Uses `libpysal.weights.lat2W(nrows, ncols)` which is the current lattice helper in libpysal.
- No public API changes in esda beyond the new utilities.

## Checklist
- [x] Focused tests added and passing for changed functionality
- [x] Imports and formatting align with ruff/black
- [x] mypy clean on `esda/missing.py` (ignoring external imports as configured)
- [x] Minimal, surgical edits

## Linked Issue
Addresses: #215
