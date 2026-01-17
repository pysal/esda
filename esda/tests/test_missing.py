"""Tests for missing value handling utilities."""

import libpysal
import numpy as np
import pytest
from numpy import testing as npt

from esda import missing


class TestIdentifyMissingObservations:
    """Test identify_missing_observations function."""

    def test_no_missing(self):
        """Test with data containing no missing values."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mask = missing.identify_missing_observations(y)
        npt.assert_array_equal(mask, np.array([True, True, True, True]))

    def test_with_missing(self):
        """Test with data containing missing values."""
        y = np.array([1.0, 2.0, np.nan, 4.0])
        mask = missing.identify_missing_observations(y)
        npt.assert_array_equal(mask, np.array([True, True, False, True]))

    def test_all_missing(self):
        """Test with all missing values."""
        y = np.array([np.nan, np.nan, np.nan])
        mask = missing.identify_missing_observations(y)
        npt.assert_array_equal(mask, np.array([False, False, False]))

    def test_raise_error_no_missing(self):
        """Test raise_error=True with no missing values."""
        y = np.array([1.0, 2.0, 3.0])
        result = missing.identify_missing_observations(y, raise_error=True)
        assert result is None

    def test_raise_error_with_missing(self):
        """Test raise_error=True with missing values."""
        y = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="missing value"):
            missing.identify_missing_observations(y, raise_error=True)


class TestMaskMissingObservations:
    """Test mask_missing_observations function."""

    def setup_method(self):
        """Set up test data."""
        self.y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        self.w = libpysal.weights.lat2W(5, 1)

    def test_basic_masking(self):
        """Test basic removal of missing observations."""
        y_clean, w_clean = missing.mask_missing_observations(self.y, self.w)
        expected_y = np.array([1.0, 2.0, 4.0, 5.0])
        npt.assert_array_equal(y_clean, expected_y)
        assert w_clean.n == 4

    def test_return_indices(self):
        """Test return of indices for non-missing observations."""
        y_clean, w_clean, idx = missing.mask_missing_observations(
            self.y, self.w, return_indices=True
        )
        npt.assert_array_equal(idx, np.array([0, 1, 3, 4]))
        assert len(y_clean) == len(idx)

    def test_no_missing_data(self):
        """Test with data containing no missing values."""
        y_complete = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_clean, w_clean = missing.mask_missing_observations(y_complete, self.w)
        npt.assert_array_equal(y_clean, y_complete)
        assert w_clean.n == 5

    def test_all_missing_raises_error(self):
        """Test that all-missing data raises error."""
        y_all_missing = np.full(5, np.nan)
        with pytest.raises(ValueError, match="All observations have missing"):
            missing.mask_missing_observations(y_all_missing, self.w)

    def test_empty_data_raises_error(self):
        """Test that empty data raises error."""
        y_empty = np.array([])
        with pytest.raises(ValueError, match="empty"):
            missing.mask_missing_observations(y_empty, self.w)


class TestRestoreMissingObservations:
    """Test restore_missing_observations function."""

    def test_basic_restoration(self):
        """Test basic restoration of missing values."""
        result = np.array([0.5, 0.3, 0.7])
        indices = np.array([0, 1, 3])
        original_length = 5

        restored = missing.restore_missing_observations(
            result, original_length, indices
        )
        expected = np.array([0.5, 0.3, np.nan, 0.7, np.nan])
        npt.assert_array_equal(
            restored[~np.isnan(restored)], expected[~np.isnan(expected)]
        )
        assert np.sum(np.isnan(restored)) == 2

    def test_no_missing_restore(self):
        """Test restoration when no observations are missing."""
        result = np.array([0.5, 0.3, 0.7, 0.2, 0.1])
        indices = np.array([0, 1, 2, 3, 4])
        original_length = 5

        restored = missing.restore_missing_observations(
            result, original_length, indices
        )
        npt.assert_array_equal(restored, result)

    def test_custom_fill_value(self):
        """Test restoration with custom fill value."""
        result = np.array([0.5, 0.3, 0.7])
        indices = np.array([0, 1, 3])
        original_length = 5

        restored = missing.restore_missing_observations(
            result, original_length, indices, fill_value=0.0
        )
        expected = np.array([0.5, 0.3, 0.0, 0.7, 0.0])
        npt.assert_array_equal(restored, expected)

    def test_undefined_fill_value(self):
        """Test restoration with 'Undefined' string fill value."""
        result = np.array([1, 2, 3], dtype=object)
        indices = np.array([0, 2, 4])
        original_length = 5

        restored = missing.restore_missing_observations(
            result, original_length, indices, fill_value="Undefined"
        )
        assert restored[1] == "Undefined"
        assert restored[3] == "Undefined"
        assert restored[0] == 1

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise error."""
        result = np.array([0.5, 0.3])
        indices = np.array([0, 1, 3])

        with pytest.raises(ValueError, match="Length of result_array"):
            missing.restore_missing_observations(result, 5, indices)


class TestMissingDataSummary:
    """Test missing_data_summary function."""

    def test_no_missing_summary(self):
        """Test summary with no missing values."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = missing.missing_data_summary(y)

        assert summary["n_total"] == 5
        assert summary["n_missing"] == 0
        assert summary["n_valid"] == 5
        assert summary["pct_missing"] == 0.0
        assert summary["complete"]

    def test_with_missing_summary(self):
        """Test summary with missing values."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        summary = missing.missing_data_summary(y)

        assert summary["n_total"] == 5
        assert summary["n_missing"] == 1
        assert summary["n_valid"] == 4
        assert summary["pct_missing"] == 20.0
        assert not summary["complete"]

    def test_all_missing_summary(self):
        """Test summary with all missing values."""
        y = np.full(5, np.nan)
        summary = missing.missing_data_summary(y)

        assert summary["n_total"] == 5
        assert summary["n_missing"] == 5
        assert summary["n_valid"] == 0
        assert summary["pct_missing"] == 100.0
        assert not summary["complete"]


class TestIssue215Integration:
    """Integration test for Issue #215: NaN handling in spatial statistics."""

    def test_lisa_with_missing_values(self):
        """
        Test that utilities enable LISA with missing values (Issue #215).

        This demonstrates the recommended workflow for handling missing data:
        1. Identify missing observations
        2. Mask and subset data/weights
        3. Perform spatial analysis
        4. Restore missing indicators to results
        """
        from esda import Moran_Local

        # Simulate grid with population (some cells unpopulated)
        y = np.array([10.0, 20.0, np.nan, 40.0, 50.0, np.nan, 70.0, 80.0, 90.0, 100.0])
        w = libpysal.weights.lat2W(5, 2)

        # Show the problem: without masking, analysis fails
        summary = missing.missing_data_summary(y)
        assert summary["n_missing"] == 2
        assert not summary["complete"]

        # Solution: mask missing observations
        y_clean, w_clean, indices = missing.mask_missing_observations(
            y, w, return_indices=True
        )

        # Perform analysis on clean data
        lisa = Moran_Local(y_clean, w_clean)

        # Restore results to original geometry
        Is = missing.restore_missing_observations(
            lisa.Is, len(y), indices, fill_value=np.nan
        )
        p_sim = missing.restore_missing_observations(
            lisa.p_sim, len(y), indices, fill_value=np.nan
        )

        # Verify results
        assert len(Is) == len(y)
        assert len(p_sim) == len(y)
        assert np.sum(np.isnan(Is)) == 2  # Two missing values
        assert np.sum(~np.isnan(Is)) == 8  # Eight valid results
