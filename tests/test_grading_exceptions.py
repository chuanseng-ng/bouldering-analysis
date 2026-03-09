"""Tests for src.grading.exceptions module.

Covers:
- src/grading/exceptions.py  — GradeEstimationError
"""

import pytest

from src.grading.exceptions import GradeEstimationError


class TestGradeEstimationError:
    """Tests for GradeEstimationError exception class."""

    def test_is_value_error_subclass(self) -> None:
        """GradeEstimationError must be a subclass of ValueError."""
        assert issubclass(GradeEstimationError, ValueError)

    def test_message_attribute_stored(self) -> None:
        """GradeEstimationError.message must hold the provided string."""
        exc = GradeEstimationError("test error")
        assert exc.message == "test error"

    def test_str_representation_matches_message(self) -> None:
        """str(exc) must equal the message string."""
        exc = GradeEstimationError("something went wrong")
        assert str(exc) == "something went wrong"

    def test_can_be_raised_and_caught_as_value_error(self) -> None:
        """GradeEstimationError must be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise GradeEstimationError("raised as ValueError")

    def test_can_be_raised_and_caught_as_grade_estimation_error(self) -> None:
        """GradeEstimationError must be catchable as GradeEstimationError."""
        with pytest.raises(GradeEstimationError):
            raise GradeEstimationError("raised directly")
