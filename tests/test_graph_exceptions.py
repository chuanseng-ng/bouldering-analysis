"""Tests for src.graph.exceptions module.

Covers:
- src/graph/exceptions.py — RouteGraphError
"""

import pytest

from src.graph.exceptions import RouteGraphError


# ---------------------------------------------------------------------------
# TestRouteGraphError
# ---------------------------------------------------------------------------


class TestRouteGraphError:
    """Tests for the RouteGraphError exception class."""

    def test_is_subclass_of_value_error(self) -> None:
        """RouteGraphError must inherit from ValueError."""
        assert issubclass(RouteGraphError, ValueError)

    def test_message_attribute_equals_constructor_argument(self) -> None:
        """RouteGraphError.message must equal the constructor string."""
        err = RouteGraphError("something went wrong")
        assert err.message == "something went wrong"

    def test_str_contains_message(self) -> None:
        """str(RouteGraphError) must include the message text."""
        err = RouteGraphError("holds must not be empty")
        assert "holds must not be empty" in str(err)

    def test_can_be_caught_as_value_error(self) -> None:
        """RouteGraphError can be caught with except ValueError."""
        with pytest.raises(ValueError):
            raise RouteGraphError("test")
