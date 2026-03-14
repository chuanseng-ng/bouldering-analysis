"""Explanation engine module exception hierarchy.

All explanation exceptions derive from ``ExplanationError``,
enabling callers to catch any explanation error with a single except clause.
"""


class ExplanationError(ValueError):
    """Raised when explanation generation from a RouteFeatures fails.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise ExplanationError("Feature vector is malformed")
    """

    def __init__(self, message: str) -> None:
        """Initialize ExplanationError with a message.

        Args:
            message: Description of the explanation error that occurred.
        """
        self.message = message
        super().__init__(self.message)
