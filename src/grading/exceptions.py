"""Grade estimation module exception hierarchy.

All grade estimation exceptions derive from ``GradeEstimationError``,
enabling callers to catch any estimation error with a single except clause.
"""


class GradeEstimationError(ValueError):
    """Raised when grade estimation from a RouteFeatures fails.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise GradeEstimationError("Feature vector is malformed")
    """

    def __init__(self, message: str) -> None:
        """Initialize GradeEstimationError with a message.

        Args:
            message: Description of the estimation error that occurred.
        """
        self.message = message
        super().__init__(self.message)
