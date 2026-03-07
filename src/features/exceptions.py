"""Feature extraction module exception hierarchy.

All feature extraction exceptions derive from ``FeatureExtractionError``,
enabling callers to catch any extraction error with a single except clause.
"""


class FeatureExtractionError(ValueError):
    """Raised when feature extraction from a RouteGraph fails.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise FeatureExtractionError("Graph has no start nodes")
    """

    def __init__(self, message: str) -> None:
        """Initialize FeatureExtractionError with a message.

        Args:
            message: Description of the extraction error that occurred.
        """
        self.message = message
        super().__init__(self.message)
