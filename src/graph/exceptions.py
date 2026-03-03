"""Route graph module exception hierarchy.

All graph module exceptions derive from ``RouteGraphError``, enabling
callers to catch any graph construction error with a single except clause.
"""


class RouteGraphError(ValueError):
    """Raised when route graph construction fails.

    Subclasses represent errors from specific graph building stages.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise RouteGraphError("holds must not be empty")
    """

    def __init__(self, message: str) -> None:
        """Initialize RouteGraphError with a message.

        Args:
            message: Description of the graph error that occurred.
        """
        self.message = message
        super().__init__(self.message)
