"""Inference pipeline exception hierarchy.

All inference module exceptions derive from InferencePipelineError,
enabling callers to catch any pipeline error with a single except clause.
"""


class InferencePipelineError(Exception):
    """Base class for all inference pipeline errors.

    Subclasses represent errors from specific pipeline stages:
    detection, crop extraction, and classification.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> try:
        ...     detect_holds(img, weights)
        ... except InferencePipelineError as exc:
        ...     print(exc.message)
    """

    def __init__(self, message: str) -> None:
        """Initialize InferencePipelineError with a message.

        Args:
            message: Description of the inference error that occurred.
        """
        self.message = message
        super().__init__(self.message)
