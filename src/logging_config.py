"""Structured JSON logging configuration.

This module configures structured logging for the application,
outputting logs in JSON format suitable for log aggregation systems.
"""

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields.

    Adds standard fields to every log record including timestamp,
    level, logger name, and any extra context.
    """

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add custom fields to the log record.

        Args:
            log_record: Dictionary to populate with log fields.
            record: The original LogRecord.
            message_dict: Message dictionary from the record.
        """
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        # Add location info for debugging
        if record.levelno >= logging.WARNING:
            log_record["location"] = f"{record.pathname}:{record.lineno}"
            log_record["function"] = record.funcName


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured JSON logging for the application.

    Sets up the root logger with JSON formatting for production
    environments. Includes timestamp, level, and structured fields
    for log aggregation systems like Elasticsearch or CloudWatch.

    Args:
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING,
            ERROR, CRITICAL).
        json_output: If True, output JSON format. If False, use standard
            format (useful for local development).

    Example:
        >>> configure_logging("DEBUG", json_output=False)
        >>> logging.info("Application started")
    """
    # Get the numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_output:
        # JSON format for production
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Convenience function to get a properly configured logger.

    Args:
        name: Name for the logger, typically __name__.

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing request", extra={"request_id": "abc123"})
    """
    return logging.getLogger(name)
