"""Tests for logging configuration module."""

import json
import logging
from io import StringIO


from src.logging_config import (
    CustomJsonFormatter,
    configure_logging,
    get_logger,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_sets_log_level(self) -> None:
        """Log level should be set on root logger."""
        configure_logging("DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_info_level(self) -> None:
        """INFO level should be set correctly."""
        configure_logging("INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_logging_warning_level(self) -> None:
        """WARNING level should be set correctly."""
        configure_logging("WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_logging_clears_existing_handlers(self) -> None:
        """Existing handlers should be cleared."""
        root_logger = logging.getLogger()
        # Add a dummy handler
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)
        # Verify we have more than one handler before configure
        assert len(root_logger.handlers) > 0

        configure_logging("INFO")

        # Should have exactly one handler after configure
        assert len(root_logger.handlers) == 1

    def test_configure_logging_json_output_true(self) -> None:
        """JSON output should use CustomJsonFormatter."""
        configure_logging("INFO", json_output=True)
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, CustomJsonFormatter)

    def test_configure_logging_json_output_false(self) -> None:
        """Non-JSON output should use standard Formatter."""
        configure_logging("INFO", json_output=False)
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert not isinstance(handler.formatter, CustomJsonFormatter)


class TestCustomJsonFormatter:
    """Tests for CustomJsonFormatter class."""

    def test_json_formatter_outputs_valid_json(self) -> None:
        """Formatter should output valid JSON."""
        # Create a logger with our formatter
        logger = logging.getLogger("test_json")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CustomJsonFormatter("%(timestamp)s %(level)s %(message)s"))
        logger.addHandler(handler)

        logger.info("Test message")
        handler.flush()

        output = stream.getvalue()
        # Should be valid JSON
        data = json.loads(output)
        assert "message" in data

    def test_json_formatter_includes_level(self) -> None:
        """JSON output should include log level."""
        logger = logging.getLogger("test_level")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CustomJsonFormatter("%(timestamp)s %(level)s %(message)s"))
        logger.addHandler(handler)

        logger.info("Test message")
        handler.flush()

        data = json.loads(stream.getvalue())
        assert data["level"] == "INFO"

    def test_json_formatter_includes_logger_name(self) -> None:
        """JSON output should include logger name."""
        logger = logging.getLogger("my_test_logger")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CustomJsonFormatter("%(timestamp)s %(level)s %(message)s"))
        logger.addHandler(handler)

        logger.info("Test message")
        handler.flush()

        data = json.loads(stream.getvalue())
        assert data["logger"] == "my_test_logger"

    def test_json_formatter_includes_timestamp(self) -> None:
        """JSON output should include timestamp."""
        logger = logging.getLogger("test_timestamp")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CustomJsonFormatter("%(timestamp)s %(level)s %(message)s"))
        logger.addHandler(handler)

        logger.info("Test message")
        handler.flush()

        data = json.loads(stream.getvalue())
        assert "timestamp" in data

    def test_json_formatter_includes_location_for_warnings(self) -> None:
        """JSON output should include location for WARNING and above."""
        logger = logging.getLogger("test_location")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CustomJsonFormatter("%(timestamp)s %(level)s %(message)s"))
        logger.addHandler(handler)

        logger.warning("Warning message")
        handler.flush()

        data = json.loads(stream.getvalue())
        assert "location" in data
        assert "function" in data


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger should return a Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self) -> None:
        """Logger should have the specified name."""
        logger = get_logger("my.module.name")
        assert logger.name == "my.module.name"

    def test_get_logger_same_name_returns_same_logger(self) -> None:
        """Same name should return the same logger instance."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2
