"""
Unit tests for stl_drawing.logging_config module.

Tests:
- JSON formatter output
- Console formatter output
- Logging setup
- Timing utilities
- Context management
"""

import json
import logging
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from stl_drawing.logging_config import (
    JSONFormatter,
    ConsoleFormatter,
    setup_logging,
    get_logger,
    log_timing,
    timed,
    LogContext,
    configure_default_logging,
)


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_basic_format(self):
        """Test basic JSON log format."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_extra_fields(self):
        """Test that extra fields are included in JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message with extras",
            args=(),
            exc_info=None,
        )
        record.file_path = "/path/to/file.stl"
        record.vertices_count = 1000

        result = formatter.format(record)
        data = json.loads(result)

        assert data["file_path"] == "/path/to/file.stl"
        assert data["vertices_count"] == 1000

    def test_location_for_warning(self):
        """Test that location info is included for warnings."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="myfile.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "location" in data
        assert data["location"]["line"] == 42

    def test_exception_format(self):
        """Test that exceptions are formatted."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_unicode_message(self):
        """Test Unicode characters in messages."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Русский текст и символы: ⌀, ±, °",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "Русский" in data["message"]
        assert "⌀" in data["message"]


class TestConsoleFormatter:
    """Tests for ConsoleFormatter class."""

    def test_basic_format(self):
        """Test basic console format."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="stl_drawing.io.loader",
            level=logging.INFO,
            pathname="loader.py",
            lineno=10,
            msg="Loading file",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "io.loader" in result  # Prefix stripped
        assert "Loading file" in result

    def test_extra_fields_shown(self):
        """Test that extra fields are shown inline."""
        formatter = ConsoleFormatter(use_colors=False, show_extra=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.size_kb = 1024.567

        result = formatter.format(record)

        assert "size_kb=" in result

    def test_colors_disabled(self):
        """Test that colors are not present when disabled."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # No ANSI escape codes
        assert "\033[" not in result


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Test that setup returns a logger."""
        logger = setup_logging(level=logging.DEBUG, console=False)
        assert isinstance(logger, logging.Logger)

    def test_console_handler_added(self):
        """Test that console handler is added."""
        logger = setup_logging(console=True)

        # Check for StreamHandler
        stream_handlers = [h for h in logger.handlers
                          if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0

    def test_json_file_handler(self):
        """Test JSON file handler creation."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            logger = setup_logging(json_file=json_path, console=False)
            logger.info("Test message", extra={"key": "value"})

            # Force flush and close handlers
            for handler in logger.handlers:
                handler.flush()
                handler.close()

            # Read and verify JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert content.strip()  # Not empty
            data = json.loads(content.strip())
            assert data["message"] == "Test message"
            assert data["key"] == "value"
        finally:
            # Clear handlers to release file
            logger.handlers.clear()
            Path(json_path).unlink(missing_ok=True)

    def test_level_setting(self):
        """Test that log level is correctly set."""
        logger = setup_logging(level=logging.WARNING, console=False)
        assert logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Test that get_logger returns a logger."""
        logger = get_logger("my.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my.module"

    def test_same_logger_returned(self):
        """Test that same logger is returned for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestLogTiming:
    """Tests for log_timing context manager."""

    def test_logs_start_and_complete(self):
        """Test that start and complete messages are logged."""
        # Setup logger with string handler
        logger = logging.getLogger("timing_test")
        logger.setLevel(logging.DEBUG)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        with log_timing(logger, "test operation"):
            pass

        output = stream.getvalue()
        assert "Starting" in output
        assert "Completed" in output
        assert "test operation" in output

    def test_timing_info_updated(self):
        """Test that timing_info dict is populated."""
        logger = logging.getLogger("timing_test2")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.NullHandler())

        with log_timing(logger, "operation") as timing_info:
            pass

        assert "elapsed_seconds" in timing_info
        assert timing_info["elapsed_seconds"] >= 0

    def test_error_logged_on_exception(self):
        """Test that errors are logged when exception occurs."""
        logger = logging.getLogger("timing_test3")
        logger.setLevel(logging.DEBUG)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)

        with pytest.raises(ValueError):
            with log_timing(logger, "failing operation"):
                raise ValueError("Test error")

        output = stream.getvalue()
        assert "ERROR" in output
        assert "Failed" in output


class TestTimedDecorator:
    """Tests for timed decorator."""

    def test_function_executed(self):
        """Test that decorated function executes correctly."""
        logger = logging.getLogger("timed_test")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.NullHandler())

        @timed(logger=logger)
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @timed()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestLogContext:
    """Tests for LogContext class."""

    def test_fields_added_to_records(self):
        """Test that context fields are added to log records."""
        # Setup test logger - use parent logger where filter is attached
        setup_logging(console=False)
        logger = get_logger("stl_drawing")  # Use parent logger directly

        # Create handler to capture records
        captured_records = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)

        capture = CaptureHandler()
        logger.addHandler(capture)

        try:
            # Log with context
            with LogContext(request_id="abc123", session="sess1"):
                logger.info("Test message")

            assert len(captured_records) > 0
            record = captured_records[0]
            assert hasattr(record, 'request_id')
            assert record.request_id == "abc123"
        finally:
            logger.removeHandler(capture)

    def test_context_current(self):
        """Test LogContext.current() returns active context."""
        assert LogContext.current() is None

        ctx = LogContext(test_field="value")
        with ctx:
            assert LogContext.current() is ctx
            assert LogContext.current().fields["test_field"] == "value"

        assert LogContext.current() is None


class TestConfigureDefaultLogging:
    """Tests for configure_default_logging function."""

    def test_info_level_default(self):
        """Test default level is INFO."""
        logger = configure_default_logging(verbose=False)
        assert logger.level == logging.INFO

    def test_debug_level_verbose(self):
        """Test verbose mode sets DEBUG level."""
        logger = configure_default_logging(verbose=True)
        assert logger.level == logging.DEBUG
