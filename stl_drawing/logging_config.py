"""
Structured logging configuration for stl_drawing package.

Provides:
- JSON formatter for machine-readable log output
- Console formatter for human-readable output
- Timing decorator for performance measurement
- Centralized logging setup

Usage:
    from stl_drawing.logging_config import setup_logging, get_logger

    # Setup at application start
    setup_logging(level=logging.INFO, json_file="app.log.json")

    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Processing STL file", extra={"file": "model.stl", "size_kb": 1024})
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs each log record as a single JSON line with standardized fields.
    Extra fields passed via `extra={}` are included in the output.

    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", ...}
    """

    def __init__(self, include_extra: bool = True):
        """Initialize JSON formatter.

        Args:
            include_extra: Include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra
        # Standard logging fields to exclude from extra
        self._reserved_keys = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime'
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON string (single line)
        """
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info for debug/error
        if record.levelno >= logging.WARNING or record.levelno <= logging.DEBUG:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self._reserved_keys:
                    # Try to serialize, fall back to str
                    try:
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with optional colors.

    Format: [TIME] LEVEL logger: message [extra_key=value ...]
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }

    def __init__(self, use_colors: bool = True, show_extra: bool = True):
        """Initialize console formatter.

        Args:
            use_colors: Use ANSI colors (disable for file output)
            show_extra: Show extra fields inline
        """
        super().__init__()
        self.use_colors = use_colors
        self.show_extra = show_extra
        self._reserved_keys = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime'
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output.

        Args:
            record: Log record to format

        Returns:
            Formatted string
        """
        # Time
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Level with optional color
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level_str = f"{self.COLORS[level]}{level:8}{self.COLORS['RESET']}"
        else:
            level_str = f"{level:8}"

        # Logger name (shortened)
        logger_name = record.name
        if logger_name.startswith("stl_drawing."):
            logger_name = logger_name[12:]  # Remove prefix

        # Base message
        message = record.getMessage()

        # Extra fields
        extra_str = ""
        if self.show_extra:
            extras = []
            for key, value in record.__dict__.items():
                if key not in self._reserved_keys:
                    # Format value compactly
                    if isinstance(value, float):
                        extras.append(f"{key}={value:.3g}")
                    elif isinstance(value, (list, tuple)) and len(value) > 3:
                        extras.append(f"{key}=[...{len(value)} items]")
                    else:
                        extras.append(f"{key}={value}")
            if extras:
                extra_str = " [" + ", ".join(extras) + "]"

        # Build final message
        result = f"[{time_str}] {level_str} {logger_name}: {message}{extra_str}"

        # Add exception if present
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


def setup_logging(
    level: int = logging.INFO,
    json_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    use_colors: bool = True,
    root_logger: bool = False,
) -> logging.Logger:
    """Configure logging for stl_drawing package.

    Sets up handlers for console (human-readable) and optionally
    JSON file output (machine-readable).

    Args:
        level: Minimum log level (default INFO)
        json_file: Optional path for JSON log file
        console: Enable console output (default True)
        use_colors: Use ANSI colors in console (default True)
        root_logger: Configure root logger instead of stl_drawing

    Returns:
        Configured logger instance

    Example:
        setup_logging(level=logging.DEBUG, json_file="stl_drawing.log.json")
    """
    # Get target logger
    logger_name = "" if root_logger else "stl_drawing"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(ConsoleFormatter(use_colors=use_colors))
        logger.addHandler(console_handler)

    # JSON file handler
    if json_file:
        json_path = Path(json_file)
        json_handler = logging.FileHandler(json_path, encoding='utf-8')
        json_handler.setLevel(level)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    # Prevent propagation to root if not root
    if not root_logger:
        logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Loading STL", extra={"path": stl_path})
    """
    return logging.getLogger(name)


@contextmanager
def log_timing(
    logger: logging.Logger,
    operation: str,
    level: int = logging.DEBUG,
    **extra_fields: Any
):
    """Context manager to log operation timing.

    Args:
        logger: Logger instance
        operation: Operation description
        level: Log level (default DEBUG)
        **extra_fields: Additional fields to include in log

    Example:
        with log_timing(logger, "Loading STL file", path=stl_path):
            vertices, faces = load_stl(stl_path)

    Yields:
        dict that can be updated with additional timing info
    """
    timing_info: Dict[str, Any] = {}
    start_time = time.perf_counter()

    logger.log(level, f"Starting: {operation}", extra={
        "event": "start",
        "operation": operation,
        **extra_fields
    })

    try:
        yield timing_info
        elapsed = time.perf_counter() - start_time
        timing_info['elapsed_seconds'] = elapsed

        logger.log(level, f"Completed: {operation} ({elapsed:.3f}s)", extra={
            "event": "complete",
            "operation": operation,
            "elapsed_seconds": elapsed,
            **extra_fields,
            **timing_info
        })
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Failed: {operation} ({elapsed:.3f}s) - {e}", extra={
            "event": "error",
            "operation": operation,
            "elapsed_seconds": elapsed,
            "error": str(e),
            **extra_fields
        })
        raise


def timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    operation: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        logger: Logger instance (uses function's module logger if None)
        level: Log level (default DEBUG)
        operation: Operation name (uses function name if None)

    Returns:
        Decorated function

    Example:
        @timed(level=logging.INFO)
        def process_mesh(vertices, faces):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger, operation

            # Get logger from function's module if not provided
            func_logger = logger or logging.getLogger(func.__module__)
            op_name = operation or func.__name__

            with log_timing(func_logger, op_name, level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


class LogContext:
    """Context holder for adding common fields to log records.

    Useful for adding request ID, session info, or other context
    that should appear in all log messages within a scope.

    Example:
        ctx = LogContext(request_id="abc123", user="john")
        with ctx:
            logger.info("Processing request")  # includes request_id, user
    """

    _current: Optional['LogContext'] = None

    def __init__(self, **fields: Any):
        """Initialize context with fields.

        Args:
            **fields: Fields to add to all log records
        """
        self.fields = fields
        self._previous: Optional['LogContext'] = None
        self._filter: Optional[logging.Filter] = None

    def __enter__(self) -> 'LogContext':
        """Enter context and install filter."""
        self._previous = LogContext._current
        LogContext._current = self

        # Create filter that adds our fields
        class ContextFilter(logging.Filter):
            def __init__(self, context: 'LogContext'):
                super().__init__()
                self.context = context

            def filter(self, record: logging.LogRecord) -> bool:
                for key, value in self.context.fields.items():
                    setattr(record, key, value)
                return True

        self._filter = ContextFilter(self)

        # Add filter to stl_drawing logger
        stl_logger = logging.getLogger("stl_drawing")
        stl_logger.addFilter(self._filter)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and remove filter."""
        if self._filter:
            stl_logger = logging.getLogger("stl_drawing")
            stl_logger.removeFilter(self._filter)

        LogContext._current = self._previous

    @classmethod
    def current(cls) -> Optional['LogContext']:
        """Get current context."""
        return cls._current


# Convenience function for quick setup
def configure_default_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with sensible defaults.

    Args:
        verbose: Enable DEBUG level if True, INFO otherwise

    Returns:
        Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    return setup_logging(level=level, console=True, use_colors=True)
