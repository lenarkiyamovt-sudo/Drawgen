"""
stl_drawing — генератор ЕСКД-чертежей из STL-файлов.

Основной пайплайн запускается через main.py.
"""

from stl_drawing.logging_config import (
    setup_logging,
    get_logger,
    configure_default_logging,
    log_timing,
    timed,
    LogContext,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "configure_default_logging",
    "log_timing",
    "timed",
    "LogContext",
]
