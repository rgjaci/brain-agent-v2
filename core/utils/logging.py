"""Brain Agent v2 — Structured logging utilities.

Provides a JSON log formatter for production use and helper functions
for timing operations.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production use.

    Outputs each log record as a single JSON line, suitable for ingestion
    by log aggregation systems (ELK, Datadog, etc.).

    Example output::

        {"timestamp": "2026-04-04T12:00:00.000Z", "level": "INFO",
         "logger": "core.agent", "message": "Turn complete",
         "duration_ms": 1234, "session_id": "abc-123"}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the log record
        for key in ("duration_ms", "session_id", "turn_id", "tool_name",
                     "retrieval_count", "memory_count", "event_type"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def setup_structured_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure the root logger to use JSON formatting.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path to write logs to. If None, logs to stderr.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = JSONFormatter()

    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)


@contextmanager
def timed_operation(
    operation_name: str,
    logger_name: str = "core.timing",
    level: int = logging.DEBUG,
) -> Generator[None, None, None]:
    """Context manager that logs the duration of an operation.

    Args:
        operation_name: Human-readable name of the operation.
        logger_name: Logger to use for the timing log.
        level: Log level for the timing message.

    Example::

        with timed_operation("retrieval"):
            results = await reader.retrieve(query)
    """
    op_logger = logging.getLogger(logger_name)
    start = time.monotonic()
    try:
        yield
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        op_logger.log(level, "%s completed in %.1fms", operation_name, duration_ms)
