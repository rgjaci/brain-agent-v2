"""Brain Agent v2 — Retry utilities.

Provides a decorator and context manager for retrying operations with
exponential backoff and jitter.  Used for external API calls (Gemini,
Ollama, OpenRouter) that may fail transiently.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay in seconds (caps exponential growth).
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to prevent thundering herd.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated function that retries on failure.

    Example::

        @retry(max_attempts=3, base_delay=1.0, exceptions=(ConnectionError,))
        def call_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise

                    # Calculate delay with optional jitter
                    if jitter:
                        delay = delay * exponential_base + random.uniform(0, delay)
                    else:
                        delay = delay * exponential_base

                    delay = min(delay, max_delay)

                    logger.warning(
                        "%s attempt %d/%d failed (%s), retrying in %.1fs...",
                        func.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            raise last_exception  # type: ignore[misc]

        return wrapper
    return decorator


class RetryContext:
    """Context manager for retrying a block of code.

    Example::

        with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                result = risky_operation()
                if result:
                    break
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.attempt = 0
        self._last_exception: Exception | None = None

    def __enter__(self) -> RetryContext:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        return False

    def __iter__(self) -> RetryContext:
        return self

    def __next__(self) -> int:
        if self.attempt >= self.max_attempts:
            if self._last_exception:
                raise self._last_exception
            raise StopIteration

        if self.attempt > 0:
            delay = min(
                self.base_delay * (2 ** (self.attempt - 1)) + random.uniform(0, self.base_delay),
                self.max_delay,
            )
            logger.warning(
                "Retry attempt %d/%d, waiting %.1fs...",
                self.attempt + 1, self.max_attempts, delay,
            )
            time.sleep(delay)

        self.attempt += 1
        return self.attempt

    def record_failure(self, exc: Exception) -> None:
        """Record that the current attempt failed."""
        self._last_exception = exc
