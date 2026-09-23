# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/timing.py
# Title       : Wall-clock timing of code blocks and functions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Timer measures elapsed time with the monotonic high-resolution clock
# time.perf_counter, which is not affected by system clock adjustments.
# It works as a context manager, records named laps and can log the result;
# timed() wraps a function so that every call is logged. The clock can be
# replaced, which makes the class testable without sleeping.
#
# Usage
# -----
#   with Timer("tiling") as t:
#       ...
#   t.elapsed          seconds spent in the block
#
#   @timed()
#   def run(...): ...
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Preserve names of wrapped functions.
import functools

# Standard logging.
import logging

# Monotonic clock.
import time

# Types of callables and loosely structured values.
from typing import Any, Callable, TypeVar

# Library logger.
from unbihexium.utils.log import get_logger

# Return type of timed functions.
T = TypeVar("T")


# Elapsed-time measurement with laps.
class Timer:
    # Configure the timer; it starts on enter or start().
    def __init__(
        self,  # The timer.
        name: str = "block",  # Label used in log messages.
        logger: logging.Logger | None = None,  # Logger of the result; None logs nothing.
        level: int = logging.INFO,  # Level of the log message.
        clock: Callable[[], float] = time.perf_counter,  # Monotonic clock in seconds.
    ) -> None:  # The constructor returns nothing.
        # Label.
        self.name = name
        # Logger of the result.
        self.logger = logger
        # Log level.
        self.level = level
        # Clock function.
        self.clock = clock
        # Start time, None before start().
        self._start: float | None = None
        # Stop time, None while running.
        self._stop: float | None = None
        # Named laps (name, seconds since the previous lap).
        self.laps: list[tuple[str, float]] = []
        # Time of the previous lap.
        self._last: float | None = None

    # Start (or restart) the timer.
    def start(self) -> Timer:
        # Current time.
        now = self.clock()
        # Start and lap reference.
        self._start = self._last = now
        # Clear the stop time.
        self._stop = None
        # Clear earlier laps.
        self.laps = []
        # Allow chaining.
        return self

    # Record a lap and return its duration.
    def lap(self, name: str) -> float:
        # Laps need a running timer.
        if self._last is None or self._stop is not None:
            # Explain the problem.
            raise RuntimeError("the timer is not running")
        # Current time.
        now = self.clock()
        # Time since the previous lap.
        duration = now - self._last
        # Record the lap.
        self.laps.append((name, duration))
        # New reference.
        self._last = now
        # Return the lap duration.
        return duration

    # Stop the timer and return the elapsed time.
    def stop(self) -> float:
        # Stopping needs a started timer.
        if self._start is None:
            # Explain the problem.
            raise RuntimeError("the timer was never started")
        # Stop time.
        self._stop = self.clock()
        # Log the result when a logger is set.
        if self.logger is not None:
            # One message with the label and the time.
            self.logger.log(self.level, "%s took %.3f s", self.name, self.elapsed)
        # Elapsed seconds.
        return self.elapsed

    # Elapsed seconds, up to now while running.
    @property
    def elapsed(self) -> float:
        # Not started yet.
        if self._start is None:
            # No time has passed.
            return 0.0
        # End of the interval.
        end = self._stop if self._stop is not None else self.clock()
        # Duration.
        return end - self._start

    # Start on entering a with block.
    def __enter__(self) -> Timer:
        # Start the timer.
        return self.start()

    # Stop on leaving the block, also after an exception.
    def __exit__(self, *exc: object) -> None:
        # Stop the timer.
        self.stop()


# Decorator that logs the duration of every call.
def timed(
    logger: logging.Logger | None = None,  # Logger; default the library logger.
    level: int = logging.DEBUG,  # Level of the messages.
) -> Callable[[Callable[..., T]], Callable[..., T]]:  # The decorator.
    # Logger of the messages.
    log = logger or get_logger("timing")

    # Wrap one function.
    def decorate(func: Callable[..., T]) -> Callable[..., T]:
        # Keep the name and documentation of the function.
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Time the call.
            with Timer(func.__qualname__, log, level):
                # Run the function.
                return func(*args, **kwargs)

        # Return the wrapper.
        return wrapper

    # Return the decorator.
    return decorate


# =============================================================================
# End of module src/unbihexium/utils/timing.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
