# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/log.py
# Title       : Logging configuration of the library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Every module logs through a child of the "unbihexium" logger, so that an
# application can control the whole library with one logger. Following the
# recommendation of the Python logging HOWTO for libraries, the library adds
# no handler by itself; applications and the command line call
# configure_logging() once:
#
#   get_logger          logger in the "unbihexium" hierarchy
#   configure_logging   add (or replace) one stream handler with a UTC ISO
#                       8601 time stamp; the level defaults to the
#                       UNBIHEXIUM_LOG_LEVEL environment variable
#   parse_level         level name or number to a logging level
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Standard logging.
import logging

# Environment variables.
import os

# UTC time stamps.
import time

# Output streams.
from typing import TextIO

# Name of the root logger of the library.
ROOT_LOGGER = "unbihexium"

# Environment variable with the default level.
LEVEL_ENV = "UNBIHEXIUM_LOG_LEVEL"

# Default record format.
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# Attribute that marks the handler installed by configure_logging.
_HANDLER_MARK = "_unbihexium_handler"


# Logger in the library hierarchy.
def get_logger(name: str | None = None) -> logging.Logger:
    # The root logger of the library.
    if not name or name == ROOT_LOGGER:
        # Return it.
        return logging.getLogger(ROOT_LOGGER)
    # Module names already carry the prefix.
    if name.startswith(ROOT_LOGGER + "."):
        # Use the name unchanged.
        return logging.getLogger(name)
    # Other names become children of the library logger.
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


# Convert a level name ("info") or number (20) to a logging level.
def parse_level(level: str | int) -> int:
    # Numbers are used as they are.
    if isinstance(level, int):
        # Return the number.
        return level
    # Numeric strings are numbers.
    text = str(level).strip().upper()
    # Accept "10", "20", ...
    if text.isdigit():
        # Return the number.
        return int(text)
    # Known level names.
    value = logging.getLevelName(text)
    # getLevelName returns a string for unknown names.
    if not isinstance(value, int):
        # Explain the accepted values.
        raise ValueError(f"unknown log level {level!r}; use DEBUG, INFO, WARNING, ERROR")
    # Return the level.
    return value


# Formatter with UTC ISO 8601 time stamps.
class UTCFormatter(logging.Formatter):
    # Time stamps in UTC.
    converter = time.gmtime

    # ISO 8601 time with milliseconds and a Z suffix.
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # Date and time to the second.
        stamp = time.strftime(datefmt or "%Y-%m-%dT%H:%M:%S", self.converter(record.created))
        # Append the milliseconds and the zone.
        return f"{stamp}.{int(record.msecs):03d}Z"


# Install one stream handler on the library logger and set its level.
def configure_logging(
    level: str | int | None = None,  # Level; default from UNBIHEXIUM_LOG_LEVEL or WARNING.
    stream: TextIO | None = None,  # Output stream; default standard error.
    fmt: str = DEFAULT_FORMAT,  # Record format.
) -> logging.Logger:  # The configured library logger.
    # Library logger.
    logger = logging.getLogger(ROOT_LOGGER)
    # Level from the argument or the environment.
    chosen = level if level is not None else os.environ.get(LEVEL_ENV, "WARNING")
    # Apply the level.
    logger.setLevel(parse_level(chosen))
    # Remove the handler of an earlier call so that calls do not accumulate.
    for handler in list(logger.handlers):
        # Only handlers installed here are removed.
        if getattr(handler, _HANDLER_MARK, False):
            # Detach it.
            logger.removeHandler(handler)
    # New stream handler.
    handler = logging.StreamHandler(stream)
    # Mark it as ours.
    setattr(handler, _HANDLER_MARK, True)
    # Format with UTC time stamps.
    handler.setFormatter(UTCFormatter(fmt))
    # Attach it.
    logger.addHandler(handler)
    # Return the logger.
    return logger


# =============================================================================
# End of module src/unbihexium/utils/log.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
