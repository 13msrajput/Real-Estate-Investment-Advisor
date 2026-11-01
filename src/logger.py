"""Shared logger factory so every module logs in the same format."""

# Standard logging library
import logging

# Used to write log records to the console
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent format."""
    # Fetch (or create) the logger with the requested name
    logger = logging.getLogger(name)
    # Only attach a handler once to avoid duplicated log lines
    if not logger.handlers:
        # Send log records to standard output
        handler = logging.StreamHandler(sys.stdout)
        # Timestamp, level, module name and message
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        # Attach the handler to the logger
        logger.addHandler(handler)
        # Default to INFO verbosity
        logger.setLevel(logging.INFO)
        # Stop records bubbling up to the root logger (prevents duplicates in notebooks)
        logger.propagate = False
    # Hand the configured logger back
    return logger
