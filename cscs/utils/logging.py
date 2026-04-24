"""Logging setup for CSCS-AL scripts."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "cscs", level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
