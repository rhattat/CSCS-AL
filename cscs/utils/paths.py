"""
Path resolution utilities.

All paths in CSCS-AL are resolved relative to a configurable project root,
never hardcoded as absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Return the project root directory.

    Resolution order:
        1. CSCS_ROOT environment variable
        2. Parent of the cscs/ package (i.e., the repo root)
    """
    env = os.environ.get("CSCS_ROOT")
    if env:
        return Path(env).resolve()
    # Walk up from this file: cscs/utils/paths.py → cscs/ → project root
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """
    Resolve a path, relative to base (or project root if base is None).

    Absolute paths are returned unchanged.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base else get_project_root()
    return (root / p).resolve()
