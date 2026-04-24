"""
YAML configuration loader for CSCS-AL.

Configs are organized as:
    configs/datasets/{name}.yaml   — dataset paths and metadata
    configs/methods/{name}.yaml    — method hyperparameters
    configs/experiments/{name}.yaml — full experiment definition
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_experiment_config(
    experiment_yaml: str | Path,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a full experiment config, resolving dataset and method sub-configs.

    An experiment YAML may reference:
        dataset_config: configs/datasets/brats.yaml
        method_config:  configs/methods/cscs.yaml

    These are merged into the returned dict under keys 'dataset' and 'method'.

    Args:
        experiment_yaml: Path to the experiment YAML.
        config_root:     Root directory for resolving relative config paths.
                         Defaults to the experiment YAML's parent directory.
    """
    exp = load_yaml(experiment_yaml)
    config_root = Path(config_root) if config_root else Path(experiment_yaml).parent

    # Resolve dataset sub-config
    if "dataset_config" in exp:
        ds_path = _resolve(exp.pop("dataset_config"), config_root)
        exp["dataset"] = load_yaml(ds_path)

    # Resolve method sub-config
    if "method_config" in exp:
        m_path = _resolve(exp.pop("method_config"), config_root)
        exp["method"] = load_yaml(m_path)

    return exp


def merge_configs(*configs: dict) -> dict:
    """
    Deep-merge multiple config dicts (later values override earlier ones).

    Args:
        *configs: Dicts to merge in order (left to right).

    Returns:
        Merged dict.
    """
    result: dict = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
