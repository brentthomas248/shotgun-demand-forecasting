"""Project configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load project configuration from a YAML file.

    Args:
        path: Absolute or relative path to settings.yaml.
            If None, defaults to config/settings.yaml relative to the
            project root (two levels up from this file).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if path is None:
        config_path = Path(__file__).resolve().parent / "settings.yaml"
    else:
        config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config
