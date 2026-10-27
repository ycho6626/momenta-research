"""Portable numeric settings for the computational examples (Methodology)."""


import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read numeric settings from the source checkout."""
    path = path or REPO / "config/settings.toml"
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    settings["config_path"] = str(path.resolve())
    return settings
