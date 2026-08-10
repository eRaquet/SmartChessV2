"""Config for UI module."""

from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class GUIConfig:
    """Config for GUI object."""
