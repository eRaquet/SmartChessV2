"""Config for board class."""

from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class BoardConfig:
    """Base config for board."""


DEFAULT_BOARD_CONFIG = BoardConfig()
