"""Config for smartchess apps."""

from dataclasses import dataclass
from pathlib import Path

from smartchess.game import GameConfig
from smartchess.pipeline import LoggerConfig


@dataclass(slots=True, kw_only=True)
class PlayConfig:
    """Config for playing a game."""

    game: GameConfig
    """
    Config for game.
    """

    output: LoggerConfig | None = None
    """
    Logger configuration, or None if game is not to be logged.
    """


@dataclass(slots=True, kw_only=True)
class SeedConfig:
    """Config for seeding the models."""

    path: Path
    """
    Path to the directory to seed the models to.
    """
