"""Config for smartchess apps."""

from dataclasses import dataclass

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
