"""Script for playing a chess game."""

from smartchess.game import create_game
from smartchess.pipeline import create_logger

from .config import PlayConfig


def play(config: PlayConfig) -> None:
    """

    Play a chess game based on the provided config.

    Parameters
    ----------
    config : PlayConfig
    """
    game = create_game(config.game)

    log = game.play_game()

    if log is not None and config.output is not None:
        logger = create_logger(config.output)
        logger.log_game(log)
