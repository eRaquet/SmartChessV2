"""Game Module."""

from .config import GameConfig as GameConfig
from .config import StandardGameConfig as StandardGameConfig
from .game_base import GameBase as GameBase


def create_game(config: GameConfig) -> GameBase:
    """

    Create a game object from the provided config.

    Parameters
    ----------
    config : GameConfig
        config object from the game

    Returns
    -------
    GameBase
        newly created game
    """
    if type(config) is StandardGameConfig:
        from .standard_game import StandardGame

        return StandardGame(config)
    msg = 'Unsupported game type.'
    raise ValueError(msg)
