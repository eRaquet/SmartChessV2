"""Board Module."""

from .config import BoardConfig as BoardConfig
from .standard_board import Board as Board


def create_board(config: BoardConfig) -> Board:
    """

    Create a board from the provided config.

    Parameters
    ----------
    config : BoardConfig
        Config to create board from.

    Returns
    -------
    Board
        Created board.
    """
    if type(config) is BoardConfig:
        return Board(config)
    msg = 'Unsupported baord config type.'
    raise ValueError(msg)
