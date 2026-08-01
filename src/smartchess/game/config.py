"""Config for game class."""

from dataclasses import dataclass

from smartchess.agent import AgentBase, AgentConfig
from smartchess.board import Board, BoardConfig
from smartchess.types import BoardRenderer


@dataclass(slots=True, kw_only=True)
class GameConfig:
    """Base config for game."""

    white: AgentBase | AgentConfig
    """
    White agent, or config for white agent.
    """

    black: AgentBase | AgentConfig
    """
    Black agent, or config for black agent.
    """

    board: Board | BoardConfig
    """
    Board object, or config to create a board object.
    """

    renderer: BoardRenderer | None
    """
    Renderer for board, skip render if unprovided.
    """

    collect: bool
    """
    Whether or not to collect the game into a game log.
    """


@dataclass(slots=True, kw_only=True)
class StandardGameConfig(GameConfig):
    """Config for standard game."""
