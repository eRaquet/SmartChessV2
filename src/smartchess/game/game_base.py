"""Base class for game."""

from abc import ABC, abstractmethod

from smartchess.types import GameLog


class GameBase(ABC):
    """Abstract class for game orchestrators."""

    @abstractmethod
    def play_game(self) -> GameLog | None:
        """

        Play through the game.

        Returns
        -------
        GameLog | None
            log of game, or None if not recorded
        """
