"""Logger base class."""

from abc import abstractmethod
from dataclasses import dataclass

from smartchess.types import GameLog


@dataclass(slots=True, kw_only=True)
class LoggerConfig:
    """Base config for logger."""


class LoggerBase:
    """Base class for logger."""

    @abstractmethod
    def log_game(self, log: GameLog) -> None:
        """

        Log the provided game log to some source.

        Parameters
        ----------
        log : GameLog
        """
