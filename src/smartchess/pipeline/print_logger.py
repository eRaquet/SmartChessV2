"""Logger for printing relevant info to the terminal at the conclusion of a game."""

from dataclasses import dataclass
from typing import override

from smartchess.types import BoardStatus, GameLog, TerminationType
from smartchess.util import format_time

from .logger_base import LoggerBase, LoggerConfig


@dataclass(slots=True, kw_only=True)
class PrintLoggerConfig(LoggerConfig):
    """Config for printing logger."""

    winner: bool = True
    """
    Display the winner of the game.
    """

    termination_cause: bool = True
    """
    Display the cause of the game termination.
    """

    total_time: bool = True
    """
    Display the total time the game took.
    """

    time_per_move: bool = True
    """
    Display the time per move.
    """


class PrintLogger(LoggerBase):
    """Logger that prints relevant results of game to the terminal."""

    def __init__(self, config: PrintLoggerConfig) -> None:
        self._config = config

    @override
    def log_game(self, log: GameLog) -> None:
        """

        Log configured game info to the terminal.

        Parameters
        ----------
        log : GameLog
            game log to display
        """
        if self._config.winner:
            match log.game.result:
                case BoardStatus.WHITE:
                    winner = 'White'
                case BoardStatus.BLACK:
                    winner = 'Black'
                case BoardStatus.DRAW:
                    winner = 'Draw'
                case BoardStatus.UNDECIDED:
                    winner = 'Undecided'
                case _:
                    msg = 'Recieved a game log with an unterminated status.'
                    raise ValueError(msg)
            print(f'Wiinner: {winner}')

        if self._config.termination_cause:
            match log.game.termination_type:
                case TerminationType.CHECKMATE:
                    cause = 'Checkmate'
                case TerminationType.REPETITION:
                    cause = 'Repetition'
                case TerminationType.INSUFFICIENT_MATERIAL:
                    cause = 'Insufficient Material'
                case TerminationType.FIFTY_MOVES:
                    cause = 'Fifty Moves'
                case TerminationType.STALEMATE:
                    cause = 'Stalemate'
                case TerminationType.ABORT:
                    cause = 'Abort'
                case _:
                    msg = 'Recieved a game log with an invalid termination type.'
                    raise ValueError(msg)
            print(f'Termination Cause: {cause}')

        if self._config.total_time:
            if log.game.dt is None:
                msg = 'No game Δt provided in game log.'
                raise ValueError(msg)
            time_str = format_time(log.game.dt)
            print(f'Total Game Time: {time_str}')

        if self._config.time_per_move:
            if log.game.dt is None:
                msg = 'No game Δt provided in game log.'
                raise ValueError(msg)
            if log.game.ply_number is None:
                msg = 'No ply number provided in game log.'
                raise ValueError(msg)
            time_str = format_time(log.game.dt / log.game.ply_number)
            print(f'Average Time Per Move: {time_str}')
