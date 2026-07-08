"""Module for the game architecture."""

from abc import ABC, abstractmethod
from typing import override

import chess

from modules.agent import AgentBase
from modules.board import Board
from modules.chess_types import GameLog, MoveContext
from modules.collector import Collector


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


class StandardGame(GameBase):
    """Class that specifies game mechanics."""

    def __init__(
        self,
        agent_white: AgentBase,
        agent_black: AgentBase,
        board: Board,
    ) -> None:
        self._agents = {
            chess.WHITE: agent_white,
            chess.BLACK: agent_black,
        }
        self._board = board

    @override
    def play_game(self) -> None:
        """Play through game on the board."""
        while not self._board.terminated:
            current_agent = self._agents[self._board.turn]
            decision = current_agent.act(self._board)

            self._board.step(decision.action)


class LoggedGame(GameBase):
    """Class that specifies game mechanics."""

    def __init__(
        self,
        agent_white: AgentBase,
        agent_black: AgentBase,
        board: Board,
        collector: Collector,
    ) -> None:
        self._agents = {
            chess.WHITE: agent_white,
            chess.BLACK: agent_black,
        }
        self._board = board
        self._collector = collector
        self._collector.select_agent(agent_white, chess.WHITE)
        self._collector.select_agent(agent_black, chess.BLACK)

    @override
    def play_game(self) -> GameLog:
        """

        Play through game on the board and return the game log.

        Returns
        -------
        GameLog
        """
        self._collector.start_game()

        while not self._board.terminated:
            context = MoveContext(
                side_to_move=self._board.turn, ply=self._board.half_move_count + 1
            )
            self._collector.start_move()

            current_agent = self._agents[self._board.turn]
            decision = current_agent.act(self._board)

            result = self._board.step(decision.action)

            self._collector.record_move(context, decision, result)

        return self._collector.finish_game(self._board.outcome)
