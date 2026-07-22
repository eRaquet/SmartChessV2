"""Standard Game Implementation."""

from typing import override

import chess

from smartchess.agent import AgentBase
from smartchess.board import Board

from .game_base import GameBase


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
