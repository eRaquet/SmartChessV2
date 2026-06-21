"""Module for the game architecture."""

import chess

from modules.agent import AgentBase
from modules.board import Board


class Game:
    """Class that specifies game mechanics."""

    def __init__(self, agent_white: AgentBase, agent_black: AgentBase, board: Board) -> None:
        self._agents = {
            chess.WHITE: agent_white,
            chess.BLACK: agent_black,
        }
        self._board = board

    def play_game(self) -> None:
        """Play through a game on the board."""
        while not self._board.terminated:
            action = self._agents[self._board.turn].act(self._board)
            self._board.step(action)
