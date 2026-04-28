"""Module for the game architecture."""

import time

import chess
from agent import AgentBase, RandomAgent
from board import Board

from modules.chess_types import DisplayMode, Trajectory


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

    @property
    def trajectory(self) -> Trajectory:
        """

        Trajectory of the game.

        Returns
        -------
        Trajectory
            shape (n, 8, 8, 18) where n is the number of board states
        """
        return self._board.trajectory


# benchmarking code snippet
if __name__ == "__main__":
    print("Start")
    start = time.perf_counter()

    game = Game(RandomAgent(), RandomAgent(), Board(DisplayMode.NONE))
    game.play_game()

    end = time.perf_counter()

    print(f"Done with {((end - start) / len(game.trajectory) * 1e3):.3f} ms per move")
