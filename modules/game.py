"""Module for the game architecture."""

import chess

from modules.agent import AgentBase
from modules.board import Board
from modules.collector import LogCollector


class Game:
    """Class that specifies game mechanics."""

    def __init__(
        self,
        agent_white: AgentBase,
        agent_black: AgentBase,
        board: Board,
        log_collector: LogCollector | None = None,
    ) -> None:
        self._agents = {
            chess.WHITE: agent_white,
            chess.BLACK: agent_black,
        }
        self._board = board
        self._log_collector = log_collector
        if self._log_collector:
            self._log_collector.select_agent(agent_white, chess.WHITE)
            self._log_collector.select_agent(agent_black, chess.BLACK)

    def play_game(self) -> None:
        """Play through a game on the board."""
        if self._log_collector:
            self._log_collector.new_game()

        while not self._board.terminated:
            if self._log_collector:
                self._log_collector.new_move()

            action = self._agents[self._board.turn].act(self._board)
            self._board.step(action)

        if self._log_collector:
            self._log_collector.write_game()
