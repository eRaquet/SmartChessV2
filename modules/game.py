"""Module for the game architecture."""

import chess

from modules.agent import AgentBase
from modules.board import Board
from modules.collector import Collector
from modules.utils import write_game


class Game:
    """Class that specifies game mechanics."""

    def __init__(
        self,
        agent_white: AgentBase,
        agent_black: AgentBase,
        board: Board,
        collector: Collector | None = None,
    ) -> None:
        self._agents = {
            chess.WHITE: agent_white,
            chess.BLACK: agent_black,
        }
        self._board = board
        self._collector = collector
        if self._collector:
            self._collector.select_agent(agent_white, chess.WHITE)
            self._collector.select_agent(agent_black, chess.BLACK)

    def play_game(self) -> None:
        """Play through a game on the board."""
        if self._collector:
            self._collector.start_game()

        while not self._board.terminated:
            if self._collector:
                self._collector.start_move()

            current_agent = self._agents[self._board.turn]
            action = current_agent.act(self._board)

            self._board.step(action)

            if self._collector:
                self._collector.get_agent_action(current_agent)
                self._collector.get_board_action(self._board)
                self._collector.finish_move()

        if self._collector:
            game_log = self._collector.finish_game(self._board.outcome)
            write_game(game_log)  # this will be depricated
