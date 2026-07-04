"""Module for the game architecture."""

import chess

from modules.agent import AgentBase
from modules.board import Board
from modules.chess_types import GameLog, MoveContext
from modules.collector import Collector


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

    def play_game(self) -> GameLog | None:
        """

        Play through game on the board, returning the game log if a collector was provided.

        Returns
        -------
        GameLog | None
        """
        if self._collector:
            self._collector.start_game()

        while not self._board.terminated:
            if self._collector:
                context = MoveContext(
                    side_to_move=self._board.turn, ply=self._board.half_move_count + 1
                )
                self._collector.start_move()

            current_agent = self._agents[self._board.turn]
            decision = current_agent.act(self._board)

            result = self._board.step(decision.action)

            if self._collector:
                self._collector.record_move(context, decision, result)

        if self._collector:
            return self._collector.finish_game(self._board.outcome)
        return None
