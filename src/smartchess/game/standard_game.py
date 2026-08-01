"""Standard Game Implementation."""

from typing import cast, override

import chess

from smartchess.agent import AgentConfig, create_agent
from smartchess.board import BoardConfig, create_board
from smartchess.pipeline import Collector
from smartchess.types import GameLog, MoveContext

from .config import StandardGameConfig
from .game_base import GameBase


class StandardGame(GameBase):
    """Class that specifies game mechanics."""

    def __init__(self, config: StandardGameConfig) -> None:
        # create agents if only a config was provided
        if isinstance(config.white, AgentConfig):
            config.white = create_agent(config.white)
        if isinstance(config.black, AgentConfig):
            config.black = create_agent(config.black)

        # create board if only a config was provided
        if isinstance(config.board, BoardConfig):
            config.board = create_board(config.board)

        # construct collector if logger config is present
        if config.collect:
            self._collector = Collector()
            self._collector.select_agent(config.white, chess.WHITE)
            self._collector.select_agent(config.black, chess.BLACK)
        else:
            self._collector = None

        self._agents = {
            chess.WHITE: config.white,
            chess.BLACK: config.black,
        }
        self._board = config.board

        self._renderer = config.renderer

    @override
    def play_game(self) -> GameLog | None:
        """

        Play through the game.

        Returns
        -------
        GameLog | None
            game log, if game was collected, else None
        """
        if self._collector is not None:
            return self._play_logged_game()

        return self._play_unlogged_game()

    def _play_logged_game(self) -> GameLog:
        self._collector = cast('Collector', self._collector)

        if self._renderer is not None:
            self._renderer.render(self._board.snapshot)

        self._collector.start_game()

        while not self._board.terminated:
            context = MoveContext(
                side_to_move=self._board.turn,
                ply=self._board.half_move_count + 1,
            )
            self._collector.start_move()

            current_agent = self._agents[self._board.turn]
            decision = current_agent.act(self._board)

            result = self._board.step(decision.action)

            self._collector.record_move(context, decision, result)

            if self._renderer is not None:
                self._renderer.render(self._board.snapshot)

        return self._collector.finish_game(self._board.outcome)

    def _play_unlogged_game(self) -> None:
        if self._renderer is not None:
            self._renderer.render(self._board.snapshot)

        while not self._board.terminated:
            current_agent = self._agents[self._board.turn]
            decision = current_agent.act(self._board)

            self._board.step(decision.action)

            if self._renderer is not None:
                self._renderer.render(self._board.snapshot)
