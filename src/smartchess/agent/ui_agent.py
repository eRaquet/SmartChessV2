"""UI Agent."""

from typing import override

from smartchess.board import Board
from smartchess.types import ABORT_ACTION, AgentDecision, Quit

from .agent_base import AgentBase
from .config import UIAgentConfig


class UIAgent(AgentBase):
    """Agent that gets user input from a board with a GUI."""

    def __init__(self, config: UIAgentConfig) -> None:
        self._move_source = config.move_source

    @override
    def act(self, board: Board) -> AgentDecision:
        """

        Choose an action for this board state.

        Parameters
        ----------
        board : Board
            board to act on

        Returns
        -------
        AgentDecision
            returned action, specified by user
        """
        state = board.snapshot

        while (move := self._move_source.request_move(state)) is None:
            pass

        action = ABORT_ACTION if move is Quit else board.moves.index(move)  # ty:ignore[invalid-argument-type]
        return self._capture(action)
