"""Random Agent."""

from typing import override

from numpy.random import default_rng

from smartchess.board import Board
from smartchess.types import AgentDecision

from .agent_base import AgentBase


class RandomAgent(AgentBase):
    """Agent that picks a random move."""

    _rng = default_rng()

    @override
    def act(self, board: Board) -> AgentDecision:
        """

        Choose a random action.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        AgentDecision
            randomly chosen action
        """
        action = int(self._rng.integers(len(board.moves)))

        return self._capture(action)
