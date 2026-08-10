"""Random Agent."""

from typing import override

from numpy.random import default_rng

from smartchess.board import Board
from smartchess.types import AgentDecision

from .agent_base import AgentBase
from .config import DEFAULT_RANDOM_AGENT_CONFIG, RandomAgentConfig


class RandomAgent(AgentBase):
    """Agent that picks a random move."""

    def __init__(self, config: RandomAgentConfig = DEFAULT_RANDOM_AGENT_CONFIG) -> None:
        self._rng = default_rng(config.seed)

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
