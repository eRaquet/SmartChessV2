"""Abstract Agent Base Class."""

from abc import ABC, abstractmethod

from smartchess.board import Board
from smartchess.types import PMF, Action, AgentDecision, SetEvaluation


class AgentBase(ABC):
    """Agent base class, specifying structure."""

    @abstractmethod
    def act(self, board: Board) -> AgentDecision:
        """

        Choose an action to play based on the provided observation.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        AgentDecision
            agent decision data
        """

    def _capture(
        self,
        action: Action,
        evals: SetEvaluation | None = None,
        dist: PMF | None = None,
    ) -> AgentDecision:
        """Capture the agent's decision metadata."""
        return AgentDecision(evals=evals, dist=dist, action=action)
