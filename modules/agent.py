"""Module for defining chess agents."""

from typing import Any

from modules.chess_types import (
    Action,
    Observation,
)


class AgentBase:
    """Agent base class, specifying structure."""

    def act(self, obs: Observation, *args: Any, **kwargs: Any) -> Action:
        """

        Choose an action to play based on the provided observation.

        Parameters
        ----------
        obs : Observation
            observation of the current board state

        Returns
        -------
        Action
            index of chosen move

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
