"""Module for defining chess agents."""

from typing import Any

import numpy as np

from modules.board import Board
from modules.chess_types import (
    Action,
)


class AgentBase:
    """Agent base class, specifying structure."""

    def act(self, board: Board, *args: Any, **kwargs: Any) -> Action:
        """

        Choose an action to play based on the provided observation.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        Action
            index of chosen move

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class RandomAgent(AgentBase):
    """Agent that picks a random move."""

    rng = np.random.default_rng()

    def act(self, board: Board) -> Action:
        """

        Choose a random action.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        Action
            randomly chosen action
        """
        return self.rng.integers(len(board.moves))
