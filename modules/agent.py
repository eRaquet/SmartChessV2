"""Module for defining chess agents."""

from typing import Any

import numpy as np
from scipy.special import softmax

from modules.board import Board
from modules.chess_types import (
    Action,
)
from modules.model import RandomModel


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


class RandomAgentModelBased(AgentBase):
    """Agent that picks a random move based on a random model."""

    model = RandomModel()
    rng = np.random.default_rng()
    confidence_factor = 1.0

    def __init__(self, confidence_factor: float | None = None) -> None:
        """

        Initialize Ranodm Agent Model Based.

        Parameters
        ----------
        confidence_factor : float | None, optional
            Confidence factor to scale the softmax function by, by default None
        """
        self.confidence_factor = confidence_factor

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
        evals = self.model.predict_batch(board.observation)

        if self.confidence_factor is None:
            return np.argmax(evals)

        choice_distribution = softmax(evals * self.confidence_factor)

        return self.rng.choice(len(choice_distribution), p=choice_distribution)
