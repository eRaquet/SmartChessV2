"""Module for defining chess agents."""

from typing import Any

import numpy as np
from scipy.special import softmax

from modules.board import Board
from modules.chess_types import (
    Action,
)
from modules.model import ModelBase


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


class StandardAgent(AgentBase):
    """Agent that picks a move based on its underlying model."""

    def __init__(self, model: ModelBase, confidence_factor: float | None) -> None:
        self.model = model
        self.rng = np.random.default_rng()
        self.confidence_factor = confidence_factor

    def act(self, board: Board) -> Action:
        """

        Choose an action.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        Action
            chosen action
        """
        evals = self.model.predict_batch(board.observation)

        if self.confidence_factor is None:
            return np.argmax(evals)

        choice_distribution = softmax(evals * self.confidence_factor)

        return self.rng.choice(len(choice_distribution), p=choice_distribution)
