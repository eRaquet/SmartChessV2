"""Module for defining chess agents."""

from typing import Any

import numpy as np
from scipy.special import softmax

from modules.board import Board, GUIBoard
from modules.chess_types import (
    Action,
)
from modules.config import DEFAULT_CONFIDENCE
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

    def __init__(
        self, model: ModelBase, confidence_factor: float | None = DEFAULT_CONFIDENCE
    ) -> None:
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


class UIAgent(AgentBase):
    """Agent that gets user input from a board with a GUI."""

    def __init__(self, board: GUIBoard) -> None:
        if type(board) is not GUIBoard:
            msg = "UI Agents can only be instantiated from a GUI Board."
            raise TypeError(msg)

        # core objects that a UIAgent contains
        self._board = board

    def act(self, _: Board) -> Action:
        """

        Get the user input.

        Returns
        -------
        Action
            action to take, specified by user
        """
        action = None
        while action is None:
            action = self._board.get_user_input()
        return action
