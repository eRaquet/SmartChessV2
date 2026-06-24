"""Module for defining chess agents."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import softmax

from modules.board import Board, GUIBoard
from modules.chess_types import (
    Action,
)
from modules.collector import LogCollector
from modules.config import DEFAULT_CONFIDENCE
from modules.model import ModelBase


class AgentBase(ABC):
    """Agent base class, specifying structure."""

    @abstractmethod
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


class RandomAgent(AgentBase):
    """Agent that picks a random move."""

    _rng = np.random.default_rng()

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
        return self._rng.integers(len(board.moves))


class StandardAgent(AgentBase):
    """Agent that picks a move based on its underlying model."""

    _rng = np.random.default_rng()

    def __init__(
        self,
        model: ModelBase,
        confidence_factor: float | None = DEFAULT_CONFIDENCE,
        log_collector: LogCollector | None = None,
    ) -> None:
        self.model = model
        self._confidence_factor = confidence_factor
        self._log_collector = log_collector

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
        evals = 1 - self.model.predict_batch(board.observation)

        if self._confidence_factor is None:
            action = np.argmax(evals)

            if self._log_collector:
                # create distribution associated with an infinite confidence
                dist = np.zeros(evals.shape)
                dist[action] = 1

                self._log_collector.insert_model_action(evals, dist, action)

        else:
            choice_distribution = softmax(evals * self._confidence_factor)

            action = self._rng.choice(len(choice_distribution), p=choice_distribution)

            if self._log_collector:
                self._log_collector.insert_model_action(evals, choice_distribution, action)

        return action


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
