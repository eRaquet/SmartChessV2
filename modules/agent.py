"""Module for defining chess agents."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import softmax

from modules.board import Board, GUIBoard
from modules.chess_types import PMF, Action, AgentActionSnapshot, SetEvaluation
from modules.config import DEFAULT_CONFIDENCE
from modules.model import ModelBase, StandardModel


class AgentBase(ABC):
    """Agent base class, specifying structure."""

    snapshot: AgentActionSnapshot | None = None

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

    @abstractmethod
    def _capture(self, *args: Any, **kwargs: Any) -> None:
        """Capture the agent's action to the snapshot."""


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
        action = self._rng.integers(len(board.moves))

        self._capture(action)

        return self._rng.integers(len(board.moves))

    def _capture(self, action: Action) -> None:
        self.snapshot = AgentActionSnapshot(evals=None, dist=None, action=action)


class StandardAgent(AgentBase):
    """Agent that picks a move based on its underlying model."""

    _rng = np.random.default_rng()

    def __init__(
        self, model: ModelBase, confidence_factor: float | None = DEFAULT_CONFIDENCE
    ) -> None:
        self._model = model
        self._confidence_factor = confidence_factor
        self.strain = model.strain if isinstance(model, StandardModel) else None
        self.generation = model.generation if isinstance(model, StandardModel) else None

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
        evals = 1 - self._model.predict_batch(board.observation)  # evaluation as seen by agent

        if self._confidence_factor is None:
            action = np.argmax(evals)

            # create distribution associated with an infinite confidence
            choice_distribution = np.zeros(evals.shape)
            choice_distribution[action] = 1

        else:
            choice_distribution = softmax(evals * self._confidence_factor)

            action = self._rng.choice(len(choice_distribution), p=choice_distribution)

        self._capture(evals, choice_distribution, action)

        return action

    def _capture(self, evals: SetEvaluation, dist: PMF, action: Action) -> None:
        self.snapshot = AgentActionSnapshot(evals=evals, dist=dist, action=action)


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

        self._capture(action)

        return action

    def _capture(self, action: Action) -> None:
        self.snapshot = AgentActionSnapshot(evals=None, dist=None, action=action)
