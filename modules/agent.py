"""Module for defining chess agents."""

from abc import ABC, abstractmethod
from typing import Any, override

import numpy as np
from scipy.special import softmax

from modules.board import Board, GUIBoard
from modules.chess_types import PMF, Action, AgentDecision, SetEvaluation
from modules.config import DEFAULT_CONFIDENCE
from modules.model import ModelBase, StandardModel


class AgentBase(ABC):
    """Agent base class, specifying structure."""

    @abstractmethod
    def act(self, board: Board, *args: Any, **kwargs: Any) -> AgentDecision:
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
        self, action: Action, evals: SetEvaluation | None = None, dist: PMF | None = None
    ) -> AgentDecision:
        """Capture the agent's decision metadata."""
        return AgentDecision(evals=evals, dist=dist, action=action)


class RandomAgent(AgentBase):
    """Agent that picks a random move."""

    _rng = np.random.default_rng()

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

    @override
    def act(self, board: Board) -> AgentDecision:
        """

        Choose an action.

        Parameters
        ----------
        board : Board
            board object

        Returns
        -------
        AgentDecision
            chosen action data
        """
        evals: SetEvaluation = 1 - self._model.predict_batch(
            board.observation.encodings
        )  # evaluation as seen by agent

        if self._confidence_factor is None:
            action = int(np.argmax(evals))

            # create distribution associated with an infinite confidence
            choice_distribution: PMF = np.zeros(evals.shape)
            choice_distribution[action] = 1

        else:
            choice_distribution: PMF = softmax(evals * self._confidence_factor)

            action = int(self._rng.choice(len(choice_distribution), p=choice_distribution))

        return self._capture(action, evals, choice_distribution)


class UIAgent(AgentBase):
    """Agent that gets user input from a board with a GUI."""

    def __init__(self, board: GUIBoard) -> None:
        if type(board) is not GUIBoard:
            msg = "UI Agents can only be instantiated from a GUI Board."
            raise TypeError(msg)

        # core objects that a UIAgent contains
        self._board: GUIBoard = board

    @override
    def act(self, _: Board) -> AgentDecision:
        """

        Get the user input.

        Returns
        -------
        AgentDecision
            action to take, specified by user
        """
        action = None
        while action is None:
            action = self._board.get_user_input()

        return self._capture(action)
