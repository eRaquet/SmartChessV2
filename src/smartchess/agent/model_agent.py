"""Model Agent."""

from typing import override

import numpy as np
from numpy.random import default_rng

from smartchess.board import Board
from smartchess.config import DEFAULT_CONFIDENCE
from smartchess.model import InferenceModel, ModelBase
from smartchess.types import PMF, AgentDecision, SetEvaluation
from smartchess.util import softmax

from .agent_base import AgentBase


class ModelAgent(AgentBase):
    """Agent that picks a move based on its underlying model."""

    _rng = default_rng()

    def __init__(
        self,
        model: ModelBase,
        confidence_factor: float | None = DEFAULT_CONFIDENCE,
    ) -> None:
        self._model = model
        self._confidence_factor = confidence_factor
        self.strain = model.strain if isinstance(model, InferenceModel) else None
        self.generation = model.generation if isinstance(model, InferenceModel) else None

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
            board.observation.encodings,
        )  # evaluation as seen by agent

        # handle case where a mate in one was found
        if board.observation.checkmate_action is not None:
            action = board.observation.checkmate_action

            # create distribution associated with an infinite confidence
            choice_distribution: PMF = np.zeros(evals.shape)
            choice_distribution[action] = 1

        # handle case where agent is set to maximum confidence (i.e., pick the best move always)
        elif self._confidence_factor is None:
            action = int(np.argmax(evals))

            # create distribution associated with an infinite confidence
            choice_distribution: PMF = np.zeros(evals.shape)
            choice_distribution[action] = 1

        else:
            choice_distribution: PMF = softmax(evals * self._confidence_factor)

            action = int(self._rng.choice(len(choice_distribution), p=choice_distribution))

        return self._capture(action, evals, choice_distribution)
