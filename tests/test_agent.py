# ruff: noqa: S101
"""Tests for the agent module."""

import numpy as np

from smartchess.agent import ModelAgent, RandomAgent
from smartchess.board import Board
from smartchess.model import InferenceModel


def test_random_agent() -> None:
    """Test for the RandomAgent class."""
    agent = RandomAgent()
    board = Board()
    decision = agent.act(board)

    assert 0 <= decision.action < len(board.moves)


def test_model_agent() -> None:
    """Test for the StandardAgent class."""
    model = InferenceModel(0)
    board = Board()
    agent_deterministic = ModelAgent(model, confidence_factor=None)
    agent_random = ModelAgent(model)

    decision = agent_random.act(board)
    decision_1 = agent_deterministic.act(board)
    decision_2 = agent_deterministic.act(board)

    assert decision_1.evals is not None
    assert decision_2.evals is not None
    assert decision_1.dist is not None
    assert decision_2.dist is not None

    assert 0 <= decision.action < len(board.moves)
    assert 0 <= decision_1.action < len(board.moves)

    assert decision_1.action == decision_2.action
    assert np.allclose(decision_1.evals, decision_2.evals)
    assert np.allclose(decision_1.dist, decision_2.dist)
