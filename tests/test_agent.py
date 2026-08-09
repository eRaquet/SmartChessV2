# ruff: noqa: S101
"""Tests for the agent module."""

import numpy as np
import pytest

from smartchess.agent import ModelAgentConfig, RandomAgentConfig, create_agent
from smartchess.board import BoardConfig, create_board
from smartchess.config import DEFAULT_CONFIDENCE
from smartchess.model import InferenceModelConfig, create_model


def test_random_agent() -> None:
    """Test for the RandomAgent class."""
    agent_config = RandomAgentConfig(seed=0)
    agent = create_agent(agent_config)
    board_config = BoardConfig()
    board = create_board(board_config)
    decision = agent.act(board)

    assert 0 <= decision.action < len(board.moves)


@pytest.mark.backend
def test_model_agent() -> None:
    """Test for the ModelAgent class."""
    model_config = InferenceModelConfig(strain=0)
    model = create_model(model_config)

    board_config = BoardConfig()
    board = create_board(board_config)

    deterministic_config = ModelAgentConfig(model=model, confidence=None, seed=0)
    agent_deterministic = create_agent(deterministic_config)
    random_config = ModelAgentConfig(model=model, confidence=DEFAULT_CONFIDENCE, seed=0)
    agent_random = create_agent(random_config)

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
