# ruff: noqa: S101
"""Tests for the agent module."""

from modules.agent import RandomAgent, StandardAgent
from modules.board import Board
from modules.model import StandardModel


def test_random_agent() -> None:
    """Test for the RandomAgent class."""
    agent = RandomAgent()
    board = Board()
    action = agent.act(board)

    assert 0 <= action < len(board.moves)


def test_standard_agent() -> None:
    """Test for the StandardAgent class."""
    model = StandardModel(0, 0)
    board = Board()
    agent_deterministic = StandardAgent(model, confidence_factor=None)
    agent_random = StandardAgent(model)

    action = agent_random.act(board)
    action_1 = agent_deterministic.act(board)
    action_2 = agent_deterministic.act(board)

    assert 0 <= action < len(board.moves)
    assert 0 <= action_1 < len(board.moves)

    assert action_1 == action_2
