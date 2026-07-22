# ruff: noqa: S101
"""Tests for the game module."""

from smartchess.agent import RandomAgent
from smartchess.board import Board
from smartchess.game import StandardGame


def test_game() -> None:
    """Test the Game class."""
    board = Board()
    random_agent = RandomAgent()
    game = StandardGame(random_agent, random_agent, board)
    game.play_game()
    assert board.terminated
