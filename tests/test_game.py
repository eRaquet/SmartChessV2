# ruff: noqa: S101
"""Tests for the game module."""

from modules.agent import RandomAgent
from modules.board import Board
from modules.game import Game


def test_game() -> None:
    """Test the Game class."""
    board = Board()
    random_agent = RandomAgent()
    game = Game(random_agent, random_agent, board)
    game.play_game()
    assert board.terminated
