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

    # check that the number of moves is equal to the number of board positions minus one
    assert len(game.trajectory) - 1 == board.half_move_count
