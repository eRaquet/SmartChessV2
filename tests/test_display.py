# ruff: noqa: S101
"""Tests for display class."""

import os

import chess

# make a blank rendering backend for testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from smartchess.ui import Display


def test_display() -> None:
    """Test the display board function of the Display class."""
    disp = Display()

    board = chess.Board()
    disp.display_board(board)

    board.push(chess.Move(chess.A2, chess.A3))
    disp.display_board(board)

    assert disp.get_user_input(board, list(board.legal_moves)) is None

    disp.exit()
