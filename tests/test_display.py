# ruff: noqa: S101
"""Tests for display class."""

import chess

from modules.display import Display


def test_display_board() -> None:
    """Test the display board function of the Display class."""
    disp = Display()

    board = chess.Board()
    disp.display_board(board)

    board.push(chess.Move(chess.A2, chess.A3))
    disp.display_board(board)

    assert disp.get_user_input(board, list(board.legal_moves)) is None

    disp.exit()
