# ruff: noqa: S101
"""File for testing tools module."""

from pathlib import Path

import chess
import numpy as np

from modules.board import Board

path = Path(__file__).parent


def test_board() -> None:
    """Integration test the Board class."""
    board = Board()

    assert board.turn == chess.WHITE
    assert board.moves == list(chess.Board().generate_legal_moves())

    test_encoding = np.load(path / "data" / "test_position_board_initial_encoding.npy")
    assert np.allclose(board.observation, test_encoding)
