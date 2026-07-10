# ruff: noqa: S101
"""File for testing tools module."""

import chess
import numpy as np

from modules.board import Board
from modules.config import PROJECT_PATH


def test_board() -> None:
    """Integration test the Board class."""
    board = Board()

    assert board.turn == chess.WHITE
    assert board.moves == list(chess.Board().generate_legal_moves())

    test_encoding = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_board_initial_encoding.npy"
    )
    assert np.allclose(board.observation.encodings, test_encoding)
