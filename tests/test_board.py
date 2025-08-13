# ruff: noqa: S101
"""File for testing tools module."""

from pathlib import Path

import numpy as np

from modules.board import Board

path = Path(__file__).parent


def test_reset() -> None:
    """Test Board's reset function."""
    board = Board()
    obs, _ = board.reset()

    # check whether the board correctly reset
    default_encoding = np.load(path / "data" / "test_default_board_encoding.npy")
    assert np.allclose(default_encoding, board.encoding)

    # check whether the observation is a valid observation
    assert board.observation_space.contains(obs)


def test_step() -> None:
    """Test Board's step function."""
    board = Board()
    board.reset()

    obs, *_ = board.step(0)

    # check whether the board correctly stepped
    encoding_after_move = np.load(path / "data" / "test_encoding_after_move.npy")
    assert np.allclose(encoding_after_move, board.encoding)

    # check whether the observation is a valid observation
    assert board.observation_space.contains(obs)
