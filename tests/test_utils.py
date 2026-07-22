# ruff: noqa: S101
#
"""File for testing tools module."""

import chess
import numpy as np
import pytest

from smartchess.config import PROJECT_PATH
from smartchess.util import (
    encode_board,
)


@pytest.fixture
def default_board() -> chess.Board:
    """Create the default board setup and piece map."""
    return chess.Board()


def test_encode_board() -> None:
    """Test the encode_board_obs function."""
    # test white position
    board = chess.Board('4k3/6P1/8/4Pp2/8/8/8/R3K2R w KQ f6 0 1')
    encoding = encode_board(board)

    encoding_truth = np.load(
        PROJECT_PATH / 'tests' / 'data' / 'test_position_white_board_encoding.npy',
    )

    assert np.allclose(encoding, encoding_truth)

    # test black position
    board = chess.Board('r3k2r/8/8/8/3Pp3/8/1p6/4K3 b kq d3 0 1')
    encoding = encode_board(board)

    encoding_truth = np.load(
        PROJECT_PATH / 'tests' / 'data' / 'test_position_black_board_encoding.npy',
    )

    assert np.allclose(encoding, encoding_truth)
