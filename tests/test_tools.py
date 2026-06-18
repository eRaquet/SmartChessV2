# ruff: noqa: S101
# # ruff: noqa: PLR2004
"""File for testing tools module."""

from collections import Counter

import chess
import numpy as np
import pytest

from modules.chess_types import (
    Players,
)
from modules.config import PROJECT_PATH
from modules.tools import (
    encode_board,
    generate_board_encodings_from_moves,
    get_action,
    get_piece_index,
    square_indices,
)


@pytest.fixture
def default_board() -> chess.Board:
    """Create the default board setup and piece map."""
    return chess.Board()


def test_generate_board_encodings_from_moves() -> None:
    """Test the generate_board_encodings_from_moves function."""
    # test white move generation  (FEN is 4k3/6P1/8/4Pp2/3p4/5N2/4P3/R3K2R w KQ f6 0 1)
    board = chess.Board("4k3/6P1/8/4Pp2/3p4/5N2/4P3/R3K2R w KQ f6 0 1")
    moves = list(board.legal_moves)
    encoding = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_white_move_start_encoding.npy"
    )
    repeated_encoding = np.load(
        PROJECT_PATH / "tests" / "data" / "test_repeated_position_white_encoding.npy"
    )
    board_state_counter = Counter(
        [repeated_encoding.copy().tobytes(), repeated_encoding.copy().tobytes()]
    )
    encodings = generate_board_encodings_from_moves(
        encoding, moves, chess.WHITE, board_state_counter
    )

    encodings_truth = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_white_move_encoding.npy"
    )

    assert np.allclose(encodings, encodings_truth)

    # test black move generation  (FEN is r3k2r/4p3/5n2/3P4/4pP2/8/1p2P3/R3K2R b kq f3 0 1)
    board = chess.Board("r3k2r/4p3/5n2/3P4/4pP2/8/1p2P3/R3K2R b kq f3 0 1")
    moves = list(board.legal_moves)
    encoding = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_black_move_start_encoding.npy"
    )
    repeated_encoding = np.load(
        PROJECT_PATH / "tests" / "data" / "test_repeated_position_black_encoding.npy"
    )
    board_state_counter = Counter(
        [repeated_encoding.copy().tobytes(), repeated_encoding.copy().tobytes()]
    )
    encodings = generate_board_encodings_from_moves(
        encoding, moves, chess.BLACK, board_state_counter
    )

    encodings_truth = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_black_move_encoding.npy"
    )

    assert np.allclose(encodings, encodings_truth)


def test_encode_board() -> None:
    """Test the encode_board_obs function."""
    # test white position
    board = chess.Board("4k3/6P1/8/4Pp2/8/8/8/R3K2R w KQ f6 0 1")
    encoding = encode_board(board)

    encoding_truth = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_white_board_encoding.npy"
    )

    assert np.allclose(encoding, encoding_truth)

    # test black position
    board = chess.Board("r3k2r/8/8/8/3Pp3/8/1p6/4K3 b kq d3 0 1")
    encoding = encode_board(board)

    encoding_truth = np.load(
        PROJECT_PATH / "tests" / "data" / "test_position_black_board_encoding.npy"
    )

    assert np.allclose(encoding, encoding_truth)


def test_get_piece_index() -> None:
    """Test the get_piece_index function."""
    # self pieces
    assert get_piece_index(chess.PAWN, Players.SELF) == 0
    assert get_piece_index(chess.KNIGHT, Players.SELF) == 1
    assert get_piece_index(chess.BISHOP, Players.SELF) == 2
    assert get_piece_index(chess.ROOK, Players.SELF) == 3
    assert get_piece_index(chess.QUEEN, Players.SELF) == 4
    assert get_piece_index(chess.KING, Players.SELF) == 5

    # opponent pieces
    assert get_piece_index(chess.PAWN, Players.OPPONENT) == 11
    assert get_piece_index(chess.KNIGHT, Players.OPPONENT) == 10
    assert get_piece_index(chess.BISHOP, Players.OPPONENT) == 9
    assert get_piece_index(chess.ROOK, Players.OPPONENT) == 8
    assert get_piece_index(chess.QUEEN, Players.OPPONENT) == 7
    assert get_piece_index(chess.KING, Players.OPPONENT) == 6


def test_square_indices() -> None:
    """Test the square_indices function."""
    assert square_indices(chess.A1, chess.WHITE) == (0, 0)
    assert square_indices(chess.A1, chess.BLACK) == (7, 0)
    assert square_indices(chess.H8, chess.WHITE) == (7, 7)
    assert square_indices(chess.H8, chess.BLACK) == (0, 7)


def test_get_action() -> None:
    """Test the get_action function."""
    board = chess.Board()
    moves = list(board.legal_moves)

    # some test moves to try getting the action for
    test_moves = [
        chess.Move(chess.G1, chess.H3),
        chess.Move(chess.G1, chess.F3),
        chess.Move(chess.B1, chess.C3),
        chess.Move(chess.B1, chess.A3),
        chess.Move(chess.H2, chess.H3),
        chess.Move(chess.G2, chess.G3),
        chess.Move(chess.F2, chess.F3),
        chess.Move(chess.E2, chess.E3),
        chess.Move(chess.D2, chess.D3),
    ]

    for i, move in enumerate(test_moves):
        assert get_action(move, moves) == i
