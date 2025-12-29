# ruff: noqa: S101
"""File for testing tools module."""

from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pytest

from modules.conventions import (
    Players,
)
from modules.tools import (
    encode_board,
    generate_board_encodings_from_moves,
    get_piece_index,
    square_indices,
)

path = Path(__file__).parent


@pytest.fixture
def default_board() -> chess.Board:
    """Create the default board setup and piece map."""
    return chess.Board()


def test_generate_board_encodings_from_moves() -> None:
    """Test the generate_board_encodings_from_moves function."""
    # test white move generation  (FEN is 4k3/6P1/8/4Pp2/3p4/5N2/4P3/R3K2R w KQ f6 0 1)
    board = chess.Board("4k3/6P1/8/4Pp2/3p4/5N2/4P3/R3K2R w KQ f6 0 1")
    moves = list(board.legal_moves)
    encoding = np.load(Path(__file__).parent / "data" / "test_position_white_move_start_encoding.npy")
    repeated_encoding = np.load(Path(__file__).parent / "data" / "test_repeated_position_white_encoding.npy")
    board_state_counter = Counter([repeated_encoding.copy().tobytes(), repeated_encoding.copy().tobytes()])
    encodings = generate_board_encodings_from_moves(encoding, moves, chess.WHITE, board_state_counter)

    encodings_truth = np.load(Path(__file__).parent / "data" / "test_position_white_move_encoding.npy")

    assert np.allclose(encodings, encodings_truth)

    # test black move generation  (FEN is r3k2r/4p3/5n2/3P4/4pP2/8/1p2P3/R3K2R b kq f3 0 1)
    board = chess.Board("r3k2r/4p3/5n2/3P4/4pP2/8/1p2P3/R3K2R b kq f3 0 1")
    moves = list(board.legal_moves)
    encoding = np.load(Path(__file__).parent / "data" / "test_position_black_move_start_encoding.npy")
    repeated_encoding = np.load(Path(__file__).parent / "data" / "test_repeated_position_black_encoding.npy")
    board_state_counter = Counter([repeated_encoding.copy().tobytes(), repeated_encoding.copy().tobytes()])
    encodings = generate_board_encodings_from_moves(encoding, moves, chess.BLACK, board_state_counter)

    encodings_truth = np.load(Path(__file__).parent / "data" / "test_position_black_move_encoding.npy")

    assert np.allclose(encodings, encodings_truth)


def test_encode_board() -> None:
    """Test the encode_board_obs function."""
    # test white position
    board = chess.Board("4k3/6P1/8/4Pp2/8/8/8/R3K2R w KQ f6 0 1")
    encoding = encode_board(board)

    encoding_truth = np.load(path / "data" / "test_position_white_board_encoding.npy")

    assert np.allclose(encoding, encoding_truth)

    # test black position
    board = chess.Board("r3k2r/8/8/8/3Pp3/8/1p6/4K3 b kq d3 0 1")
    encoding = encode_board(board)

    encoding_truth = np.load(path / "data" / "test_position_black_board_encoding.npy")

    assert np.allclose(encoding, encoding_truth)


def test_get_piece_index(default_board: tuple[chess.Board, dict[chess.Square, chess.Piece]]) -> None:
    """Test the get_piece_index function."""
    player = chess.WHITE

    _, piece_map = default_board

    # extract pieces from the piece_map and sort them into a consistent order
    pieces = sorted(piece_map.values(), key=lambda piece: hash(piece))
    piece_indices_white = [get_piece_index(piece.piece_type, Players(player == piece.color)) for piece in pieces]
    piece_indices_black = [get_piece_index(piece.piece_type, Players(player != piece.color)) for piece in pieces]

    piece_indices_white_truth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 9, 9, 8, 8, 7, 6]

    assert piece_indices_white_truth == piece_indices_white
    assert piece_indices_white_truth == [*piece_indices_black[16:32], *piece_indices_black[0:16]]


def test_square_indices() -> None:
    """Test the square_indices function."""
    assert square_indices(chess.A1, chess.WHITE) == (0, 0)
    assert square_indices(chess.A1, chess.BLACK) == (7, 0)
    assert square_indices(chess.H8, chess.WHITE) == (7, 7)
    assert square_indices(chess.H8, chess.BLACK) == (0, 7)
