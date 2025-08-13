# ruff: noqa: S101
"""File for testing tools module."""

from pathlib import Path

import chess
import numpy as np
import pytest

from modules.conventions import (
    Players,
)
from modules.tools import (
    encode_board_obs,
    generate_board_encodings_from_moves,
    get_piece_index,
    square_indices,
)

path = Path(__file__).parent


@pytest.fixture
def test_boards() -> list[chess.Board]:
    """Set of test positions."""
    fens = [
        # 1. Back Rank Mate Threat
        "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
        # 2. Smothered Mate Setup
        "6k1/5ppp/8/8/8/5N2/5PPP/6RK w - - 0 1",
        # 3. Underpromotion to Knight
        "7k/P7/8/8/8/8/6pp/7K w - - 0 1",
        # 4. Zugzwang in Endgame
        "8/8/8/5k2/5p2/5P2/6K1/8 b - - 0 1",
        # 5. Desperado Piece
        "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 4",
        # 6. Trapped Queen
        "rnb1kbnr/ppp2ppp/8/3qp3/3P4/2N1P3/PPP2PPP/R1BQKBNR w KQkq - 0 6",
        # 7. Stalemate Trap
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
        # 8. Knight Fork Puzzle
        "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2N2N2/PPP1QPPP/R1B1KB1R w KQkq - 4 5",
        # 9. Classic Queen Sacrifice
        "r1bqkb1r/pppp1ppp/2n2n2/4P3/1b1P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5",
        # 10. Opening Trap (Legal's Mate)
        "rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 3",
    ]

    return [chess.Board(fen) for fen in fens]


@pytest.fixture
def default_board() -> tuple[chess.Board, dict[chess.Square, chess.Piece]]:
    """Create the default board setup and piece map."""
    board = chess.Board()
    piece_map = board.piece_map()

    return (board, piece_map)


def test_generate_board_encodings_from_moves() -> None:
    """Test the generate_board_encodings_from_moves function."""
    # en passant position
    en_passant_board = chess.Board("8/8/8/4pP2/8/8/8/8 w - e6 0 1")
    en_passant_encoding = encode_board_obs(en_passant_board.piece_map(), chess.WHITE)
    en_passant_move = generate_board_encodings_from_moves(en_passant_encoding, list(en_passant_board.legal_moves), chess.WHITE)[
        1, :, :, [get_piece_index(chess.PAWN, Players.SELF), get_piece_index(chess.PAWN, Players.OPPONENT)]
    ]  # get the en passant move encoding for the pawns
    en_passant_move_truth = np.load(path / "data" / "test_en_passant.npy")
    assert np.allclose(en_passant_move, en_passant_move_truth)

    # promotion
    promotion_board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    promotion_encoding = encode_board_obs(promotion_board.piece_map(), chess.WHITE)
    promotion_move = generate_board_encodings_from_moves(promotion_encoding, list(promotion_board.legal_moves), chess.WHITE)[
        6, :, :, [get_piece_index(chess.PAWN, Players.SELF), get_piece_index(chess.KNIGHT, Players.SELF)]
    ]  # get the promotion move encodings for the pawns and the knight
    promotion_move_truth = np.load(path / "data" / "test_promotion.npy")
    assert np.allclose(promotion_move, promotion_move_truth)

    # castling
    castling_board = chess.Board("r3k2r/pppb1ppp/2npbn2/4p3/2B1P3/2N2N2/PPP2PPP/R3K2R w KQkq - 0 1")
    castling_encoding = encode_board_obs(castling_board.piece_map(), chess.WHITE)
    castling_move = generate_board_encodings_from_moves(castling_encoding, list(castling_board.legal_moves), chess.WHITE)[
        29, :, :, [get_piece_index(chess.ROOK, Players.SELF), get_piece_index(chess.KING, Players.SELF)]
    ]  # get the castling move encodings for the rooks and kings
    castling_move_truth = np.load(path / "data" / "test_castling.npy")
    assert np.allclose(castling_move, castling_move_truth)


def test_encode_board_obs(default_board: tuple[chess.Board, dict[chess.Square, chess.Piece]]) -> None:
    """Test the encode_board_obs function."""
    _, piece_map = default_board
    encoding_white = encode_board_obs(piece_map, chess.WHITE)
    encoding_black = encode_board_obs(piece_map, chess.BLACK)

    encoding_truth = np.load(path / "data" / "test_default_board_encoding.npy")

    assert np.allclose(encoding_white, encoding_truth)
    assert np.allclose(encoding_black, encoding_truth)


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
