"""File containing helper functions for chess bot."""

import chess
import numpy as np

import modules.conventions as lib
from modules.chess_types import (
    BoardEncoding,
    Players,
    SetEncoding,
)


def encode_board_obs(piece_map: dict[chess.Square, chess.Piece], color_to_move: chess.Color) -> BoardEncoding:
    """

    Encode a board position from the perspective of a certain player color.

    Parameters
    ----------
    piece_map : dict[chess.Square, chess.Piece]
        Piece map of board position, found by calling chess.Board.piece_map()
    color_to_move : chess.Color
        Color to encode for.  It reflects the board along it's length, but not it's width,
        to swap from one player to another.

    Returns
    -------
    BoardEncoding
        Board encoding, of shape (8, 8, 12).
        Axes discription:
            rank,
            file,
            piece type (first 6 are your pieces, last 6 are opponent's pieces)
    """
    # allocate memory for board encoding
    # description: (rank, file, piece type)
    board_encoded = np.zeros((8, 8, 12), dtype=np.uint8)

    # iterate through piece map and set respective bits
    for square, piece in piece_map.items():
        row, col = square_indices(square, color_to_move)
        board_encoded[row, col, get_piece_index(piece.piece_type, Players(piece.color == color_to_move))] = 1

    # return contructed board encoding
    return board_encoded


def generate_board_encodings_from_moves(encoding: BoardEncoding, moves: list[chess.Move], player_color: chess.Color) -> SetEncoding:
    """

    Generate the set of bit boards that correspond to a particular position's possible moves.

    Parameters
    ----------
    encoding : BoardEncoding
        The board encoding of the current position
    moves : list[chess.Move]
        List of moves possible from the current position
    player_color : chess.Color
        Color of the player whose turn it is (and whose BoardEncoding was created)

    Returns
    -------
    SetEncoding
        The returned board encodings,
        shape: (number of moves, 8, 8, 12)
    """
    num_moves = len(moves)
    encodings = np.repeat(encoding[np.newaxis, ...], num_moves, axis=0)

    ### Construct move info

    # move squares
    move_squares_index = np.array(
        [  # (move index, from/to square, row/column)
            [square_indices(move.from_square, player_color), square_indices(move.to_square, player_color)] for move in moves
        ]
    )

    # create slicing tools
    move_range = np.arange(num_moves, dtype=np.uint8)
    from_squares_row = move_squares_index[:, 0, 0]
    from_squares_col = move_squares_index[:, 0, 1]
    to_squares_row = move_squares_index[:, 1, 0]
    to_squares_col = move_squares_index[:, 1, 1]

    # get piece types
    piece_indices = np.argmax(encoding[from_squares_row, from_squares_col], axis=1)
    promotions = np.array([get_piece_index(move.promotion, Players.SELF) if move.promotion is not None else 0 for move in moves])

    ### Run parallel move encoding

    # check for en passant

    en_passant_indices = np.where(
        (piece_indices == get_piece_index(chess.PAWN, Players.SELF))  # piece is a pawn
        & (from_squares_col != to_squares_col)  # pawn moved diagonally
        & (
            np.any(encodings[move_range, to_squares_row, to_squares_col, 7:12], axis=1) == 0
        )  # no piece at the location the pawn moved to capture
    )

    if len(en_passant_indices[0]) != 0:
        # remove captured pawn
        encodings[
            move_range[en_passant_indices],
            to_squares_row[en_passant_indices] - 1,
            to_squares_col[en_passant_indices],
            get_piece_index(chess.PAWN, Players.OPPONENT),
        ] = 0

    # move the piece to the new location (note that this also removes any pieces in the new location...capture!)
    encodings[move_range, to_squares_row, to_squares_col] = encodings[move_range, from_squares_row, from_squares_col]

    # remove the old piece location
    encodings[move_range, from_squares_row, from_squares_col, piece_indices] = 0

    # handle promotions
    promotion_indices = np.where(promotions != 0)
    # if there are any promotion moves
    if len(promotion_indices[0]) > 0:
        # remove promoted pawns
        encodings[
            move_range[promotion_indices],
            to_squares_row[promotion_indices],
            to_squares_col[promotion_indices],
            get_piece_index(chess.PAWN, Players.SELF),
        ] = 0
        # place new pieces
        encodings[
            move_range[promotion_indices],
            to_squares_row[promotion_indices],
            to_squares_col[promotion_indices],
            promotions[promotion_indices],
        ] = 1

    # handle castling
    king_side_indices = np.where(
        (piece_indices == get_piece_index(chess.KING, Players.SELF))  # piece is a king
        & (to_squares_col - from_squares_col == 2)  # noqa: PLR2004
    )
    queen_side_indices = np.where(
        (piece_indices == get_piece_index(chess.KING, Players.SELF))  # piece is a king
        & (to_squares_col - from_squares_col == -2)  # noqa: PLR2004
    )

    # move rooks
    if len(king_side_indices[0]) > 0:
        encodings[move_range[king_side_indices], 0, 7, get_piece_index(chess.ROOK, Players.SELF)] = 0
        encodings[move_range[king_side_indices], 0, 5, get_piece_index(chess.ROOK, Players.SELF)] = 1
    if len(queen_side_indices[0]) > 0:
        encodings[move_range[queen_side_indices], 0, 0, get_piece_index(chess.ROOK, Players.SELF)] = 0
        encodings[move_range[queen_side_indices], 0, 3, get_piece_index(chess.ROOK, Players.SELF)] = 1

    return encodings


def get_piece_index(piece_type: chess.PieceType, player: Players) -> int:
    """

    Get the piece type index in a board encoding for a given piece.

    Parameters
    ----------
    piece_type : chess.PieceType
        Type of piece as defined by chess library
    player : Players
        Player type.
        **Note**: not piece color, but whether or not the piece belongs to the player the encoding is being created for.

    Returns
    -------
    int
        Piece index into board encoding
    """
    return lib.piece_index[(piece_type, player)]


def square_indices(square: chess.Square, player_color: chess.Color) -> tuple[int, int]:
    """

    Find the indices of a certain square from the perspective of the given player.

    Parameters
    ----------
    square : chess.Square
        Square to calculate for
    player_color : chess.Color
        Player color to view from

    Returns
    -------
    tuple[int, int]
        row and column index of square in encoding space
    """
    return square // 8 if player_color == chess.WHITE else 7 - square // 8, square % 8
