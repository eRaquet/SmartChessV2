"""File containing helper functions for chess bot."""

from collections import Counter

import chess
import numpy as np
from scipy.stats import entropy

from modules.chess_types import (
    BOARD_ENCODING_SHAPE,
    MAX_COL_INDEX,
    MAX_ROW_INDEX,
    MIN_COL_INDEX,
    PIECE_ENCODING_SHAPE,
    PIECE_INDEX,
    PMF,
    ROOK_CASTLE_FILE_KINGSIDE,
    ROOK_CASTLE_FILE_QUEENSIDE,
    Action,
    BoardEncoding,
    MoveVector,
    PieceEncoding,
    Players,
    SetEncoding,
)

rng = np.random.default_rng()


def encode_pieces(
    piece_map: dict[chess.Square, chess.Piece], color_to_move: chess.Color
) -> PieceEncoding:
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
    PieceEncoding
        Board encoding, of shape PIECE_ENCODING_SHAPE.
        Axes discription:
            rank,
            file,
            piece type (first 6 are your pieces, next 6 are opponent's pieces)
    """
    # allocate memory for board encoding
    # description: (rank, file, piece type)
    encoded_pieces = np.zeros(PIECE_ENCODING_SHAPE, dtype=np.uint8)

    # iterate through piece map and set respective bits
    for square, piece in piece_map.items():
        row, col = square_indices(square, color_to_move)
        encoded_pieces[
            row, col, get_piece_index(piece.piece_type, Players(piece.color == color_to_move))
        ] = 1

    # return contructed board encoding
    return encoded_pieces


def encode_board(board: chess.Board) -> BoardEncoding:
    """

    Generate a board encoding from the provided board.

    Parameters
    ----------
    board : chess.Board
        board to encode

    Returns
    -------
    BoardEncoding
        encoded board
    """
    ones_board = np.ones((8, 8), dtype=np.uint8)
    zero_board = np.zeros((8, 8), dtype=np.uint8)

    encoded_board = np.zeros(BOARD_ENCODING_SHAPE, dtype=np.uint8)

    # insert the piece encoding
    encoded_board[:, :, 0:12] = encode_pieces(board.piece_map(), board.turn)

    # insert the castling rights encoding
    if bool(board.castling_rights & (chess.BB_A1 if board.turn == chess.WHITE else chess.BB_A8)):
        encoded_board[:, :, 12] = ones_board
    else:
        encoded_board[:, :, 12] = zero_board

    if bool(board.castling_rights & (chess.BB_H1 if board.turn == chess.WHITE else chess.BB_H8)):
        encoded_board[:, :, 13] = ones_board
    else:
        encoded_board[:, :, 13] = zero_board

    if bool(board.castling_rights & (chess.BB_H8 if board.turn == chess.WHITE else chess.BB_H1)):
        encoded_board[:, :, 14] = ones_board
    else:
        encoded_board[:, :, 14] = zero_board

    if bool(board.castling_rights & (chess.BB_A8 if board.turn == chess.WHITE else chess.BB_A1)):
        encoded_board[:, :, 15] = ones_board
    else:
        encoded_board[:, :, 15] = zero_board

    if board.is_repetition() or board.is_fifty_moves():
        encoded_board[:, :, 16] = ones_board
    else:
        encoded_board[:, :, 16] = zero_board

    if board.has_legal_en_passant():
        encoded_board[*square_indices(board.ep_square, board.turn), 17] = 1  # type: ignore[arg-type]

    return encoded_board


def generate_board_encodings_from_moves(  # noqa: PLR0915
    encoding: BoardEncoding,
    moves: MoveVector,
    player_color: chess.Color,
    board_state_counter: Counter[bytes],
) -> SetEncoding:
    """

    Generate the set of bit boards that correspond to a particular position's possible moves.

    Parameters
    ----------
    encoding : BoardEncoding
        The board encoding of the current position from the view of player_color
    moves : MoveVector
        List of moves possible from the current position
    player_color : chess.Color
        Color of the player whose turn it is (and whose BoardEncoding was created)
    board_state_counter : Counter[bytes]
        Counter containing the bytes of all previous board encodings to use for finding three-fold
        repetition

    Returns
    -------
    SetEncoding
        The returned board encodings, as seen by the other player
        shape: (number of moves, 8, 8, 18)
    """
    num_moves = len(moves)
    opponent_color = not player_color

    # allocate the encodings array
    encodings = np.zeros((len(moves), 8, 8, 18), dtype=np.uint8)

    # transfer the piece encodings
    flipped_piece_encoding = np.flip(
        encoding[:, :, 0:12], (-3, -1)
    )  # flip the encodings to the other player
    encodings[:, :, :, 0:12] = flipped_piece_encoding  # fill in the flipped piece encodings

    # transfer the caslting encodings
    flipped_castle_encoding = np.flip(
        encoding[:, :, 12:16], -1
    )  # flip the castling rights to the other player
    encodings[:, :, :, 12:16] = flipped_castle_encoding  # fill in the flipped castle encoding

    ### Construct move info

    # move squares
    move_squares_index = np.array(
        [  # (move index, from/to square, row/column)
            [
                square_indices(move.from_square, opponent_color),
                square_indices(move.to_square, opponent_color),
            ]
            for move in moves
        ]
    )

    # create slicing tools
    move_range = np.arange(num_moves, dtype=np.uint8)
    from_squares_row = move_squares_index[:, 0, 0]
    from_squares_col = move_squares_index[:, 0, 1]
    to_squares_row = move_squares_index[:, 1, 0]
    to_squares_col = move_squares_index[:, 1, 1]

    # get piece types
    piece_indices = np.argmax(flipped_piece_encoding[from_squares_row, from_squares_col], axis=1)
    promotions = np.array(
        [
            get_piece_index(move.promotion, Players.OPPONENT) if move.promotion is not None else 0
            for move in moves
        ]
    )

    ### Run parallel move encoding

    # check for en passant captures

    en_passant_indices = np.where(
        (piece_indices == get_piece_index(chess.PAWN, Players.OPPONENT))  # piece is a pawn
        & (from_squares_col != to_squares_col)  # pawn moved diagonally
        & (
            np.any(encodings[move_range, to_squares_row, to_squares_col, 0:6], axis=1) == 0
        )  # no piece at the location the pawn moved to capture
    )

    if len(en_passant_indices[0]) != 0:
        # remove captured pawn
        encodings[
            move_range[en_passant_indices],
            to_squares_row[en_passant_indices] + 1,
            to_squares_col[en_passant_indices],
            get_piece_index(chess.PAWN, Players.SELF),
        ] = 0

    # check for en passant moves
    en_passant_move_indices = np.where(
        (piece_indices == get_piece_index(chess.PAWN, Players.OPPONENT))  # piece is a pawn
        & (
            from_squares_row == to_squares_row + 2
        )  # the pawn moved twice (since it will always be opponent movement, I don't need abs())
        & (
            (
                np.any(
                    encodings[
                        move_range,
                        to_squares_row,
                        np.clip(to_squares_col + 1, MIN_COL_INDEX, MAX_COL_INDEX),
                        0,
                    ]
                )
            )  # self pawn to the right
            | (
                np.any(
                    encodings[
                        move_range,
                        to_squares_row,
                        np.clip(to_squares_col - 1, MIN_COL_INDEX, MAX_COL_INDEX),
                        0,
                    ]
                )
            )  # self pawn to the left
        )
    )

    if len(en_passant_move_indices[0]) != 0:
        encodings[
            move_range[en_passant_move_indices],
            to_squares_row[en_passant_move_indices] + 1,
            to_squares_col[en_passant_move_indices],
            17,  # this is the en passant capture channel
        ] = 1

    # move the piece to the new location (note that this also removes any pieces in the new
    # location...capture!)
    encodings[move_range, to_squares_row, to_squares_col] = encodings[
        move_range, from_squares_row, from_squares_col
    ]

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
            get_piece_index(chess.PAWN, Players.OPPONENT),
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
        (piece_indices == get_piece_index(chess.KING, Players.OPPONENT))  # piece is a king
        & (to_squares_col - from_squares_col == 2)  # noqa: PLR2004
    )
    queen_side_indices = np.where(
        (piece_indices == get_piece_index(chess.KING, Players.OPPONENT))  # piece is a king
        & (to_squares_col - from_squares_col == -2)  # noqa: PLR2004
    )

    # move rooks
    if len(king_side_indices[0]) > 0:
        encodings[
            move_range[king_side_indices],
            MAX_ROW_INDEX,
            MAX_COL_INDEX,
            get_piece_index(chess.ROOK, Players.OPPONENT),
        ] = 0
        encodings[
            move_range[king_side_indices],
            MAX_ROW_INDEX,
            ROOK_CASTLE_FILE_KINGSIDE,
            get_piece_index(chess.ROOK, Players.OPPONENT),
        ] = 1
    if len(queen_side_indices[0]) > 0:
        encodings[
            move_range[queen_side_indices],
            MAX_ROW_INDEX,
            MIN_COL_INDEX,
            get_piece_index(chess.ROOK, Players.OPPONENT),
        ] = 0
        encodings[
            move_range[queen_side_indices],
            MAX_ROW_INDEX,
            ROOK_CASTLE_FILE_QUEENSIDE,
            get_piece_index(chess.ROOK, Players.OPPONENT),
        ] = 1

    # adjust castling rights

    # determine if the king moved and undo castling rights if so
    if (
        encoding[0, 0, 12] == 1 or encoding[0, 0, 13] == 1
    ):  # if the opponent had any caslting rights
        king_movement_non_castling = np.where(
            piece_indices == get_piece_index(chess.KING, Players.OPPONENT)
        )

        if len(king_movement_non_castling[0]) > 0:
            encodings[move_range[king_movement_non_castling], :, :, 14:16] = 0

    # check queen side castling rights
    if encoding[0, 0, 12] == 1:
        queen_side_rook_movement = np.where(
            (piece_indices == get_piece_index(chess.ROOK, Players.OPPONENT))  # piece is a rook
            & (from_squares_col == MIN_COL_INDEX)  # piece moved from the left of the chessboard
            & (from_squares_row == MAX_ROW_INDEX)  # piece moved from the top of the chessboard
        )
        if len(queen_side_rook_movement[0]) > 0:
            encodings[move_range[queen_side_rook_movement], :, :, 15] = 0
        if len(queen_side_indices[0]) > 0:
            encodings[move_range[queen_side_indices], :, :, 15] = 0

    # check king side castling rights
    if encoding[0, 0, 13] == 1:
        king_side_rook_movement = np.where(
            (piece_indices == get_piece_index(chess.ROOK, Players.OPPONENT))  # piece is a rook
            & (from_squares_col == MAX_COL_INDEX)  # piece moved from the left of the chessboard
            & (from_squares_row == MAX_ROW_INDEX)  # piece moved from the top of the chessboard
        )
        if len(king_side_rook_movement[0]) > 0:
            encodings[move_range[king_side_rook_movement], :, :, 14] = 0
        if len(king_side_indices[0]) > 0:
            encodings[move_range[king_side_indices], :, :, 14] = 0

    # look for three-fold repetition
    counts = np.array(
        [board_state_counter.get(board.tobytes(), 0) for board in encodings], dtype=np.uint8
    )
    repetition_indices = np.where(counts > 1)

    if len(repetition_indices[0]) > 0:
        encodings[move_range[repetition_indices], :, :, 16] = 1

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
        **Note**: not piece color, but whether or not the piece belongs to the player the encoding
        is being created for.

    Returns
    -------
    int
        Piece index into board encoding
    """
    return PIECE_INDEX[(piece_type, player)]


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


def get_action(move: chess.Move, vector: MoveVector) -> Action:
    """

    Obtain the action that gives the selected move from the given move vector.

    Parameters
    ----------
    move : chess.Move
        move to create action for
    vector : MoveVector
        current move vector

    Returns
    -------
    Action
        index of given move in move vector
    """
    return vector.index(move)


def calculate_policy_entropy(dist: PMF | None) -> float | None:
    """

    Calculate the policy entropy from the provided distribution.

    Parameters
    ----------
    dist : PMF | None
        choice distribution of policy, None if no N/A

    Returns
    -------
    float | None
        returned Shannon entropy, or None if not applicable
    """
    if dist is not None:
        return entropy(dist)
    return None


def get_new_id() -> int:
    """

    Temporary function to return dummy "id's" for testing.

    Returns
    -------
    int
        returned id
    """
    return rng.integers(0, 2**16 - 1)
