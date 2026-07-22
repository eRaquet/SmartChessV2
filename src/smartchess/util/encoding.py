"""Helper Functions for Encoding Boards."""

import chess
import numpy as np

from smartchess.types import (
    BOARD_ENCODING_SHAPE,
    PIECE_ENCODING_SHAPE,
    PIECE_INDEX,
    Action,
    BoardEncoding,
    Observation,
    PieceEncoding,
    Players,
    SetEncoding,
)


def encode_pieces_slow(
    piece_map: dict[chess.Square, chess.Piece],
    color_to_move: chess.Color,
) -> PieceEncoding:
    """

    Encode a board position from the perspective of a certain player color.

    This method is used as a reference for the vectorized method.

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
            row,
            col,
            get_piece_index(piece.piece_type, Players(piece.color == color_to_move)),
        ] = 1

    # return contructed board encoding
    return encoded_pieces


def encode_pieces(board: chess.Board) -> PieceEncoding:
    """

    Encode the board pieces using numpy vectorized operations.

    Parameters
    ----------
    board : chess.Board

    Returns
    -------
    PieceEncoding
        output piece encoding, shape (8, 8, 12)
    """
    turn = board.turn

    self = board.occupied_co[turn]
    opponent = board.occupied_co[not turn]

    bitboards = np.array(
        [
            self & board.pawns,
            self & board.knights,
            self & board.bishops,
            self & board.rooks,
            self & board.queens,
            self & board.kings,
            opponent & board.kings,
            opponent & board.queens,
            opponent & board.rooks,
            opponent & board.bishops,
            opponent & board.knights,
            opponent & board.pawns,
        ],
        dtype='<u8',
    )

    # reinterpret the bitboards by partitioning them into individual bytes
    rank_bytes = bitboards.view(np.uint8).reshape(12, 8).T

    if turn == chess.BLACK:
        rank_bytes = rank_bytes[::-1]

    # this step unpacks the remaining axis, using each byte
    return np.unpackbits(
        rank_bytes,
        axis=0,
        bitorder='little',
    ).reshape(PIECE_ENCODING_SHAPE)


def encode_board(board: chess.Board, encoding_array: BoardEncoding | None = None) -> BoardEncoding:
    """

    Generate a board encoding from the provided board.

    Parameters
    ----------
    board : chess.Board
        board to encode
    encoding_array : BoardEncoding | None
        optional output array to encode the board into, assumed to be allocated with zeros, default
        None

    Returns
    -------
    BoardEncoding
        encoded board
    """
    encoded_board: BoardEncoding = (
        np.zeros(BOARD_ENCODING_SHAPE, dtype=np.uint8) if encoding_array is None else encoding_array
    )

    # insert the piece encoding
    encoded_board[:, :, 0:12] = encode_pieces(board)

    # insert the castling rights encoding
    if bool(board.castling_rights & (chess.BB_A1 if board.turn == chess.WHITE else chess.BB_A8)):
        encoded_board[:, :, 12] = 1

    if bool(board.castling_rights & (chess.BB_H1 if board.turn == chess.WHITE else chess.BB_H8)):
        encoded_board[:, :, 13] = 1

    if bool(board.castling_rights & (chess.BB_H8 if board.turn == chess.WHITE else chess.BB_H1)):
        encoded_board[:, :, 14] = 1

    if bool(board.castling_rights & (chess.BB_A8 if board.turn == chess.WHITE else chess.BB_A1)):
        encoded_board[:, :, 15] = 1

    is_draw = (len(board.move_stack) >= 4 and board.is_repetition()) or (  # noqa: PLR2004
        board.halfmove_clock >= 100 and board.is_fifty_moves()  # noqa: PLR2004
    )

    if is_draw:
        encoded_board[:, :, 16] = 1

    ep_square = board.ep_square
    if ep_square is not None and board.has_legal_en_passant():
        encoded_board[*square_indices(ep_square, board.turn), 17] = 1

    return encoded_board


def generate_observation(board: chess.Board, moves: list[chess.Move]) -> Observation:
    """

    Generate the complete observation of the current chess board.

    This is accomplished by pushing each move onto the move stack, encoding the board as is, and
    then popping the move back off the stack.

    Parameters
    ----------
    board : chess.Board
        chess board, in it's current state
    moves : list[chess.Move]
        list of chess moves possible from current board

    Returns
    -------
    Observation
        completed observation of board
    """
    num_moves = len(moves)
    encodings: SetEncoding = np.zeros((num_moves, *BOARD_ENCODING_SHAPE), dtype=np.uint8)
    checkmate_action: Action | None = None

    for i, move in enumerate(moves):
        board.push(move)

        encode_board(board, encodings[i])

        if checkmate_action is None and board.is_checkmate():
            checkmate_action = i

        board.pop()

    return Observation(encodings, checkmate_action)


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
