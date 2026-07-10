"""File containing helper functions for chess bot."""

from dataclasses import astuple, fields
from pathlib import Path
from typing import cast

import chess
import numpy as np
from scipy.stats import entropy
from tabulate import tabulate

from modules.chess_types import (
    BOARD_ENCODING_SHAPE,
    PIECE_ENCODING_SHAPE,
    PIECE_INDEX,
    PMF,
    Action,
    AgentLogEntry,
    BoardEncoding,
    GameLog,
    GameLogEntry,
    MoveLogEntry,
    MoveVector,
    Observation,
    PieceEncoding,
    Players,
    SetEncoding,
)

rng = np.random.default_rng()


def encode_pieces_slow(
    piece_map: dict[chess.Square, chess.Piece], color_to_move: chess.Color
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
            row, col, get_piece_index(piece.piece_type, Players(piece.color == color_to_move))
        ] = 1

    # return contructed board encoding
    return encoded_pieces


_rank_shifts_by_color = {
    chess.WHITE: 8 * np.arange(0, 8, 1, dtype=np.uint64)[:, np.newaxis],
    chess.BLACK: 8 * np.arange(7, -1, -1, dtype=np.uint64)[:, np.newaxis],
}


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
        dtype=np.uint64,
    )

    # this step shifts each byteboard out into each rank, and then truncates to a single byte (which
    # represents the file)
    shifted_bytes = (bitboards >> _rank_shifts_by_color[turn]).astype(np.uint8)

    # this step unpacks the remaining axis, using each byte
    return np.unpackbits(
        shifted_bytes,
        axis=0,
        bitorder="little",
    ).reshape((8, 8, 12))


def encode_board(board: chess.Board, encoding_array: BoardEncoding | None = None) -> BoardEncoding:
    """

    Generate a board encoding from the provided board.

    Parameters
    ----------
    board : chess.Board
        board to encode
    encoding_array : BoardEncoding | None
        optional output array to encode the board into, default None

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

    if board.is_repetition() or board.is_fifty_moves():
        encoded_board[:, :, 16] = 1

    if board.has_legal_en_passant():
        ep_square = cast("int", board.ep_square)
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
    encodings: SetEncoding = np.zeros((num_moves, 8, 8, 18), dtype=np.uint8)
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


def write_game(game_log: GameLog) -> None:
    """Write game to output (currently just a text file)."""
    game_headers = [f.name for f in fields(GameLogEntry)]
    agent_headers = ["agent_color", *[f.name for f in fields(AgentLogEntry)]]
    move_headers = [f.name for f in fields(MoveLogEntry)]

    game_data = [list(astuple(game_log.game))]
    agent_data = [[color, *astuple(game_log.agents[color])] for color in [chess.BLACK, chess.WHITE]]
    move_data = [list(astuple(move)) for move in game_log.moves]

    game_string = tabulate(game_data, headers=game_headers, tablefmt="grid")
    agent_string = tabulate(agent_data, headers=agent_headers, tablefmt="grid")
    move_string = tabulate(move_data, headers=move_headers, tablefmt="grid")

    path = Path("temp.txt")

    with path.open("w") as file:
        print("Game Data", file=file)
        print(game_string, file=file)
        print("\nAgent Data", file=file)
        print(agent_string, file=file)
        print("\nMove Data", file=file)
        print(move_string, file=file)
