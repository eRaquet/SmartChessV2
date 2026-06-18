"""File that contains naming and numbering conventions for the chess software."""

import chess

from modules.chess_types import Players

# piece type index in observation space (used because reflection not switches side)
piece_index = {
    (chess.PAWN, Players.SELF): 0,
    (chess.KNIGHT, Players.SELF): 1,
    (chess.BISHOP, Players.SELF): 2,
    (chess.ROOK, Players.SELF): 3,
    (chess.QUEEN, Players.SELF): 4,
    (chess.KING, Players.SELF): 5,
    (chess.KING, Players.OPPONENT): 6,
    (chess.QUEEN, Players.OPPONENT): 7,
    (chess.ROOK, Players.OPPONENT): 8,
    (chess.BISHOP, Players.OPPONENT): 9,
    (chess.KNIGHT, Players.OPPONENT): 10,
    (chess.PAWN, Players.OPPONENT): 11,
}

MIN_ROW_INDEX = 0
MAX_ROW_INDEX = 7
MIN_COL_INDEX = 0
MAX_COL_INDEX = 7
