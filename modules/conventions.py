"""File that contains naming and numbering conventions for the chess software."""

from enum import IntFlag

import chess

from modules.chess_types import Players

# board array representation:
#    array of bools with shape (8, 8, 12)
#    with description of (rank, file, piece type)

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


### Game Storage Table Schemes


## game table


# - id {INTEGER}
# - white agent id {INTEGER}
# - black agent id {INTEGER}
# - result (white win, black win, draw) {INTEGER}
class DataResult(IntFlag):
    """Enum for mapping integer values to game results for the database."""

    WHITE = 0
    BLACK = 1
    DRAW = 2


# - termination type
#   (checkmate, stalemate, repetition, fifty moves, insufficient material, abort) {INTEGER}
class DataTerminationType(IntFlag):
    """Enum for mapping integer values to causes for game termination for the database."""

    CHECKMATE = 0
    STALEMATE = 1
    REPETITION = 2
    FIFTY_MOVES = 3
    INSUFFICIENT_MATERIAL = 4
    ABORT = 5


# - ply number {INTEGER}

# - starting fen (NULL if standard starting fen) {TEXT}
# - average agent entropy white (NULL if not model based) {REAL}
# - average agent entropy black (NULL if not model based) {REAL}

# - timestamp {INTEGER}


## agent table

# - id {INTEGER}
# - agent type (name of class, "StandardAgent" for example) {TEXT}
# - strain (NULL if no strain) {INTEGER}
# - generation (NULL if no generation) {INTEGER}
# - timestamp (NULL if not a model-based agent) {INTEGER}


## move table

# - id {INTEGER}
# - game id {INTEGER}
# - agent id {INTEGER}
# - ply {INTEGER}

# - uci {TEXT}
# - promotion (NULL if no promotion, using chess.Piece) {INTEGER}
# - side to move (using chess.Color) {INTEGER}
# - piece type (using chess.Piece) {INTEGER}

# - position evaluation after move (NULL if no model) {REAL}
# - policy entropy {REAL}
# - probability of choice for selected move (NULL if random agent or human agent) {REAL}


# - capture piece type (NULL if no capture, using chess.Piece) {INTEGER}
# - is en passant {INTEGER}
# - is check {INTEGER}
# - castle type (NULL if no castle) {INTEGER}
class DataCastleType(IntFlag):
    """Enum for mapping integer values to castling sides for the database."""

    KINGSIDE = 0
    QUEENSIDE = 1


# - position zobrist hash {INTEGER}
# - legal move count {INTEGER}
