"""File for containing all the type aliases and structures used in SmartChessV2."""

from enum import Enum, IntFlag, auto
from typing import (
    TypedDict,
)

import chess
import numpy as np
from numpy.typing import NDArray

# a piece encoding is the datatype that contains the information about piece positions
type PieceEncoding = NDArray[np.uint8]  # has a shape of (8, 8, 12)
PIECE_ENCODING_SHAPE = (8, 8, 12)

# a board encoding is the datatype that contains the information of a board that is given to a model
# it is formatted in the following way:
# AXES:
#   - Row
#   - Column
#   - Information Type
#       - 0:6 Self Pieces -- Pawn, Knight, Bishops, Rooks, Queen, King
#       - 6:12 Opponent Pieces -- King, Queen, Rooks, Bishops, Knights, Pawn
#       - 12:16 Castling Rights (all Rows and Columns are 1 if it has rights) -- Bottom Left, Bottom
#         Right, Top Right, Top Left
#       - 16 Draw Conditions (all Rows and Columns are 1 if the position could be a draw)
#       - 17 Aun Passant (squares where an Aun Passant capture could happen)
# NOTE: board encodings always represent the board from the perspect of the player whose turn it is.
type BoardEncoding = NDArray[np.uint8]  # has a shape of (8, 8, 18)
BOARD_ENCODING_SHAPE = (8, 8, 18)

# an set of some number of board encodings, to be kept together (i.e. game trajectories or
# observations)
type SetEncoding = NDArray[np.uint8]  # has a shape of (n, 8, 8, 18)
type Observation = SetEncoding
type Trajectory = SetEncoding
# vector of all possible moves for a given state
# the order must be static, as as Action is defined as an index into the move vector
type MoveVector = list[chess.Move]

# alias for gym.Env terminology
# corresponds to the index of the chosen move
# -1 means a resignation
type Action = int
RESIGN: Action = -1

type Evaluation = float
type SetEvaluation = NDArray[np.double]


class BoardOutcome(IntFlag):
    """Enumeration of various states of outcome for a chess board."""

    UNDECIDED = auto()
    WHITE = auto()
    BLACK = auto()
    DRAW = auto()

    TERMINATED = WHITE | BLACK | DRAW
    WON = WHITE | BLACK


# dictionary structure to specify the info read off from the board at each position
class BoardInfo(TypedDict):
    """

    Class that defines board info.

    Currently empty.

    """


class Players(Enum):
    """Enum for player types."""

    SELF = True
    OPPONENT = False


# piece type index in observation space (used because reflection not switches side)
PIECE_INDEX = {
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

# constants for move identification
MIN_ROW_INDEX = 0
MAX_ROW_INDEX = 7
MIN_COL_INDEX = 0
MAX_COL_INDEX = 7
ROOK_CASTLE_FILE_KINGSIDE = 5
ROOK_CASTLE_FILE_QUEENSIDE = 3
