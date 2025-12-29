"""File for containing all the type aliases and structures used in SmartChessV2."""

from enum import Enum
from typing import (
    TypedDict,
)

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
#       - 12:16 Castling Rights (all Rows and Columns are 1 if it has rights) -- Bottom Left, Bottom Right, Top Right, Top Left
#       - 16 Draw Conditions (all Rows and Columns are 1 if the position could be a draw)
#       - 17 Aun Passant (squares where an Aun Passant capture could happen)
type BoardEncoding = NDArray[np.uint8]  # has a shape of (8, 8, 18)
BOARD_ENCODING_SHAPE = (8, 8, 18)

# an set of some number of board encodings, to be kept together (i.e. game trajectories or observations)
type SetEncoding = NDArray[np.uint8]  # has a shape of (n, 8, 8, 18)
type IsOver = bool
type MoveReward = float

# alias for gym.Env terminology
# corresponds to the index of the chosen move
type Action = int


# Enum for display modes
class DisplayMode(Enum):
    """Enumeration for different board display modes."""

    GUI = 1
    ASCII = 2
    NONE = 3


# dictionary structure to encode board observations
class Observation(TypedDict):
    """

    Encoding dictionary for an observation of the board.

    This works by noting that there can only be at most 218 valid moves for a certain player.

    """

    # the encoding is from the serspective of the player to move
    # last axis of encoding is delegated in the following
    #   - 0:6 players pieces
    encoding: SetEncoding  # shape: (218, 8, 8, 18)

    num_moves: int


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
