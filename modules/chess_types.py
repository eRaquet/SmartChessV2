"""File for containing all the type aliases and structures used in SmartChessV2."""

from enum import Enum
from typing import (
    TypedDict,
)

import numpy as np
from numpy.typing import NDArray

type BoardEncoding = NDArray[np.uint8]  # has a shape of (8, 8, 12), 0 is no piece, 1 is a piece
type SetEncoding = NDArray[np.uint8]  # has a shape of (n, 8, 8, 12), 0 is no piece, 1 is a piece
type IsOver = bool
type MoveReward = float

# alias for gym.Env terminology
# corresponds to the index of the chosen move
type Action = int


# dictionary structure to encode board observations
class Observation(TypedDict):
    """

    Encoding dictionary for an observation of the board.

    This works by noting that there can only be at most 218 valid moves for a certain player.

    """

    # the encoding is from the serspective of the player to move (i.e., the first 6 pieces always belong to self)
    encoding: np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]]  # shape: (218, 8, 8, 12)

    # castling rights format
    # 1: bottom left castling rights
    # 2: bottom right castling rights
    # 3: top left castling rights
    # 4: top left castling rights
    castling_rights: np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # shape: (218, 4)

    # draw check
    is_draw: np.ndarray[tuple[int], np.dtype[np.uint8]]  # shape: (218)

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
