"""File for containing all the type aliases and structures used in SmartChessV2."""

from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from typing import TypedDict

import chess
from numpy import float64, uint8
from numpy.typing import NDArray

# a piece encoding is the datatype that contains the information about piece positions
type PieceEncoding = NDArray[uint8]  # has a shape of (8, 8, 12)
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
type BoardEncoding = NDArray[uint8]  # has a shape of (8, 8, 18)
BOARD_ENCODING_SHAPE = (8, 8, 18)

# alias for gym.Env terminology
# corresponds to the index of the chosen move
# -1 means a resignation
type Action = int
ABORT_ACTION: Action = -1

# an set of some number of board encodings, to be kept together (i.e. game trajectories or
# observations)
type SetEncoding = NDArray[uint8]  # has a shape of (n, 8, 8, 18)


@dataclass(slots=True)
class Observation:
    """Dataclass that stores an observation of a board."""

    encodings: SetEncoding
    checkmate_action: Action | None


# vector of all possible moves for a given state
# the order must be static, as as Action is defined as an index into the move vector
type MoveVector = list[chess.Move]

type Evaluation = float
type SetEvaluation = NDArray[float64]
type PMF = NDArray[float64]


class BoardOutcome(IntFlag):
    """Enumeration of various states of outcome for a chess board."""

    UNDECIDED = auto()
    WHITE = auto()
    BLACK = auto()
    DRAW = auto()
    ABORT = auto()

    TERMINATED = WHITE | BLACK | DRAW | ABORT
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

### Game Storage Table Schemes


## Data Enums


class LogResult(IntFlag):
    """Enum for mapping integer values to game results for the database."""

    WHITE = 0
    BLACK = 1
    DRAW = 2
    UNRESOLVED = 3


class LogTerminationType(IntFlag):
    """Enum for mapping integer values to causes for game termination for the database."""

    CHECKMATE = 0
    STALEMATE = 1
    REPETITION = 2
    FIFTY_MOVES = 3
    INSUFFICIENT_MATERIAL = 4
    ABORT = 5


class LogCastleType(IntFlag):
    """Enum for mapping integer values to castling sides for the database."""

    KINGSIDE = 0
    QUEENSIDE = 1


## Game Table

# - id {INTEGER}
# - white agent id {INTEGER}
# - black agent id {INTEGER}
# - result (white win, black win, draw, unresolved) {INTEGER}
# - termination type
#   (checkmate, stalemate, repetition, fifty moves, insufficient material, abort) {INTEGER}
# - ply number {INTEGER}
# - starting fen (NULL if standard starting fen) {TEXT}
# - timestamp in nanoseconds since last Unix epoch {INTEGER}
# - dt in nanoseconds for the duration of the game {INTEGER}


## Agent Table

# - id {INTEGER}
# - agent type (name of class, "StandardAgent" for example) {TEXT}
# - strain (NULL if no strain) {INTEGER}
# - generation (NULL if no generation) {INTEGER}
# - timestamp in nanoseconds since last Unix epoch {INTEGER}


## Move Table

# - id {INTEGER} populated by collector
# - game id {INTEGER} populated by collector
# - agent id {INTEGER} populated by collector
# - ply {INTEGER} populated by collector
# - uci {TEXT} populated by board
# - promotion (NULL if no promotion, using chess.Piece) {INTEGER} populated by board
# - side to move (using chess.Color) {INTEGER} populated by collector
# - piece type (using chess.Piece) {INTEGER} populated by board
# - position evaluation (for current player) after move (NULL if no model) {REAL} populated by agent
# - policy entropy {REAL} populated by agent
# - probability of choice for selected move (NULL if random agent or human agent) {REAL} populated
#   by agent
# - capture piece type (NULL if no capture, using chess.Piece) {INTEGER} populated by board
# - is check {INTEGER} populated by board
# - castle type (NULL if no castle) {INTEGER} populated by board
# - zobrist hash after move {INTEGER} populated by board
# - legal move count {INTEGER} populated by board
# - timestamp in nanoseconds since last Unix epoch {INTEGER}
# - dt in nanoseconds between the beginning and the end of a move {INTEGER}


## Log Entry Dataclasses


@dataclass(slots=True, kw_only=True)
class GameLogEntry:
    """Data class for storing metadata to go into the game table of the game database."""

    result: LogResult | None = None
    termination_type: LogTerminationType | None = None
    ply_number: int | None = None
    starting_fen: str | None = None
    timestamp: int | None = None
    dt: int | None = None


@dataclass(slots=True, kw_only=True)
class AgentLogEntry:
    """Data class for storing metadata to go into the agent table of the game database."""

    agent_type: str | None = None
    strain: int | None = None
    generation: int | None = None
    timestamp: int | None = None


@dataclass(slots=True, kw_only=True)
class MoveLogEntry:
    """Data class for storing metadata to go into the move table of the game database."""

    ply: int | None = None
    uci: str | None = None
    promotion: chess.PieceType | None = None
    side_to_move: chess.Color | None = None
    piece_type: chess.PieceType | None = None
    position_eval_after_move: float | None = None
    policy_entropy: float | None = None
    probability_of_choice: float | None = None
    capture_piece_type: chess.PieceType | None = None
    is_check: bool | None = None
    castle_type: LogCastleType | None = None
    zobrist_after_move: int | None = None
    legal_move_count: int | None = None
    timestamp: int | None = None
    dt: int | None = None


### Collector Data Classes


@dataclass(slots=True, kw_only=True)
class MoveContext:
    """Data class for storing the move context information."""

    side_to_move: chess.Color
    ply: int


@dataclass(slots=True, kw_only=True)
class AgentDecision:
    """Data class for capturing agent action metadata for collector to access."""

    evals: SetEvaluation | None
    dist: PMF | None
    action: Action


@dataclass(slots=True, kw_only=True)
class BoardStepResult:
    """Data class for capturing the results from a give agent action."""

    uci: str
    promotion: chess.PieceType | None
    move_piece: chess.PieceType
    capture_piece: chess.PieceType | None
    castle_type: LogCastleType | None
    legal_move_count: int
    is_check: bool
    pos_hash: int


@dataclass(slots=True, kw_only=True)
class GameLog:
    """Data class that stores a completed, immutable game log."""

    game: GameLogEntry
    agents: dict[chess.Color, AgentLogEntry]
    moves: tuple[MoveLogEntry, ...]
