"""Module that defines datacollectors to abstract data flow from actual objects."""

import chess
import numpy as np

from modules.chess_types import (
    PMF,
    Action,
    AgentLogEntry,
    GameLogEntry,
    LogCastleType,
    MoveLogEntry,
    SetEvaluation,
)
from modules.utils import get_new_id


class LogCollector:
    """Class that can accept and store metadata from various parts of games for useage elsewhere."""

    def __init__(self) -> None:
        self._game = GameLogEntry()
        self._moves: list[MoveLogEntry] = []
        self._current_move: MoveLogEntry | None = None
        self._agents = {
            chess.WHITE: AgentLogEntry(),
            chess.BLACK: AgentLogEntry(),
        }

    def new_move(self) -> None:
        """Create a new move log object to insert to."""
        move_id = get_new_id()
        ply = len(self._moves) + 1
        side_to_move = chess.WHITE if ply % 2 == 1 else chess.BLACK
        self._current_move = MoveLogEntry(
            id=move_id,
            game_id=self._game.id,
            ply=ply,
            agent_id=self._agents[side_to_move].id,
            side_to_move=side_to_move,
        )
        self._moves.append(self._current_move)

    def insert_chess_move(self, chess_move: chess.Move) -> None:
        """

        Insert the information from a `chess.Move` object into the collector.

        Parameters
        ----------
        chess_move : chess.Move
            chess move object to log info from
        """
        self._current_move.uci = chess_move.uci()
        self._current_move.promotion = chess_move.promotion

    def insert_model_action(
        self, evaluation: SetEvaluation, distribution: PMF, action: Action
    ) -> None:
        """

        Insert the move log info that corresponds to a model evaluation.

        Parameters
        ----------
        evaluation : SetEvaluation
            raw evaluation
        distribution : PMF
            actual distribution that the agent picks from
        action : Action
            chosen action
        """
        self._current_move.position_eval_after_move = evaluation[action]
        self._current_move.policy_entropy = -np.sum(distribution * np.log2(distribution))
        self._current_move.probability_of_choice = distribution[action]

    def insert_piece_type(self, piece_type: chess.PieceType) -> None:
        """

        Insert a specific piece type to the current move.

        Parameters
        ----------
        piece_type : chess.PieceType
        """
        self._current_move.piece_type = piece_type

    def insert_capture_piece_type(self, piece_type: chess.PieceType) -> None:
        """

        Insert a specific capture piece type to the current move.

        Parameters
        ----------
        piece_type : chess.PieceType
        """
        self._current_move.capture_piece_type = piece_type

    def insert_is_en_passant(self, *, is_en_passant: bool) -> None:
        """

        Insert the is en passant flag to the current move.

        Parameters
        ----------
        is_en_passant : bool
        """
        self._current_move.is_en_passant = is_en_passant

    def insert_is_check(self, *, is_check: bool) -> None:
        """

        Insert the is check flag to the current move.

        Parameters
        ----------
        is_check : bool
        """
        self._current_move.is_check = is_check

    def insert_castle_type(self, castle_type: LogCastleType) -> None:
        """

        Insert the castle type into the current move.

        Parameters
        ----------
        castle_type : LogCastleType
        """
        self._current_move.castle_type = castle_type

    def insert_legal_move_count(self, legal_move_count: int) -> None:
        """

        Insert the number of legal moves from which the current move was chosen.

        Parameters
        ----------
        legal_move_count : int
        """
        self._current_move.legal_move_count = legal_move_count

    def insert_zobrist_after_move(self, zobrist_hash: int) -> None:
        """

        Insert the zobrist hash of the position after the current move.

        Parameters
        ----------
        zobrist_hash : int
        """
        self._current_move.zobrist_after_move = zobrist_hash
