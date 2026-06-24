"""Module that defines datacollectors to abstract data flow from actual objects."""

import chess
import numpy as np
from chess.polyglot import zobrist_hash

from modules.chess_types import (
    PMF,
    Action,
    AgentLogEntry,
    GameLogEntry,
    LogCastleType,
    MoveLogEntry,
    MoveVector,
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

    def insert_board_action_pre(
        self, move: chess.Move, move_vector: MoveVector, board: chess.Board
    ) -> None:
        """

        Insert the move log info that corresponds to a board action before move actually happens.

        Parameters
        ----------
        move : chess.Move
            chosen chess move
        move_vector : MoveVector
            list of possible moves
        board : chess.Board
            chess board the move will be played on
        """
        self._current_move.uci = move.uci()
        self._current_move.promotion = move.promotion
        self._current_move.piece_type = board.piece_at(move.from_square)
        self._current_move.capture_piece_type = board.piece_at(move.to_square)
        self._current_move.castle_type = (
            LogCastleType.KINGSIDE
            if board.is_kingside_castling(move)
            else LogCastleType.QUEENSIDE
            if board.is_queenside_castling(move)
            else None
        )
        self._current_move.legal_move_count = len(move_vector)

    def insert_board_action_post(self, board: chess.Board) -> None:
        """

        Insert the move log info that corresponds to a board action after move is played.

        Parameters
        ----------
        board : chess.Board
            chess board the move will be played on
        """
        self._current_move.is_check = board.is_check()
        self._current_move.zobrist_after_move = zobrist_hash(board)
