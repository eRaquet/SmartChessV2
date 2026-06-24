"""Module that defines datacollectors to abstract data flow from actual objects."""

import time
from dataclasses import fields
from pathlib import Path

import chess
import numpy as np
from chess.polyglot import zobrist_hash
from tabulate import tabulate

from modules.agent import AgentBase, StandardAgent
from modules.chess_types import (
    PMF,
    Action,
    AgentLogEntry,
    GameLogEntry,
    LogCastleType,
    LogResult,
    LogTerminationType,
    MoveLogEntry,
    MoveVector,
    SetEvaluation,
)
from modules.utils import get_new_id


class LogCollector:
    """Class that can accept and store metadata from various parts of games for useage elsewhere."""

    def __init__(self) -> None:
        self._game = None
        self._moves: list[MoveLogEntry] = []
        self._current_move: MoveLogEntry | None = None
        self._agents = {
            chess.WHITE: None,
            chess.BLACK: None,
        }

    def select_agent(self, agent: AgentBase, color: chess.Color) -> None:
        """Select the agent playing the specific color."""
        self._agents[color] = AgentLogEntry(
            id=get_new_id(),
            agent_type=type(agent).__name__,
            strain=agent.strain if isinstance(agent, StandardAgent) else None,
            generation=agent.generation if isinstance(agent, StandardAgent) else None,
            timestamp=time.time_ns(),
        )

    def new_game(self) -> None:
        """Create a new game log entry."""
        self._game = GameLogEntry(
            id=get_new_id(),
            white_agent_id=self._agents[chess.WHITE].id,
            black_agent_id=self._agents[chess.BLACK].id,
            timestamp=time.time_ns(),
        )

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
            timestamp=time.time_ns(),
        )
        self._moves.append(self._current_move)

    def terminate_game(self, board: chess.Board) -> None:
        """Terminate the game for this collector."""
        outcome = board.outcome()
        if outcome:
            if outcome.winner is chess.WHITE:
                self._game.result = LogResult.WHITE
            elif outcome.winner is chess.BLACK:
                self._game.result = LogResult.BLACK
            else:
                self._game.result = LogResult.DRAW

            if outcome.termination is chess.Termination.CHECKMATE:
                self._game.termination_type = LogTerminationType.CHECKMATE
            elif outcome.termination is chess.Termination.STALEMATE:
                self._game.termination_type = LogTerminationType.STALEMATE
            elif outcome.termination in (
                chess.Termination.FIVEFOLD_REPETITION,
                chess.Termination.THREEFOLD_REPETITION,
            ):
                self._game.termination_type = LogTerminationType.REPETITION
            elif outcome.termination in (
                chess.Termination.FIFTY_MOVES,
                chess.Termination.SEVENTYFIVE_MOVES,
            ):
                self._game.termination_type = LogTerminationType.FIFTY_MOVES
            elif outcome.termination is chess.Termination.INSUFFICIENT_MATERIAL:
                self._game.termination_type = LogTerminationType.INSUFFICIENT_MATERIAL
            else:
                self._game.termination_type = LogTerminationType.ABORT

        else:
            self._game.result = LogResult.UNRESOLVED
            self._game.termination_type = LogTerminationType.ABORT

        self._game.ply_number = len(self._moves)

    def write_game(self) -> None:
        """Write game to output (currently just a text file)."""
        game_headers = [f.name for f in fields(GameLogEntry)]
        agent_headers = [f.move for f in fields(AgentLogEntry)]
        move_headers = [f.name for f in fields(MoveLogEntry)]

        game_data = [list(self._game.astuple())]
        agent_data = [list(agent.astuple()) for agent in self._agents.values()]
        move_data = [list(move.astuple()) for move in self._moves]

        game_string = tabulate(game_data, headers=game_headers, tablefmt="grid")
        agent_string = tabulate(agent_data, headers=agent_headers, tablefmt="grid")
        move_string = tabulate(move_data, headers=move_headers, tablefmt="grid")

        with Path.open("temp.txt", "w") as file:
            print(game_string, file=file)
            print(agent_string, file=file)
            print(move_string, file=file)

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
