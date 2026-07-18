"""Module that defines datacollectors to abstract data flow from actual objects."""

import time
from typing import cast

import chess

from modules.agent import AgentBase
from modules.chess_types import (
    ABORT_ACTION,
    PMF,
    Action,
    AgentDecision,
    AgentLogEntry,
    BoardStepResult,
    GameLog,
    GameLogEntry,
    MoveContext,
    MoveLogEntry,
    Outcome,
    SetEvaluation,
)
from modules.utils import calculate_policy_entropy


class Collector:
    """Class that can accept and store metadata from various parts of games for useage elsewhere."""

    def __init__(self) -> None:
        self._game = None
        self._moves: list[MoveLogEntry] = []
        self._active_move: MoveLogEntry | None = None
        self._agents: dict[chess.Color, AgentLogEntry | None] = {
            chess.WHITE: None,
            chess.BLACK: None,
        }

        self._move_start_time = 0
        self._move_stop_time = 0

    def reset(self) -> None:
        """Reset collector to a fresh status."""
        self.__init__()

    def select_agent(self, agent: AgentBase, color: chess.Color) -> None:
        """Select the agent playing the specific color."""
        if self._game is not None:
            msg = "Collector cannot select an agent when a game is currently active."
            raise RuntimeError(msg)

        if self._active_move is not None:
            msg = "Collector cannot select an agent when a move is active."
            raise RuntimeError(msg)

        if hasattr(agent, "strain") and hasattr(agent, "generation"):
            strain = cast("int", agent.strain)
            generation = cast("int", agent.generation)
        else:
            strain = None
            generation = None

        self._agents[color] = AgentLogEntry(
            agent_type=type(agent).__name__,
            strain=strain,
            generation=generation,
            timestamp=time.time_ns(),
        )

    def start_game(self) -> None:
        """Create a new game log entry."""
        if self._game is not None:
            msg = "Current game must be finished before starting another."
            raise RuntimeError(msg)

        if not self._is_agents_populated():
            msg = "Agents must be selected before starting a game."
            raise RuntimeError(msg)

        self._game = GameLogEntry(
            timestamp=time.time_ns(),
        )

    def finish_game(self, outcome: Outcome) -> GameLog:
        """

        Close current log collection and return the completed game log.

        Parameters
        ----------
        outcome : Outcome
            board outcome

        Returns
        -------
        GameLog
            log that contains all relevant metadata for the game.
        """
        if self._game is None:
            msg = "Cannot finish a non-existant game!"
            raise RuntimeError(msg)

        if self._active_move is not None:
            msg = "Cannot finish a game with an unresolved move."
            raise RuntimeError(msg)

        self._close_game_entry(outcome)

        agents = cast("dict[chess.Color, AgentLogEntry]", self._agents)

        log = GameLog(
            game=self._game,
            agents=agents,
            moves=tuple(self._moves),
        )

        self.reset()

        return log

    def start_move(self) -> None:
        """Start logging current move."""
        if self._game is None:
            msg = "Cannot start a new move on a non-existant game."
            raise RuntimeError(msg)
        if self._active_move is not None:
            msg = "Cannot start a new move when there is already an active move."
            raise RuntimeError(msg)

        self._active_move = MoveLogEntry()
        self._move_start_time = time.time_ns()

    def reset_move(self) -> None:
        """Clear the current move metadata (useful for aborts)."""
        self._active_move = None

    def record_move(
        self, context: MoveContext, decision: AgentDecision, result: BoardStepResult | None
    ) -> None:
        """

        Log the provided move metadata.

        Parameters
        ----------
        decision : AgentDecision
            decision made by agent
        result : BoardStepResult | None
            result when played on board, None if game was aborted before decision was played
        """
        if self._active_move is None:
            msg = "No active move."
            raise RuntimeError(msg)

        self._move_stop_time = time.time_ns()

        self._write_context(context)
        self._write_decision(decision)
        if result is not None:
            self._write_result(result)
        self._write_times()

        self._moves.append(self._active_move)

        self._active_move = None

    def _close_game_entry(self, outcome: Outcome) -> None:
        """Terminate the game for this collector."""
        if outcome.cause is None:
            msg = "Cannot close game entry without a cause of termination."
            raise RuntimeError(msg)

        game: GameLogEntry = cast("GameLogEntry", self._game)

        game_start_time: int = cast("int", game.timestamp)
        game_end_time = time.time_ns()

        game.dt = game_end_time - game_start_time

        game.result = outcome.status
        game.termination_type = outcome.cause

        game.ply_number = len(self._moves)

    def _is_agents_populated(self) -> bool:
        """

        Determine if collector has already obtained agents.

        Returns
        -------
        bool
        """
        return self._agents[chess.WHITE] is not None and self._agents[chess.BLACK] is not None

    def _write_context(self, context: MoveContext) -> None:
        """

        Populate the fields of the current move that pertain to the move context.

        Parameters
        ----------
        context : MoveContext
            move context metadata object
        """
        move: MoveLogEntry = cast("MoveLogEntry", self._active_move)
        move.side_to_move = context.side_to_move
        move.ply = context.ply

    def _write_decision(self, decision: AgentDecision) -> None:
        """

        Populate the fields of the current move that pertain to the agent action.

        Parameters
        ----------
        decision : AgentDecision
            agent decision metadata object
        """
        move: MoveLogEntry = cast("MoveLogEntry", self._active_move)

        action: Action = decision.action
        evals: SetEvaluation | None = decision.evals
        dist: PMF | None = decision.dist

        if evals is not None:
            move.position_eval_after_move = evals[action] if action != ABORT_ACTION else None
        if dist is not None:
            move.probability_of_choice = dist[action] if action != ABORT_ACTION else None
            move.policy_entropy = calculate_policy_entropy(dist)

    def _write_result(self, result: BoardStepResult) -> None:
        """

        Populate the fields of the current move that pertain to the board step result.

        Parameters
        ----------
        result : BoardStepResult
            step result metadata object
        """
        move: MoveLogEntry = cast("MoveLogEntry", self._active_move)

        move.capture_piece_type = result.capture_piece
        move.castle_type = result.castle_type
        move.is_check = result.is_check
        move.legal_move_count = result.legal_move_count
        move.piece_type = result.move_piece
        move.promotion = result.promotion
        move.uci = result.uci
        move.zobrist_after_move = result.pos_hash

    def _write_times(self) -> None:
        """Populate the fields of the current move that pertain to the start/stop timestamps."""
        move: MoveLogEntry = cast("MoveLogEntry", self._active_move)

        move.timestamp = self._move_start_time
        move.dt = self._move_stop_time - self._move_start_time
