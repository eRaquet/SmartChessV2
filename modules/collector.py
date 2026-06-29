"""Module that defines datacollectors to abstract data flow from actual objects."""

import time
from dataclasses import astuple, fields
from pathlib import Path

import chess
from tabulate import tabulate

from modules.agent import AgentBase
from modules.board import Board
from modules.chess_types import (
    AgentActionSnapshot,
    AgentLogEntry,
    BoardStepSnapshotPost,
    BoardStepSnapshotPre,
    GameLog,
    GameLogEntry,
    LogResult,
    LogTerminationType,
    MoveLogEntry,
)
from modules.utils import calculate_policy_entropy, get_new_id


class Collector:
    """Class that can accept and store metadata from various parts of games for useage elsewhere."""

    def __init__(self) -> None:
        self._game = None
        self._moves: list[MoveLogEntry] = []
        self._active_move = False
        self._agents = {
            chess.WHITE: None,
            chess.BLACK: None,
        }

        # temporary move data storage
        self._clear_metadata()

    def reset(self) -> None:
        """Reset collector to a fresh status."""
        self.__init__()

    def select_agent(self, agent: AgentBase, color: chess.Color) -> None:
        """Select the agent playing the specific color."""
        if not isinstance(agent, AgentBase):
            msg = "Provided agent has incorrect type."
            raise TypeError(msg)

        try:
            strain = agent.strain
            generation = agent.generation
        except AttributeError:
            strain = None
            generation = None

        self._agents[color] = AgentLogEntry(
            id=get_new_id(),
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

        self._game = GameLogEntry(
            id=get_new_id(),
            white_agent_id=self._agents[chess.WHITE].id,
            black_agent_id=self._agents[chess.BLACK].id,
            timestamp=time.time_ns(),
        )

    def finish_game(self, outcome: chess.Outcome) -> GameLog:
        """

        Close current log collection and return the completed game log.

        Returns
        -------
        GameLog
            Log that contains all relevant metadata for the game.
        """
        self._close_game_entry(outcome)

        log = GameLog(
            game=self._game,
            agents=tuple(self._agents[k] for k in sorted(self._agents.keys())),
            moves=tuple(self._moves),
        )

        self.reset()

        return log

    def start_move(self) -> None:
        """Start a new move."""
        if self._active_move:
            msg = "Cannot start a new move when there is already an active move."
            raise RuntimeError(msg)

        self._active_move = True

        self._move_start_time = time.time_ns()

    def finish_move(self) -> None:
        """Create  completed move into the move list."""
        self._move_stop_time = time.time_ns()

        if not self._is_move_populated():
            msg = "Current move is not fully populated."
            raise RuntimeError(msg)

        self._create_current_move_entry()

        self._moves.append(self._current_move)

        self._clear_metadata()

        self._active_move = False

    def get_board_action(self, board: Board) -> None:
        """

        Get metadata from provided board for the current move.

        Parameters
        ----------
        board : Board
        """
        if not self._active_move:
            msg = "No active move."
            raise RuntimeError(msg)

        self._board_action_pre_snapshot = board.snapshot_pre
        self._board_action_post_snapshot = board.snapshot_post

        # clear board snapshot for efficiency and better error catching
        board.snapshot_pre = None
        board.snapshot_post = None

    def get_agent_action(self, agent: AgentBase) -> None:
        """

        Get metadata from provided agent for the current move.

        Parameters
        ----------
        agent : AgentBase
        """
        if not self._active_move:
            msg = "No active move."
            raise RuntimeError(msg)

        self._agent_action_snapshot = agent.snapshot

        # clear agent snapshot for efficiency and better error catching
        agent.snapshot = None

    def _close_game_entry(self, outcome: chess.Outcome) -> None:
        """Terminate the game for this collector."""
        self._game.dt = time.time_ns() - self._game.timestamp

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

    def write_game(self, game_log: GameLog) -> None:
        """Write game to output (currently just a text file)."""
        game_headers = [f.name for f in fields(GameLogEntry)]
        agent_headers = [f.name for f in fields(AgentLogEntry)]
        move_headers = [f.name for f in fields(MoveLogEntry)]

        game_data = [list(astuple(game_log.game))]
        agent_data = [list(astuple(agent)) for agent in game_log.agents]
        move_data = [list(astuple(move)) for move in game_log.moves]

        game_string = tabulate(game_data, headers=game_headers, tablefmt="grid")
        agent_string = tabulate(agent_data, headers=agent_headers, tablefmt="grid")
        move_string = tabulate(move_data, headers=move_headers, tablefmt="grid")

        with Path.open("temp.txt", "w") as file:
            print("Game Data", file=file)
            print(game_string, file=file)
            print("\nAgent Data", file=file)
            print(agent_string, file=file)
            print("\nMove Data", file=file)
            print(move_string, file=file)

    def _is_agents_populated(self) -> bool:
        """

        Determine if collector has already obtained agents.

        Returns
        -------
        bool
        """
        return self._agents[chess.WHITE] and self._agents[chess.BLACK]

    def _is_move_populated(self) -> bool:
        """

        Determine if the collector has obtained the necessary move metadata.

        Returns
        -------
        bool
        """
        return (
            self._agent_action_snapshot is not None
            and self._board_action_pre_snapshot is not None
            and self._board_action_post_snapshot is not None
            and self._move_start_time is not None
            and self._move_stop_time is not None
        )

    def _create_current_move_entry(self) -> None:
        """Create the current move entry from the collected metadata."""
        move_id = get_new_id()
        ply = len(self._moves) + 1
        side_to_move = chess.WHITE if ply % 2 == 1 else chess.BLACK
        action = self._agent_action_snapshot.action
        evals = self._agent_action_snapshot.evals
        dist = self._agent_action_snapshot.dist
        capture_piece = self._board_action_pre_snapshot.capture_piece
        self._current_move = MoveLogEntry(
            id=move_id,
            game_id=self._game.id,
            agent_id=self._agents[side_to_move].id,
            ply=ply,
            uci=self._board_action_pre_snapshot.move.uci(),
            promotion=self._board_action_pre_snapshot.move.promotion,
            side_to_move=side_to_move,
            piece_type=self._board_action_pre_snapshot.move_piece.piece_type,
            position_eval_after_move=evals[action] if evals else None,
            policy_entropy=calculate_policy_entropy(self._agent_action_snapshot.dist),
            probability_of_choice=dist[action] if dist else None,
            capture_piece_type=capture_piece.piece_type if capture_piece else None,
            is_check=self._board_action_post_snapshot.is_check,
            castle_type=self._board_action_pre_snapshot.castle_type,
            zobrist_after_move=self._board_action_post_snapshot.pos_hash,
            legal_move_count=self._board_action_pre_snapshot.num_moves,
            timestamp=self._move_start_time,
            dt=self._move_stop_time - self._move_start_time,
        )

    def _clear_metadata(self) -> None:
        """Clear the current move metadata fields."""
        self._agent_action_snapshot: AgentActionSnapshot | None = None
        self._board_action_pre_snapshot: BoardStepSnapshotPre | None = None
        self._board_action_post_snapshot: BoardStepSnapshotPost | None = None
        self._move_start_time: int | None = None
        self._move_stop_time: int | None = None
