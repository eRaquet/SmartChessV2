"""Module to define the board environment for training and playing."""

from typing import cast, override

import chess
from chess.polyglot import zobrist_hash

from modules.chess_types import (
    ABORT_ACTION,
    Action,
    BoardStatus,
    BoardStepResult,
    LogCastleType,
    MoveVector,
    Observation,
    Outcome,
    TerminationType,
)
from modules.display import Display
from modules.utils import (
    generate_observation,
)


class Board:
    """Chess board environment with generation of valid moves and creation of observations."""

    def __init__(self) -> None:
        """Initiate a board object."""
        self._board = chess.Board()
        self._moves: list[chess.Move] = list(self._board.legal_moves)

        self._outcome = Outcome()

        # allocate the observation space
        self._observation: Observation = cast("Observation", None)
        self._observed = False

    def reset(self) -> None:
        """Reset board position."""
        self._board.reset()
        self._moves: list[chess.Move] = list(self._board.legal_moves)
        self._observed = False
        self._render()

    def step(self, action: Action) -> BoardStepResult | None:
        """

        Step the board through one move, encoded by `action`.

        Parameters
        ----------
        action : Action
            Action performed on the board (chess move)

        Returns
        -------
        BoardStepResult | None
            result from step, or None if step not performed
        """
        if self._outcome.status in BoardStatus.TERMINATED:
            msg = "Board is in terminal state, and cannot be stepped."
            raise RuntimeError(msg)

        if action == ABORT_ACTION:
            self._outcome.status = BoardStatus.UNDECIDED
            self._outcome.cause = TerminationType.ABORT
            return None

        result = self.update_state(action)

        self._check_end_conditions()

        self._render()

        return result

    def _observe(self) -> None:
        """Make the environment reflect the board state and generate an observation."""
        if self._outcome.status in BoardStatus.TERMINATED:
            msg = "Tried to observe board after status was terminated."
            raise RuntimeError(msg)

        self._observation = generate_observation(self._board, self._moves)
        self._observed = True

    def update_state(self, action: Action) -> BoardStepResult:
        """

        Update the state of the environment without generating an observation or rendering.

        Parameters
        ----------
        action : Action
            played action, represented by the index of move to play

        Returns
        -------
        BoardStepResult
            the result from the updating the board
        """
        move = self._moves[action]

        result = self._capture_pre(move)

        self._board.push(move)

        result = self._capture_post(result)

        self._moves = list(self._board.legal_moves)

        self._observed = False

        return result

    def _render(self) -> None:
        """Render method for board, empty for base class."""
        return

    @property
    def moves(self) -> MoveVector:
        """

        List of possible moves from the current board state.

        Returns
        -------
        MoveVector
            list of moves
        """
        return self._moves

    @property
    def terminated(self) -> bool:
        """

        Is the board at a terminal state.

        Returns
        -------
        bool
        """
        return self._outcome.status in BoardStatus.TERMINATED

    @property
    def winner(self) -> chess.Color | None:
        """

        The winner in the current board state, None if not applicable.

        Returns
        -------
        chess.Color | None
            The winning color, or None if no winner is determined (draw or unfinished game)
        """
        if self._outcome.status in BoardStatus.WON:
            return chess.WHITE if self._outcome.status == BoardStatus.WHITE else chess.BLACK
        return None

    @property
    def outcome(self) -> Outcome:
        """

        The outcome of the board.

        Returns
        -------
        Outcome
        """
        return self._outcome

    @property
    def observation(self) -> Observation:
        """

        Observation of current board state.

        Returns
        -------
        Observation
            shape (n, 8, 8, 18), where n is the number of valid moves
        """
        if not self._observed:
            self._observe()
        return self._observation

    @property
    def turn(self) -> chess.Color:
        """

        Player whose turn it is.

        Returns
        -------
        chess.Color
        """
        return self._board.turn

    @property
    def full_move_count(self) -> int:
        """

        Total number of full moves in the game so far.

        Half moves count the number of individual moves from players, whereas full moves are the
        total number of pairs of moves from both players.

        Returns
        -------
        int
        """
        return self._board.fullmove_number

    @property
    def half_move_count(self) -> int:
        """

        Total number of half moves in the game so far.

        Half moves count the number of individual moves from players, whereas full moves are the
        total number of pairs of moves from both players.

        Returns
        -------
        int
        """
        return self._board.ply()

    def _capture_pre(self, move: chess.Move) -> BoardStepResult:
        moved = cast("chess.Piece", self._board.piece_at(move.from_square))
        move_piece = moved.piece_type
        captured = self._board.piece_at(move.to_square)
        capture_piece = (
            captured.piece_type
            if captured is not None
            else chess.PAWN
            if self._board.is_en_passant(move)
            else None
        )
        castle_type = (
            LogCastleType.KINGSIDE
            if self._board.is_kingside_castling(move)
            else LogCastleType.QUEENSIDE
            if self._board.is_queenside_castling(move)
            else None
        )

        return BoardStepResult(
            uci=move.uci(),
            promotion=move.promotion,
            move_piece=move_piece,
            capture_piece=capture_piece,
            castle_type=castle_type,
            legal_move_count=len(self._moves),
            is_check=False,  # dummy value to be filled in after move
            pos_hash=-1,  # dummy value to be filled in after move
        )

    def _capture_post(self, result: BoardStepResult) -> BoardStepResult:
        result.is_check = self._board.is_check()
        result.pos_hash = zobrist_hash(self._board)
        return result

    def _check_end_conditions(self) -> None:
        if self._board.is_checkmate():
            self._outcome.status = (
                BoardStatus.BLACK if self._board.turn == chess.WHITE else BoardStatus.WHITE
            )
            self._outcome.cause = TerminationType.CHECKMATE
        elif self._board.is_repetition():
            self._outcome.status = BoardStatus.DRAW
            self._outcome.cause = TerminationType.REPETITION
        elif self._board.is_fifty_moves():
            self._outcome.status = BoardStatus.DRAW
            self._outcome.cause = TerminationType.FIFTY_MOVES
        elif self._board.is_insufficient_material():
            self._outcome.status = BoardStatus.DRAW
            self._outcome.cause = TerminationType.INSUFFICIENT_MATERIAL
        elif self._board.is_stalemate():
            self._outcome.status = BoardStatus.DRAW
            self._outcome.cause = TerminationType.STALEMATE


class ASCIIBoard(Board):
    """Board with simple ASCII visualization."""

    @override
    def _render(self) -> None:
        """Render board as ASCII."""
        print("-" * 15)
        print(self._board)
        print("-" * 15)


class GUIBoard(Board):
    """Board with full pygame gui."""

    def __init__(self) -> None:
        super().__init__()

        self._display = Display()

    @override
    def _render(self) -> None:
        """Render board display."""
        self._display.display_board(self._board)

    def get_user_input(self) -> Action | None:
        """

        Check if the user has given GUI input, and return the move if possible.

        Returns
        -------
        Action | None
            Action selected by the user, or None if no action is yet selected
        """
        return self._display.get_user_input(self._board, self._moves)
