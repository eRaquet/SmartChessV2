"""Module to define the board environment for training and playing."""

from collections import Counter

import chess
import numpy as np

from modules.chess_types import (
    RESIGN,
    Action,
    BoardOutcome,
    Observation,
    Trajectory,
)
from modules.display import Display
from modules.tools import (
    encode_board,
    generate_board_encodings_from_moves,
)


class Board:
    """Chess board environment with generation of valid moves and creation of observations."""

    def __init__(self) -> None:
        """Initiate a board object."""
        self._board = chess.Board()
        self._encoding = encode_board(self._board)
        self._state_list = [self._encoding.copy()]
        self._board_state_counter = Counter(self._encoding.tobytes())
        self._moves = list(self._board.legal_moves)

        self._status = BoardOutcome.UNDECIDED

        # allocate the observation space
        self._observation: Observation = np.array([], dtype=np.uint8)
        self._observe()

    def reset(self) -> None:
        """Reset board position."""
        self._board.reset()
        self._encoding = encode_board(self._board)
        self._state_list = [self._encoding.copy()]
        self._board_state_counter = Counter(self._encoding.tobytes())
        self._moves = list(self._board.legal_moves)
        self._observe()
        self._render()

    def step(self, action: Action) -> None:
        """

        Step the board through one move, encoded by `action`.

        Parameters
        ----------
        action : Action
            Action performed on the board (chess move)
        """
        if self._status not in BoardOutcome.TERMINATED:
            if action != RESIGN:
                self.update_state(action)

                # check for end conditions
                if self._board.is_checkmate():
                    self._status = (
                        BoardOutcome.BLACK
                        if self._board.turn == chess.WHITE
                        else BoardOutcome.WHITE
                    )
                elif (
                    self._board.is_repetition()
                    or self._board.is_fifty_moves()
                    or self._board.is_insufficient_material()
                    or self._board.is_stalemate()
                ):
                    self._status = BoardOutcome.DRAW

                self._observe()

                self._render()
            else:
                self._status = (
                    BoardOutcome.BLACK if self._board.turn == chess.WHITE else BoardOutcome.WHITE
                )
        else:
            msg = "Board is in terminal state, and cannot be stepped."
            raise RuntimeError(msg)

    def _observe(self) -> None:
        """Make the environment reflect the board state and generate an observation."""
        if self._status not in BoardOutcome.TERMINATED:
            # create observation encodings
            self._observation = generate_board_encodings_from_moves(
                self._encoding, self._moves, self._board.turn, self._board_state_counter
            )
        else:
            self._observation = np.array([], dtype=np.uint8)

    def update_state(self, action: Action) -> None:
        """

        Update the state of the environment without generating an observation or rendering.

        Parameters
        ----------
        action : Action
            played action, represented by the index of move to play
        """
        move = self._moves[action]
        self._board.push(move)
        self._moves = list(self._board.legal_moves)
        self._encoding = encode_board(self._board)
        self._state_list.append(self._encoding.copy())
        self._board_state_counter.update(self._encoding.tobytes())

    def _render(self) -> None:
        """Render method for board, empty for base class."""
        return

    @property
    def moves(self) -> list[chess.Move]:
        """

        List of possible moves from the current board state.

        Returns
        -------
        list[chess.Move]
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
        return self._status in BoardOutcome.TERMINATED

    @property
    def winner(self) -> chess.Color | None:
        """

        The winner in the current board state, None if not applicable.

        Returns
        -------
        chess.Color | None
            The winning color, or None if no winner is determined (draw or unfinished game)
        """
        if self._status in BoardOutcome.WON:
            return chess.WHITE if BoardOutcome.WHITE else chess.BLACK
        return None

    @property
    def trajectory(self) -> Trajectory:
        """

        Trajectory of game states as a SetEncoding.

        Returns
        -------
        Trajectory
            shape (n, 8, 8, 18), where n is the index of each successive board state
        """
        if self._status in BoardOutcome.TERMINATED:
            return np.array(self._state_list, dtype=np.uint8)
        msg = "Board is not in terminal state, and does not contain a valid trajectory."
        raise RuntimeError(msg)

    @property
    def observation(self) -> Observation:
        """

        Observation of current board state.

        Returns
        -------
        Observation
            shape (n, 8, 8, 18), where n is the number of valid moves
        """
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


class ASCIIBoard(Board):
    """Board with simple ASCII visualization."""

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

    def _render(self) -> None:
        """Render board display."""
        self._display.display_board(self._board)

    def get_user_input(self) -> chess.Move | None:
        """

        Check if the user has given GUI input, and return the move if possible.

        Returns
        -------
        chess.Move | None
            move played by the user, or None if move not available
        """
        return self._display.get_user_input(self._board)
