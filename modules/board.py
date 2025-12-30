"""Module to define the board environment for training and playing."""

from collections import Counter

import chess
import gym
import numpy as np
from gym import spaces

from modules.chess_types import (
    Action,
    BoardInfo,
    DisplayMode,
    IsOver,
    MoveReward,
    Observation,
)
from modules.display import Display
from modules.tools import (
    encode_board,
    generate_board_encodings_from_moves,
)


class Board(gym.Env):
    """Chess board gym environment with generation of valid moves and creation of observations."""

    def __init__(self, rendering_mode: DisplayMode = DisplayMode.NONE) -> None:
        """Initiate a board object."""
        self.board = chess.Board()
        self.encoding = encode_board(self.board)
        self.board_state_counter = Counter(self.encoding.tobytes())
        self.moves = list(self.board.legal_moves)

        # define the action and observation space as the
        # index of chosen move, noting that 218 is the
        # maximum number of moves possible in a valid chess position
        self.action_space = spaces.Discrete(218)
        self.observation_space = spaces.Sequence(spaces.MultiBinary([8, 8, 18]))

        # allocate the observation space
        self.observation = np.array([])

        self.rendering_mode = rendering_mode
        if self.rendering_mode is DisplayMode.GUI:
            self.display = Display()

    def reset(self) -> tuple[Observation, BoardInfo]:  # type: ignore[override]
        """

        Reset board position.

        Returns
        -------
        Observation
            First observation
        """
        self.board.reset()
        self.encoding = encode_board(self.board)
        self.board_state_counter = Counter(self.encoding.tobytes())
        self.moves = list(self.board.legal_moves)
        self.generate_observation()
        info: BoardInfo = {}
        self.render()
        return self.observation, info

    def step(self, action: Action) -> tuple[Observation, MoveReward, IsOver, IsOver, BoardInfo]:  # type: ignore[override]
        """

        Step the board through one move, encoded by `action`.

        Parameters
        ----------
        action : Action
            Action performed on the board (chess move)

        Returns
        -------
        tuple[Observation, MoveReward, IsOver, BoardInfo]
            Information about board:
                Observation: board state after move
                float: immediate reward from action
                bool: whether the position is a checkmate
                bool: whether the position is a draw
                dict: info about the game
        """
        self.update_state(action)
        self.generate_observation()
        terminated = self.board.is_checkmate()
        truncated = (
            self.board.is_repetition() or self.board.is_fifty_moves() or self.board.is_insufficient_material() or self.board.is_stalemate()
        )
        reward = 0.0
        self.render()

        return (
            self.observation,
            reward,
            terminated,
            truncated,
            {},
        )

    def generate_observation(self) -> None:
        """Make the environment reflect the board state and generate an observation."""
        num_moves = len(self.moves)

        if num_moves != 0:
            # create observation encodings
            self.observation = generate_board_encodings_from_moves(self.encoding, self.moves, self.board.turn, self.board_state_counter)

    def update_state(self, action: Action) -> None:
        """

        Update the state of the environment without generating an observation.

        Parameters
        ----------
        action : Action
            played action, represented by the index of move to play
        """
        move = self.moves[action]
        self.board.push(move)
        self.moves = list(self.board.legal_moves)
        self.encoding = encode_board(self.board)
        self.board_state_counter.update(self.encoding.tobytes())

    def render(self) -> None:
        """Render the board object according to the set render mode."""
        if self.rendering_mode is DisplayMode.NONE:
            return
        if self.rendering_mode is DisplayMode.ASCII:
            print("\033[2J\033[H", end="")
            print("-" * 15)
            print(self.board)
            print("-" * 15)
        if self.rendering_mode is DisplayMode.GUI:
            self.display.display_board(self.board, self.board.piece_map())
