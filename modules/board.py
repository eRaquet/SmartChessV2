"""Module to define the board environment for training and playing."""

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
    encode_board_obs,
    generate_board_encodings_from_moves,
)


class Board(gym.Env):
    """Chess board gym environment with generation of valid moves and creation of observations."""

    def __init__(self, render_mode: DisplayMode = DisplayMode.NONE) -> None:
        """Initiate a board object."""
        self.board = chess.Board()
        self.encoding = encode_board_obs(self.board.piece_map(), self.board.turn)
        self.moves = list(self.board.legal_moves)

        # define the action and observation space
        self.action_space = spaces.Discrete(218)  # index of chosen move
        self.observation_space = spaces.Dict(
            {
                "encoding": spaces.Box(0, 1, shape=(218, 8, 8, 12), dtype=np.uint8),
                "castling_rights": spaces.Box(0, 1, shape=(218, 4), dtype=np.uint8),
                "is_draw": spaces.Box(0, 1, shape=(218,), dtype=np.uint8),
                "num_moves": spaces.Discrete(218),
            }
        )

        # allocate the observation space
        self.observation: Observation = {
            "encoding": np.zeros((218, 8, 8, 12), dtype=np.uint8),
            "castling_rights": np.zeros((218, 4), dtype=np.uint8),
            "is_draw": np.zeros((218,), dtype=np.uint8),
            "num_moves": 0,
        }

        self.saved_encoding = np.zeros((218, 8, 8, 12), dtype=np.uint8)

        self.render_mode = render_mode
        if self.render_mode is DisplayMode.GUI:
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
        self.encoding = encode_board_obs(self.board.piece_map(), self.board.turn)
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

        # clear old observation
        self.clear_observation()

        if num_moves != 0:
            self.observation["num_moves"] = num_moves

            # create observation encodings
            self.observation["encoding"][0:num_moves] = generate_board_encodings_from_moves(self.encoding, self.moves, self.board.turn)

            # check for castling rights and draw rights
            for i, move in enumerate(self.moves):
                self.board.push(move)

                self.observation["castling_rights"][i] = [
                    bool(self.board.castling_rights & (chess.BB_A1 if self.board.turn == chess.WHITE else chess.BB_A8)),
                    bool(self.board.castling_rights & (chess.BB_H1 if self.board.turn == chess.WHITE else chess.BB_H8)),
                    bool(self.board.castling_rights & (chess.BB_A8 if self.board.turn == chess.WHITE else chess.BB_A1)),
                    bool(self.board.castling_rights & (chess.BB_H8 if self.board.turn == chess.WHITE else chess.BB_H1)),
                ]

                self.observation["is_draw"][i] = self.board.is_repetition() or self.board.is_fifty_moves()

                self.board.pop()

            self.saved_encoding[0:num_moves] = self.observation["encoding"][0:num_moves]

    def update_state(self, index: int) -> None:
        """

        Update the state of the environment without generating an observation.

        Parameters
        ----------
        index : int
            index of move to play
        """
        move = self.moves[index]
        self.board.push(move)
        self.moves = list(self.board.legal_moves)
        self.encoding = np.flip(self.saved_encoding[index], axis=(0, 2)).copy()

    def clear_observation(self) -> None:
        """Clear the old observation fields."""
        old_num_moves = self.observation["num_moves"]
        self.observation["encoding"][0:old_num_moves].fill(0)
        self.observation["castling_rights"][0:old_num_moves].fill(0)
        self.observation["is_draw"][0:old_num_moves].fill(0)
        self.observation["num_moves"] = 0

    def render(self) -> None:
        """Render the board object according to the set render mode."""
        if self.render_mode is DisplayMode.NONE:
            return
        if self.render_mode is DisplayMode.ASCII:
            print("\033[2J\033[H", end="")
            print("-" * 15)
            print(self.board)
            print("-" * 15)
        if self.render_mode is DisplayMode.GUI:
            self.display.display_board(self.board, self.board.piece_map())
