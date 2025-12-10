"""Module for handling the board and managing different agents."""

import chess
import gym
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from modules.board import Board
from modules.chess_types import (
    Action,
    BoardInfo,
    DisplayMode,
    GameInfo,
    IsOver,
    MappedAction,
    MappedObservation,
    MoveReward,
    Observation,
)
from modules.display import Display
from modules.tools import (
    encode_board_obs,
    generate_board_encodings_from_moves,
)


class MultiAgentChess(MultiAgentEnv):
    """

    Turn-based multi-agent wrapper around the single-agent Board env.

    Only the agent whose turn it is will receive an observation and must act.

    """

    def __init__(self, render_mode: DisplayMode = DisplayMode.NONE) -> None:
        self.board = Board(render_mode=render_mode)
        self.agent_ids = ["white", "black"]
        self.current_agent = "white"

        self.observation_spaces = {
            "white": self.board.observation_space,
            "black": self.board.observation_space,
        }
        self.action_spaces = {
            "white": self.board.action_space,
            "black": self.board.action_space,
        }

    def reset(self) -> tuple[MappedObservation, GameInfo]:  # type: ignore[override]
        """Reset the game."""
        obs, _ = self.board.reset()
        self.current_agent = "white"
        # only the current agent gets an observation
        obs_map: MappedObservation = {"white": obs} if self.current_agent == "white" else {"black": obs}
        info: GameInfo = {}
        return obs_map, info

    def step(self, action_dict: MappedAction) -> tuple[MappedObservation,]:
        # apply only the current agent's action
        action = action_dict[self.current_agent]
        obs, reward, terminated, truncated, info = self.board.step(action)

        done = terminated or truncated

        # prepare output dicts
        rewards = {aid: 0.0 for aid in self.agent_ids}
        rewards[self.current_agent] = reward

        if done:
            obs_dict = {}
            dones = {aid: True for aid in self.agent_ids}
            dones["__all__"] = True
            infos = {aid: info for aid in self.agent_ids}
        else:
            # switch turn
            self.current_agent = self._opponent(self.current_agent)
            obs_dict = {self.current_agent: obs}
            dones = {aid: False for aid in self.agent_ids}
            dones["__all__"] = False
            infos = {self.current_agent: info}

        return obs_dict, rewards, dones, infos

    def _opponent(self, agent_id: str) -> str:
        return "black" if agent_id == "white" else "white"
