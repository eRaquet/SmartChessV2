"""Configs for agents."""

from dataclasses import dataclass

from smartchess.config import DEFAULT_CONFIDENCE
from smartchess.model import ModelBase, ModelConfig
from smartchess.types import MoveSource


@dataclass(slots=True, kw_only=True)
class AgentConfig:
    """Base config for the agent."""


@dataclass(slots=True, kw_only=True)
class ModelAgentConfig(AgentConfig):
    """Config for Model Agent."""

    model: ModelBase | ModelConfig
    """
    Model object, or config for model object.
    """

    confidence: float | None = DEFAULT_CONFIDENCE
    """
    Confidence factor to use when choosing moves, default to project wide default value.
    """

    seed: int | None = None
    """
    Seed for RNG.
    """


@dataclass(slots=True, kw_only=True)
class RandomAgentConfig(AgentConfig):
    """Config for random agent."""

    seed: int | None = None
    """
    Optional seed for RNG, default None.
    """


DEFAULT_RANDOM_AGENT_CONFIG = RandomAgentConfig()


@dataclass(slots=True, kw_only=True)
class UIAgentConfig(AgentConfig):
    """Config for a UI agent."""

    move_source: MoveSource
    """
    Move source for UIAgent, likely a GUI.
    """
