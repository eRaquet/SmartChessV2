"""Agents Module."""

from .agent_base import AgentBase as AgentBase
from .config import AgentConfig as AgentConfig
from .config import ModelAgentConfig as ModelAgentConfig
from .config import RandomAgentConfig as RandomAgentConfig
from .config import UIAgentConfig as UIAgentConfig


def create_agent(config: AgentConfig) -> AgentBase:
    """

    Create an agent from the provided config.

    Parameters
    ----------
    config : AgentConfig
        Config to create the agent from.

    Returns
    -------
    AgentBase
        Created agent.
    """
    if type(config) is RandomAgentConfig:
        from .random_agent import RandomAgent

        return RandomAgent(config)
    if type(config) is ModelAgentConfig:
        from .model_agent import ModelAgent

        return ModelAgent(config)
    if type(config) is UIAgentConfig:
        from .ui_agent import UIAgent

        return UIAgent(config)
    msg = 'Unsupported agent config.'
    raise ValueError(msg)
