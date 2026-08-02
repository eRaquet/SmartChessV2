"""Tools for parsing agents in CLI."""

from dataclasses import dataclass
from enum import StrEnum

import typer

from smartchess.agent import AgentConfig, ModelAgentConfig, RandomAgentConfig, UIAgentConfig
from smartchess.config import PROJECT_PATH
from smartchess.model import InferenceModelConfig, RandomModelConfig
from smartchess.types import MoveSource

from .util import nonnegative_integer


class AgentKind(StrEnum):
    """Agent implementations selectable from the command line."""

    UI = 'ui'
    RANDOM = 'random'
    RANDOM_MODEL = 'random-model'
    MODEL = 'model'


class LogMode(StrEnum):
    """Available game-log destinations."""

    NONE = 'none'
    PRINT = 'print'
    FILE = 'file'


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Validated command-line description of one player."""

    kind: AgentKind
    strain: int | None = None
    generation: int | None = None


def complete_agent(incomplete: str) -> list[str]:
    """Complete agent names and locally available inference models."""
    candidates = ['ui', 'human', 'random', 'random-model', *_available_model_specs()]
    return sorted(candidate for candidate in candidates if candidate.startswith(incomplete))


def parse_agent(value: str) -> AgentSpec:
    """Parse an agent specification from its compact CLI representation."""
    parts = value.lower().split(':')

    match parts:
        case ['ui'] | ['human']:
            return AgentSpec(AgentKind.UI)
        case ['random']:
            return AgentSpec(AgentKind.RANDOM)
        case ['random-model']:
            return AgentSpec(AgentKind.RANDOM_MODEL)
        case ['model', strain]:
            return AgentSpec(AgentKind.MODEL, strain=nonnegative_integer(strain, 'strain'))
        case ['model', strain, generation]:
            return AgentSpec(
                AgentKind.MODEL,
                strain=nonnegative_integer(strain, 'strain'),
                generation=nonnegative_integer(generation, 'generation'),
            )
        case _:
            msg = (
                f'invalid agent {value!r}; expected ui, human, random, random-model, '
                'model:STRAIN, or model:STRAIN:GENERATION'
            )
            raise typer.BadParameter(msg)


def create_agent_config(
    agent: AgentSpec,
    interface: MoveSource | None,
    confidence: float | None,
) -> AgentConfig:
    """Translate a CLI agent specification into an application config."""
    match agent.kind:
        case AgentKind.UI:
            if interface is None:
                msg = 'UI agent config requested without a move source.'
                raise RuntimeError(msg)
            return UIAgentConfig(move_source=interface)

        case AgentKind.RANDOM:
            return RandomAgentConfig()

        case AgentKind.RANDOM_MODEL:
            return ModelAgentConfig(
                model=RandomModelConfig(),
                confidence=confidence,
            )

        case AgentKind.MODEL:
            if agent.strain is None:
                msg = 'Inference model config requested without a strain.'
                raise RuntimeError(msg)

            return ModelAgentConfig(
                model=InferenceModelConfig(
                    strain=agent.strain,
                    generation=agent.generation,
                ),
                confidence=confidence,
            )

        case _:
            msg = 'Unsupported agent kind.'
            raise ValueError(msg)


def _available_model_specs() -> list[str]:
    """Return inference-model specifications available in the local model store."""
    saved_models = PROJECT_PATH / 'data' / 'saved_models'
    candidates: list[str] = []

    for strain_directory in saved_models.glob('strain_*'):
        strain_text = strain_directory.name.removeprefix('strain_')
        if not strain_text.isdigit():
            continue

        candidates.append(f'model:{strain_text}')
        prefix = f'strain_{strain_text}_gen_'
        for model_path in strain_directory.glob(f'{prefix}*.keras'):
            generation_text = model_path.stem.removeprefix(prefix)
            if generation_text.isdigit():
                candidates.append(f'model:{strain_text}:{generation_text}')

    if not candidates:
        candidates.append('model:')

    return candidates
