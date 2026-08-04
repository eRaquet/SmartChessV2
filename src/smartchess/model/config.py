"""Config for model class."""

from dataclasses import dataclass
from pathlib import Path

from smartchess.config import PROJECT_PATH


@dataclass(slots=True, kw_only=True)
class ModelConfig:
    """Base config for model."""


@dataclass(slots=True, kw_only=True)
class RandomModelConfig(ModelConfig):
    """Config for random model."""

    seed: int | None = None
    """
    Optional seed value for RNG, None by default.
    """


DEFAULT_RANDOM_MODEL_CONFIG = RandomModelConfig()


@dataclass(slots=True, kw_only=True)
class InferenceModelConfig(ModelConfig):
    """Config for inference model."""

    strain: int
    """
    Strain number of the inference model.
    """

    generation: int | None = None
    """
    Generation of the inference model within it's strain, if None use the most recent generation,
    default None.
    """

    construct: bool = False
    """
    Whether to construct the model from scratch and save to memory, or load it from memory,
    by default False.
    """

    path: Path = PROJECT_PATH / 'artifacts' / 'saved_models'
    """
    Path to the folder in which to the model is stored.
    """
