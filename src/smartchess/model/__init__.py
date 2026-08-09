"""Models Module."""

from __future__ import annotations

from smartchess.capabilities import require_capability

from .config import InferenceModelConfig as InferenceModelConfig
from .config import ModelConfig as ModelConfig
from .config import RandomModelConfig as RandomModelConfig
from .model_base import ModelBase as ModelBase  # noqa: TC001


def create_model(config: ModelConfig) -> ModelBase:
    """Create a model from the provided config."""
    if type(config) is RandomModelConfig:
        from .random_model import RandomModel

        return RandomModel(config)

    if type(config) is InferenceModelConfig:
        from smartchess.config import KERAS_BACKEND

        if KERAS_BACKEND == '':
            msg = 'No keras backend specified or available.'
            raise RuntimeError(msg)

        require_capability(KERAS_BACKEND)

        from .inference_model import InferenceModel

        return InferenceModel(config)
    msg = 'Unsupported Model Config Type'
    raise ValueError(msg)
