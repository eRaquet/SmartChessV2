"""Models Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .model_base import ModelBase as ModelBase
from .random_model import RandomModel as RandomModel

if TYPE_CHECKING:
    from .inference_model import InferenceModel


def __getattr__(name: Literal['InferenceModel']) -> type[InferenceModel]:
    if name == 'InferenceModel':
        try:
            from .inference_model import InferenceModel
        except ModuleNotFoundError as error:
            if error.name in {'keras', 'jax', 'mlx'}:
                msg = (
                    'InferenceModel requires an inference backend.  '
                    'Install smartchess-v2[jax] or smartchess-v2[mlx]'
                )
                raise ImportError(msg) from error
            raise

        return InferenceModel

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
