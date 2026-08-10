"""Seed the smartchess models."""

import json
from pathlib import Path

from smartchess.config import STRAIN_COUNT
from smartchess.model import InferenceModelConfig, create_model

from .config import SeedConfig


def seed(config: SeedConfig) -> None:
    """

    Seed a set of starting smartchess models.

    Parameters
    ----------
    config : SeedConfig
        config for seeding
    """
    config.path.mkdir(parents=True, exist_ok=True)

    # define metadata to track model generations
    metadata = {}
    for i in range(STRAIN_COUNT):
        metadata[f'strain_{i}_curr_gen'] = 0

    with Path.open(config.path / 'metadata.json', 'w') as file:
        json.dump(metadata, file)

    for i in range(STRAIN_COUNT):
        model_config = InferenceModelConfig(
            strain=i, generation=0, construct=True, path=config.path
        )
        create_model(model_config)
