"""Script to seed all the model strains when beginning a training session."""

import json
from pathlib import Path

from modules.config import PROJECT_PATH, STRAIN_COUNT
from modules.model import StandardModel

if __name__ == "__main__":
    saved_models_path = PROJECT_PATH / "data" / "saved_models"
    saved_models_path.mkdir(parents=True, exist_ok=True)

    # define metadata to track model generations
    metadata = {}
    for i in range(STRAIN_COUNT):
        metadata[f"strain_{i}_curr_gen"] = 0

    with Path.open(saved_models_path / "metadata.json", "w") as file:
        json.dump(metadata, file)

    for i in range(STRAIN_COUNT):
        StandardModel(i, 0, construct=True)
