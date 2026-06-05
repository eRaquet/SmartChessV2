"""Script to seed all the model strains when beginning a training session."""

import json
from pathlib import Path

from modules.config import PROJECT_PATH
from modules.model import StandardModel

if __name__ == "__main__":
    # define metadata to track model generations
    metadata = {
        "strain_0_curr_gen": 0,
        "strain_1_curr_gen": 0,
        "strain_2_curr_gen": 0,
        "strain_3_curr_gen": 0,
        "strain_4_curr_gen": 0,
        "strain_5_curr_gen": 0,
        "strain_6_curr_gen": 0,
        "strain_7_curr_gen": 0,
    }

    with Path.open(PROJECT_PATH / "data" / "saved_models" / "metadata.json", "w") as file:
        json.dump(metadata, file)

    for i in range(8):
        StandardModel(i, 0, construct=True)
