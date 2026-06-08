"""Python file that contains various project-wide configuration features."""

from pathlib import Path

# path object to project directory
PROJECT_PATH = Path(__file__).parent.parent

# number of model strains
STRAIN_COUNT = 4

# parameterize the model
MODEL_PARAMS = {
    # shuffling layer, that doesn't change the width of the board, but adds way more channels
    # (note that padding is "same", not "valid")
    "1": {
        "filters": 256,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "same",
        "data_format": "channels_last",
    },
    # condensing layer, that squishes the width of the board down to a single value across many
    # channels.
    "2": {
        "filters": 128,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
    "3": {
        "filters": 64,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
    "4": {
        "filters": 32,
        "kernal_size": 4,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
}
